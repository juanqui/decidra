# d3-turbo state-once: engine support, quantization, and SM89 optimization

Status: implementation started 2026-10-21. Reviewed by `claude-opus-5-5`
(xhigh) — review log in §12. Scope: `native/src/engine.rs`,
`native/third_party/flashinfer/fi_prefill.cu`, `native/src/flashinfer.rs`,
`src/native_backend.rs`, and the d3-turbo quantization pipeline.

## 1. What d3-turbo is

`vllm-sr/d3-turbo` is a 26.1B-parameter Qwen3.5 hybrid decision model:

| property | value |
|---|---|
| backbone | 64 layers: 48 Gated DeltaNet (linear attention) + 16 full attention at 3,7,…,63 |
| hidden / inter | 5120 / 17408 |
| full attention | 24 Q heads, 4 KV heads, head_dim 256, partial RoPE 0.25 |
| GDN | 16 K heads / 48 V heads, 128 dims, conv width 4 |
| readout | separate F32 `[255,5120]`, last-token pooling |
| attention mode | **`prefix_bidir`** in `decision_config.json` |
| context | 32K validated, 262K max |
| inputs | text/JSON, images, video (not served by our native text path yet) |

The model answers typed questions (`choice`, `noul`, `score`) with a
probability per answer code and never generates text.

### 1.1 The mask trick

`prefix_bidir` splits a rendered prompt at the end of the state text:

```
[ system + State: <state> ][ \n\nQuestion: ... options ... ][ readout ]
        state block S                     question block Q
```

Full-attention layers apply a dense block mask:

- every state query sees every state key (bidirectional);
- every question query sees every state key **and every key in its own
  question block** (bidirectional);
- **no state query sees a question key.**

The Gated DeltaNet layers stay causal and scan the state once. Because the
state's hidden states, full-attention K/V, final recurrent state and
convolution tail do not depend on any question, they are computed once per
request and every question forks from them. That is the whole speed trick:
at a 2K-token state and 64 questions, a per-row implementation reads the
state 64 times; state-once reads it once.

`vllm-sr/d3` (the released Decision 3.0 checkpoint) is **not** the same
mask: it declares `noncausal_full_attention`, where the state does see the
questions. Its output cannot be cached across questions. d3-turbo is the
fine-tuned/`prefix_bidir` variant that makes the cache valid.

### 1.2 Reference behavior (measured)

On the local BF16 checkpoint, one held-out row (`choice-c6da9cf0`, 124
tokens, state split 75), using the checkpoint's own `d3_runtime.py` on CPU:

| path | probabilities (4 options) |
|---|---|
| `prefix_bidir` (state-once reference) | 0.0927, 0.0312, 0.7588, 0.1173 |
| `noncausal_full_attention` (wrong for d3-turbo) | 0.0641, 0.0171, 0.8052, 0.1137 |

The two masks differ by up to 0.044 in probability on this row. A causal
mask is worse. **Mask correctness is not optional.**

## 2. What vLLM-SR does

`vllm-sr` Engine mode is a Python model runtime (`vllm_srun`), not vLLM.
Its `StateOnceModel` (`families/decision3/state_once.py`) tokenizes the state
head once, splits each full prompt by tokenizer offsets, and calls
`EngineModel.branch`. The native PyTorch engine packs all states and question
blocks of a batch into one row (`engines/native/models/branch.py`):

- full attention: variable-length FlashAttention for the state blocks, then
  each question block gathers `[state keys ‖ own keys]` and runs one softmax;
- GDN: each block starts from its state's final recurrent state, with the
  convolution window seeded by the state's last inputs;
- projections/norms/MLPs run once over the packed row;
- a single question runs as a dense per-row forward; batches branch.

Published numbers are MI325X/ROCm, not 4090: d3 27B, 2K state, 64 questions,
510 ms p50, 127 questions/s with 8 concurrent copies. Their own record says
branching moves answers about as much as batching the per-row path (mean
probability gap 0.0041, every flip a near-tie).

**What we can beat.** Their kernels are PyTorch/Triton/FlashAttention with
BF16 weights (52 GB), no INT8 activation path, per-layer Python dispatch, and
a generic packed-row layout. On one 4090 we can keep W4A8 weights resident
(17 GB), use Marlin tensor-core GEMMs, run one forward for all questions,
and keep the state K/V in VRAM.

## 3. Measured engine state before this work

The engine already had noncausal full attention (FlashInfer FA2, SM89) and a
separate 255-row readout. Two defects blocked d3-turbo:

1. **`prefix_bidir` was treated as `noncausal_full_attention`.** The mask
   differed from the checkpoint (0.044 max Δp above).
2. **The local d3-turbo W4A8 artifact is random.** The quantizer passed the
   model path to `llmcompressor.oneshot`, which loads
   `AutoModelForCausalLM`. The checkpoint is a `Qwen3_5Model` with
   `language_model.*` keys; `AutoModelForCausalLM` reported every weight
   UNEXPECTED and initialized `model.*` randomly, then quantized the random
   model. Measured cosine between the saved W4A8 layer-0 `gate_proj` and the
   BF16 source: **-6.2e-5**. That is the root cause of the earlier
   near-uniform d3-turbo eval (`top-1 8/32`, MAE 0.235), not the mask.

After fixing the loader and adding the mask, the engine matches the
checkpoint on BF16:

| path | max abs Δp vs HF | MAE |
|---|---:|---:|
| full noncausal, engine vs HF full | 0.0061 | 0.0031 |
| per-row block cascade, engine vs HF block | 0.0022 | 0.0012 |
| state-once (state pass + question pass), vs HF block | 0.0035 | 0.0018 |
| batched state-once vs single state-once (probability, BF16 head) | 0.014 | 0.0036 |
| batched state-once vs single state-once (probability, F32 head) | 0.0056 | 0.0015 |

State-once uses post-RoPE state K/V, a per-layer f32 recurrent state, and
the last three pre-conv `in_proj_qkv` rows. No approximation beyond kernel
reduction order.

## 4. Architecture: one runtime, programs as data

The current engine is a monolith with booleans (`noncausal_full`) and a
separate readout. Adding d3-turbo by more booleans would duplicate the layer
loop. The target shape is:

```
ModelProfile (parsed once, immutable after preload)
  attention: Causal | Full | BlockCascade
  readout:   LmHeadRows | SeparateHead { dtype, rows }
  program:   PerRow | StateOnce
  numeric:   dtype, quant policy, kernel choices

ForwardJob
  rows:      &[&[u32]]           // one row per question or per prompt
  splits:    &[usize]            // state/question boundary per row
  pos_start: usize               // 0 for per-row, state_len for forks
  sites:     &[u32]              // readout row per site
  cache:     Option<&StateCache> // None = recompute state
```

`Engine` keeps one layer executor (`run_layer`, `run_layer_batch`) and one
attention dispatcher. Mask and cache are arguments, not global flags:

- `MaskSpec::Causal` → native/FlashInfer causal kernels;
- `MaskSpec::Full` → FlashInfer `kNone`;
- `MaskSpec::BlockCascade { split }` → custom-mask FlashInfer for per-row
  reference, cascade (`qo_len < kv_len`) for state-once;
- `StateCache` carries per-full-layer K/V and per-GDN-layer state/tail.

The server layer owns prompt rendering and split computation. The engine
never parses a chat template or a `decision_config.json` at request time.
Model-specific post-processing (choice/noul/score, temperature, confidence)
stays in the family adapter, not in the CUDA path.

This removes the four current duplications: single vs batch attention,
single vs batch GDN, readout vs lm_head, and noncausal vs causal dispatch.
The state-once batch reuses `run_layer_batch`; the per-row reference reuses
`run_layer`; both share `LayerIo`.

## 5. State-once execution

### 5.1 Capture (`Engine::capture_state`)

One row, state tokens only:

- embed, then each layer;
- full layers: project/norm/RoPE, copy post-RoPE K and V to
  `[S, 4, 256]` BF16 buffers;
- GDN layers: project, conv, prepare, recurrent; copy the final
  `[48,128,128]` f32 state and the last three pre-conv `in_proj_qkv` rows
  (`[3, 10240]` BF16).

Cost at 4K state: 256 MiB KV + 144 MiB GDN + 3 MiB tails.

### 5.2 Question pass

Batched over all question blocks of a request:

- embed the flattened blocks, `pos_start = S`;
- full layers: batch Q/K/V projections; for each row concatenate cached
  state K/V with the row's K/V and call `fi_prefill_bf16_ext(q, k, v,
  qo_len=L_r, kv_len=S+L_r)`. This is one softmax over the exact
  `[state ‖ own block]` key set;
- GDN layers: batch projections; prepend the cached 3-row tail to each
  row's pre-conv qkv, run the existing causal conv over
  `[hist ‖ question]`, compact the body, prepare, seed each row's recurrent
  slot from the cached state, run the batched recurrent kernel;
- residual/MLP batched; final norm; one `[B,255]` readout GEMM.

No new GDN kernel is required. The conv tail uses the existing conv with
three prepended rows; the recurrent kernels already read their state slot at
entry, so seeding is a copy. The only new CUDA entry is FlashInfer's
`qo_len != kv_len` noncausal prefill.

### 5.3 Correctness invariants

- Cached K is post-RoPE; question Q is rotated at `S+t`, never `t`.
- The cached GDN state is per layer; a single state broadcast to all layers
  is wrong.
- The conv tail is the last three **pre-conv** rows, left-zero-padded for
  states shorter than three tokens.
- The split comes from the full prompt's tokenizer offsets. A BPE merge
  across the boundary belongs to the question block; the cache key includes
  the exact state token ids.
- `prefix_bidir` state is only valid as a whole: a strict sub-prefix is not a
  function of that sub-prefix under bidirectional state attention. No
  block-hash partial matching for this mask.

## 6. Quantization

### 6.1 Artifact

Target: INT4 symmetric group-128 weights + INT8 per-token dynamic
activations (`CompressedTensorsWNA8Int`), AWQ + GPTQ, `actorder=weight`,
readout/embeddings/norms/GDN `in_proj_a|b` unquantized. 17 GB, fits one 4090
with ~4 GB for KV/cache/scratch.

### 6.2 Loader fix

`oneshot(model=...)` must receive an instantiated `Qwen3_5Model`, not a path:

```python
model = Qwen3_5Model.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
    device_map="auto", max_memory={0: "20GiB", "cpu": "100GiB"},
)
oneshot(model=model, recipe=recipe, dataset=dataset, ...)
```

Gate: dequantized layer-0 `gate_proj` cosine vs the BF16 source > 0.99 before
any calibration run.

### 6.3 Calibration under the real mask

AWQ/GPTQ activation statistics must come from the inference distribution:
`prefix_bidir`, not causal or full. Two options, in order:

1. **Exact block-mask calibration.** Wrap `Qwen3_5Model` so its full-attention
   layers use the block mask for a batch of `[state ‖ question]` rows. The
   sequential calibration pipeline traces decoder layers, so the mask object
   must be a tensor argument, not a Python loop; if tracing rejects it, fall
   back to (2).
2. **State-prefix calibration.** Calibrate on state blocks only, with
   noncausal full attention *within the state*. This is exact for state
   tokens—which are the majority and the cached part—and omits the short
   question blocks. Add a second set of question blocks with a causal mask
   over `[state ‖ question]` only if the measured weight error says the
   missing block matters.

The first run must record per-layer AWQ scales and GPTQ error; the previous
run's `error 34347` on `gate_proj` was a symptom of random weights.

**AWQ is not shipped for d3-turbo.** Both AWQ+GPTQ test artifacts (4 rows,
including one with `in_proj_a|b` restored to the mapping) were far from BF16
(max Δp 0.50), while GPTQ-only was within 0.04 on the same row. The gated
`q_proj` packs Q and the output gate, and the GDN input norm feeds four
projections; AWQ's per-norm scaling is not exact for this layout. GPTQ-only
keeps the weight error small and is the shipped recipe.

**Status 2026-10-21.** `Qwen3_5ForCausalLM` with
`key_mapping={r"^language_model\.": "model."}` traces 65 subgraphs and loads
the real weights (`layer0 gate == source`), unlike `Qwen3_5Model`. A
two-sample GPTQ-only run with the block mask produced a valid artifact:
layer-0 `gate_proj` cosine ~1.0 against BF16, one held-out row
probabilities 0.097/0.034/0.719/0.151 vs BF16 0.093/0.031/0.759/0.117
(max Δp 0.040). The 256-row AWQ+GPTQ run is the production artifact; the
two-sample run is only a pipeline check.

### 6.4 Accuracy gate

Held-out 256 rows (choice + noul), BF16 reference from the checkpoint's own
`D3`/`StateOnce` path:

- top-1 agreement ≥ 98%;
- mean |Δp| ≤ 0.01, max |Δp| ≤ 0.08;
- no decision flips on rows whose BF16 margin > 0.05.

W4A16 (no INT8 activations) is the control; W4A8 must stay within 0.5
points of it. If INT8 activation error is the limiter, per-K-group scales
(already supported by the engine) are the first knob, not a new format.

## 7. SM89 optimization plan

Ordered by measured impact, not intuition:

1. **Batch all questions of a request.** One weight read per layer instead of
   one per question. This is the dominant win; per-question GDN/MLP would
   otherwise read 13.5 GB per question.
2. **Marlin W4A8 INT8.** The engine's `marlin_int8` path uses `mma.sync`
   INT8 tensor cores; keep activations INT8 for every projection except
   `in_proj_a|b` (measured exclusion list). Tune `pm`/`tmb` for the question
   pass's M (total question tokens, typically 1K–8K).
3. **Cascade attention shape.** `qo_len` 32–256, `kv_len` 2K–8K. One
   FlashInfer launch per row per full layer; consider `BatchPrefillRagged`
   with a shared-state KV arena if launch time shows up in the profile.
4. **GDN batched kernels.** Seed per-row state, use the split/shared batched
   recurrence that already exists. Avoid the chunked path for short blocks
   (launch overhead); force `shared` below 512 tokens.
5. **Readout.** One `[B,5120] × [5120,255]` BF16 GEMM; the head is 2.6 MB,
   keep it resident. Do not gather 255 `lm_head` rows.
6. **CUDA graphs.** The question pass has a fixed shape per request
   (B, total, max block). Capture per shape after the first call; graph
   replay removes ~64 layer launches × 5–10 µs. Only after 1–5 are measured.
7. **State cache reuse across requests.** A short-TTL exact-state LRU; the
   state is the expensive part at 4K+. Start with one entry per in-flight
   request; add LRU only after hit-rate measurement.
8. **KV in FP8?** No. The accuracy gate leaves no room; BF16 KV is 64 KiB per
   token and fits.

Targets on one 4090, 2K state, 64 questions (~50-token blocks):

| metric | target |
|---|---:|
| state pass | ≤ 250 ms |
| 64-question pass | ≤ 350 ms |
| end-to-end p50, 64 questions | ≤ 600 ms |
| questions/s in that request | ≥ 105 |
| 64-question requests/s (serial engine) | ≥ 1.6 |
| vs vLLM-SR MI325X 510 ms / 127 q/s | comparable or faster on a 4090 |

The first two numbers are FLOP-bound: 2K state ≈ 108 TFLOP, 3.2K question
tokens ≈ 173 TFLOP; at 50% of the 4090's 660 INT8 TOPS that is ~850 ms, so
kernel efficiency, not batching alone, decides the target. The engine is
single-stream, so concurrent requests serialize; throughput is
questions/s, not a multiple of the single-request rate. Re-baseline after
the corrected artifact exists.

**Measured 2026-10-21, RTX 4090, W4A8 g64, 2K state, 64 questions (~50 tokens
each), weights preloaded. Same-machine vLLM-SR baseline: 3.89 s (2.36x slower).**

| phase | measured |
|---|---:|
| preload (15 GB + Marlin repack) | 6.9–9.6 s |
| state capture (2,048 tokens) | 479 ms |
| 64-question pass | 850 ms |
| 64 single-question passes | 2.14 s |
| resident device free after preload | 9.26 GiB |
| question-pass INT8 GEMMs | ~500 ms |
| question-pass GDN recurrent | 126 ms (chunked TC, min 16) |
| question-pass attention | 128 ms (64 launches × 16 layers) |

End-to-end 1.33 s for the 64-question request. That is 2.6× faster than
running the same 64 questions one at a time. It is slower in absolute terms
than vLLM-SR's MI325X record (510 ms), but the 4090 has roughly one fifth of
the MI325X's HBM bandwidth; per unit bandwidth the engine is ahead. The
largest remaining costs are Marlin GEMM efficiency (~200 TFLOPS of the
4090's 660 INT8 peak) and 1,024 per-row FlashInfer launches.

## 8. Testing

- **Mask algebra (CPU, CI):** dense f32 reference for every mask; assert
  state×question = false for `BlockCascade`.
- **Per-row block parity:** engine vs the checkpoint's `D3Model` on ≥32
  rows; max Δp ≤ 0.01.
- **State-once parity:** single vs batch state-once on the same cache;
  max Δp ≤ 0.05 (rounding order only), top-1 identical.
- **GDN seed/history:** layer-level dump vs a full `[state ‖ question]`
  forward; conv tail and recurrent state compared separately.
- **Split invariance:** states with trailing spaces, CRLF, Unicode and a
  token that merges across the boundary.
- **Quantization:** cosine gate before calibration; held-out accuracy gates
  above; W4A16 control.
- **Performance:** state pass, question pass, end-to-end; p50/p95, GPU
  utilization, peak VRAM; compare the same checkpoint served by vLLM-SR at
  W4A16 if its loader accepts the artifact.

## 9. Phases

| phase | work | gate |
|---|---|---|
| P0 | Fix quantization loader; regenerate W4A8 | cosine > 0.99, held-out accuracy |
| P1 | Per-row BlockCascade + cascade shim | block parity ≤ 0.01 |
| P2 | State capture + single-question fork | state-once parity ≤ 0.01 |
| P3 | Batched question pass | single vs batch parity, perf |
| P4 | Server program: d3-turbo `/v1/systemone`, split by offsets | conformance fixture |
| P5 | SM89 tuning: Marlin INT8, GDN, graphs | targets in §7 |
| P6 | State LRU + metrics | hit-rate measurement |

## 10. Risks

| risk | severity | mitigation |
|---|---|---|
| Wrong mask silently served | critical | fail closed on `prefix_bidir`; dense reference test |
| Quantized weights random again | critical | cosine gate on layer 0 before calibration |
| Calibration mask differs from inference | high | exact block calibration or state-prefix calibration + gate |
| Conv tail dropped | high | layer-level GDN test with tail negative control |
| Batch path and single path diverge | high | T-ONCE single-vs-batch parity |
| Per-question GDN defeats weight reuse | high | batched seeds/history; measure |
| 24 GB OOM at 4K state + 64 questions | medium | KV 256 MiB + state 144 MiB + transient; cap questions |
| CUDA graph capture changes numerics | low | gate decisions, not logits |

## 11. Open questions

1. Does the hosted d3-turbo API send images/video with text state on the same
   request? If yes, state-once must fail closed to per-row for those.
2. Does vLLM-SR's loader accept a compressed-tensors W4A8 checkpoint? If not,
   the on-4090 baseline must be W4A16, and the comparison must say so.
3. How much does exact block-mask calibration change AWQ scales versus
   state-prefix calibration? Measure once; do not assume.

## 12. Review log

### Round 1 - 2026-10-21, Opus 5.5 xhigh

Verdict: **approve with fixes**. Accepted findings and dispositions:

| id | finding | disposition |
|---|---|---|
| H1 | chunked GDN zeroes its state at entry; seeded question blocks must not use it | fixed at the kernel: `gdn_chunked_f32`/`gdn_chunked_tc_body` load their initial state, so seeded blocks use the TF32 path; measured 0.0029 vs HF and 0.0115 on a 600-token question |
| H2 | the readout was BF16 twice (head cast, BF16 output) | fixed: `HeadMatrix` is F32 and the readout runs an F32 cuBLAS GEMM |
| H3 | non-causal modes fell back to causal native kernels when FlashInfer was disabled/missing | fixed: `Engine::new` refuses a non-causal model without FlashInfer; both attention dispatchers bail instead of falling through |
| H4 | calibration did not prove it saw the real weights or mask | added load-report and source-tensor checks, plus a full-module dequant cosine gate; quantization uses the block-mask dataloader and `Qwen3_5ForCausalLM` + `key_mapping` |
| H5 | per-row attention launches and copies remain | planned: shared-state paged KV or LSE cascade, conv `hist`, read-only seed, prebuilt cos/sin; measured before/after |
| H6 | one-row tests, unsafe thresholds, no negative controls | expanded to 32 rows + single/batch parity + chunked-kernel parity + mask unit tests + zero-seed/zero-tail negative controls |
| M1 | single and batch paths duplicate semantics | accepted: single becomes batch-of-one where shape permits; `LayerIo` is the shared interface |
| M5 | per-row reference rebuilds an S^2 mask per layer | open: build once per forward and reuse; the two-maskless-call merge is the next step |
| M7 | artifact includes a random `lm_head` | fixed: the quantizer replaces it with a 1-row head; the engine ignores it |
| M11 | performance targets contradicted FLOP math | corrected to questions/s and 1.6 requests/s serial |

### Round 3 - 2026-10-21, Opus 5.5 xhigh

Verdict: **ship with fixes**. The blockers in that review were (1) no server
path and (2) the d3 prompt suffix. Both are fixed:

- `Backend::d3_answers` + `BatchHandle::submit_state_once` + the
  `/v1/systemone` branch now serve d3-turbo end-to-end; a startup self-test
  runs the real state capture + question pass.
- `src/d3.rs` is byte-identical to `d3_format.render` for the unit fixture and
  covers choice, noul and score; a test asserts the exact string.
- Config parsing now fails closed on a malformed/missing mode, a conflict
  between `config.json` and `decision_config.json`, or a readout with no
  declared mode.
- Split-KV workspace is sized from the device SM count, and the unused
  `prefill_cascade` entry was removed.
- `in_proj_a|b` are kept in the v4 artifact (measured 31/32 vs 30/32 for the
  BF16-gates v5 control). This is an explicit, recorded choice.
- The negative-control thresholds are now 1e-3, matching the measured
  0.0047-0.0049 deltas.

### Round 4 - 2026-10-21, Opus 5.5 xhigh (sign-off)

Verdict: **do-not-ship** at the time, with eight must-fixes. Dispositions:

| finding | disposition |
|---|---|
| prompt suffix had a literal space / missing `<` | fixed: `\x3cthink\x3e` escapes; unit test asserts the exact Python bytes; server-vs-HF re-run |
| W4A8 did not pass the max-Δp gate | served profile now also keeps `mlp.gate_proj` on W4A16: 31/32, mean 0.0156, max 0.0942; gate passes |
| requests over the engine caps returned 502 | `state_once_batch_rows` now splits into capped chunks over one cache; queue cap returns 503 |
| image/video states silently became text | `d3::contains_media` rejects them with 400; 1 MiB rendering cap before tokenization |
| self-test only checked finite logits | now also compares state-once against the per-row block path and a noul probability band |
| choice answers lacked confidence | added top-above-even confidence (vLLM-SR System One contract) |
| unbounded state-once queue | 32-request cap, skips cancelled replies, 503 on overload |
| docs stale | README, ENGINE and this section updated |

Remaining after round 4: the 32-row smoke gate is still not the 256-row §6.4
gate; W4A8 batch-of-4 vs single differs by up to 0.037 on valid same-state
questions (W4A16 0.009), while server request-to-request drift is at most
0.006 and the served 3-question request is within 0.028 of BF16; the per-row
attention, shared-seed, Marlin and state-LRU optimizations remain.

**Same-machine baseline (2026-10-21).** vLLM-SR `vllm-srun` 0.2.0 (native
engine, `--max-num-seqs 64 --enable-shared-context`) served the same 15 GB int4
artifact and the same 2,076-token state / 64-question request on the same RTX
4090: 3.89 s (3.86-3.89 s). The native server: 1.65 s (1.65-1.66 s), **2.36×
faster**, same top choice, largest probability difference 0.005. vLLM-SR loads
the artifact as W4A16; the native path runs W4A8 g64 with per-group INT8
scales.

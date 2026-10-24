# Decidra engine architecture

Status: Decidra 0.2 implements the d3-turbo path end to end — native CUDA
engine, GPTQ W4A8 quantization, eval harness, and the `/v1/systemone` server.
The server self-tests the exact state-once path at startup, refuses
non-`prefix_bidir` packages, and keeps an exact-state GPU cache plus a host L2 for repeated or
concurrent states. H100/H200 use the SM90 FA2 paths by default; FA3 is opt-in
and measured slower for this workload on H200. The remaining work is a larger
accuracy sample and further kernel tuning (§7).

## 1. What the model does

`vllm-sr/d3-turbo` is a 26.1B Qwen3.5 hybrid decision model: 48 Gated DeltaNet
layers and 16 full-attention layers (indices 3, 7, …, 63), hidden 5120, 24 Q /
4 KV heads at head_dim 256, 16/48 GDN K/V heads at 128, a separate F32
`[255,5120]` answer-code readout, and `attention_mode: prefix_bidir`.

`prefix_bidir` splits a prompt at the end of the state:

```
[ system + State: <state> ][ \n\nQuestion: ... options ... ][ readout ]
        state block S                     question block Q
```

- state queries see every state key, both directions;
- question queries see every state key **and every key in their own block**,
  both directions;
- **no state query sees a question key.**

The GDN layers stay causal and scan the state once. The state's hidden states,
full-attention K/V, final recurrent state and convolution tail are therefore
independent of every question. d3-turbo computes them once per request and
forks every question from that cache. `vllm-sr/d3` (Decision 3.0) is a
different mask (`noncausal_full_attention`); its state does see the questions,
so it cannot use this cache.

## 2. Engine API

```rust
let engine = Engine::new(model_dir)?;
engine.set_marlin(true);
engine.set_marlin_int8(true);
engine.set_int8_group_scale(true);   // group-64 W4A8
engine.set_gdn_chunked(true);
engine.set_gdn_chunked_tc(true);
engine.set_gdn_chunked_min_seq(16);
engine.preload_all()?;

let cache = engine.capture_state(&state_tokens)?;
let logits = engine.readout_state_once_batch(&cache, &question_blocks)?;
```

`capture_state` stores, per full layer, post-RoPE K/V `[S,4,256]`, and per GDN
layer the final f32 `[48,128,128]` recurrent state and the last three
pre-conv `in_proj_qkv` rows. `readout_state_once_batch` runs all question
blocks as one ragged batch: one GEMM per projection, one cascade attention
launch per row, a seeded GDN recurrence, and one `[B,255]` F32 readout.

Per-row reference for tests:

```rust
engine.decision_logits_split(&full_tokens, state_len, &codes)?;
```

`Engine::new` refuses a non-causal checkpoint without the vendored FlashInfer
library; `prefix_bidir` never falls through to the causal native kernels. On SM90,
FlashInfer FA3 can run the equal-length non-causal state pass and the
independent-length (`qo_len != kv_len`) cascade question pass, but it is
opt-in; FA2 is the measured default for d3.

## 3. Correctness (BF16, reference `d3_runtime.py`)

| check | max abs Δp |
|---|---:|
| full noncausal per-row, engine vs HF | 0.0061 |
| per-row block cascade, engine vs HF | 0.0022 |
| state-once (state + question pass), vs HF block | 0.0035 |
| state-once with production chunked GDN, vs HF block | 0.0029 |
| state-once vs per-row block, 75-token state / 49-token question | 0.0052 |
| state-once vs per-row block, production W4A8/g64 v4 | 0.00645 |
| batch-vs-single, production W4A8/g64 v4 | 0.02277 |
| zero layer-0 GDN seed (negative control) | 0.00834 |
| zero layer-0 convolution tail (negative control) | 0.00474 |

The F32 readout matters: with the previous BF16 head, the batch-vs-single gap
was 0.014. `HeadMatrix` now stores the head in F32 and runs an F32 cuBLAS GEMM,
matching `hidden.float() @ readout.T`.

## 3.1 Serving

Build with `--features native` and start with:

```
CUDA_VISIBLE_DEVICES=0 DECIDRA_BIND_ADDR=127.0.0.1:8899 \
DECIDRA_CHECKPOINT=/srv/models/d3-turbo-w4a8-g64-v4 \
DECIDRA_NATIVE_MODEL=/srv/models/d3-turbo-w4a8-g64-v4 \
DECIDRA_NATIVE_GPU_EMBED=0 ./target/release/decidra
```

The server detects `decision_config.json` with `attention_mode: prefix_bidir`,
renders the checkpoint's prompt with `src/d3.rs`, splits each prompt by
tokenizer offsets, verifies every question produced the same state tokens,
captures the state once, runs all question blocks as one batch, and returns
the `/v1/systemone` answer shape (choice, noul and score). Startup runs a
self-test through the same path and compares it with the per-row block path; a
broken mask, artifact or FlashInfer build refuses to serve. A missing
`prefix_bidir` config, missing `readout.safetensors`, or non-d3 checkpoint is a
startup error.

Text only: image/video states return an error until the vision tower is
ported. Per request: at most 64 questions; each question's state plus its own
block is at most 32K tokens. The serving worker coalesces requests, keeps an
exact-state cache and host L2, and combines question blocks from all requests sharing a
state. On H100/H200, `DECIDRA_STATE_ONCE_MAX_ROWS` (default 64) can be raised
to amortize weight reads across coalesced requests; the engine still splits at
the key/score caps.

GPU admission is memory-aware. `DECIDRA_BATCH_ROWS` budgets the **sum** of
question rows across coalesced requests, not just the first request. Before a
state restore/capture and again before the question pass, the worker checks
free VRAM, trims the scratch pools, trims the CUDA async-allocator pool with
`cuMemPoolTrimTo`, and offloads pinned states to the host L2 if the pass would
otherwise run below the reserve. The reserve scales with state length
(192 KiB/token + 1 GiB, minimum 1.5 GiB), because state capture and the
question pass allocate several per-token scratch buffers in addition to K/V.
State capture also forces scratch buffers to be released as they are dropped
for the duration of the capture; retaining them made a 12,192-token state
exceed 24 GiB while a 12,288-token state passed only because it crossed the old
low-memory cutoff. A real state-capture or question-pass OOM gets one trim-and-retry attempt. The state cache, host cache, and free-VRAM gauges are
exported as `semel_state_cache_*`, `semel_host_cache_states`, and
`semel_gpu_free_bytes`.

## 4. Quantization

The shipped recipe is **GPTQ-only, INT4 symmetric group 64, INT8 per-token
activations with per-64-group scales**, 1,024 block-masked calibration rows.
AWQ is not shipped: both AWQ+GPTQ test artifacts were far from BF16 (max Δp
0.50), while GPTQ-only stayed within 0.04 on the same row. The shipped v4
artifact quantizes `in_proj_a|b` as well; a v5 control kept them BF16 and was
slightly worse on the 32-row gate (30/32 vs 31/32), so v4 is the artifact.

The previous W4A8 artifact was quantized from **randomly initialized weights**
because `llmcompressor.oneshot(model=<path>)` loaded `AutoModelForCausalLM`
while the checkpoint uses `language_model.*` keys. The fixed pipeline loads
`Qwen3_5ForCausalLM` with
`key_mapping={r"^language_model\.": "model."}`, verifies a source tensor
bit-exactly, and uses the real block mask for calibration:

```
<safetensors>  state queries -> state keys only
               question queries -> state + own question block
```

The GDN layers receive the 2-D padding mask. The mask path is
`Qwen3_5TextModel.forward`'s `attention_mask` dict.

Artifact: `/srv/models/d3-turbo-w4a8-g64-v4` (15 GB after stripping the
CausalLM `lm_head`). Dequant gate: 496/496 modules pass cosine ≥ 0.95 against
the BF16 source; the v4 32-row eval passes:

| variant | top-1 | mean MAE | max Δp | flips |
|---|---:|---:|---:|---|
| W4A16 (dequant weights) | 31/32 | 0.0154 | 0.083 | 1 near-tie (BF16 margin 0.025) |
| W4A8, shipped SM89 defaults (gate/up/down/o/q INT8 group, k/v W4A16) | 31/32 | 0.0158 | 0.0837 | 1 near-tie |
| W4A8, old gate-W4A16 config | 31/32 | 0.0156 | 0.0942 | 1 near-tie |
| W4A8, all linears INT8 group | 31/32 | 0.0147 | 0.1047 near-tie / 0.0855 non-near | 1 near-tie |
| W4A16, same artifact | 31/32 | 0.0151 | 0.0875 | 1 near-tie |

`eval/d3_compare.py` gates on ≥95% top-1, mean MAE ≤0.02, max Δp ≤0.10 on
non-near-tie rows, and no flip whose BF16 margin is above 0.05.

Batch-shape checks: batch-of-one is bit-identical to the single path; four
questions sharing one valid state move by at most 0.037 (W4A8) or 0.009
(W4A16). Through the server, the same question in 1, 2, 4 and 8-question
requests moves by at most 0.006.

End-to-end on a 3-question request (choice + noul + score) through the server,
against the checkpoint's own `D3.system_one` on CPU BF16:

| answer | server | BF16 |
|---|---|---|
| route returns | 0.999938 | 0.999955 |
| receipt yes | 0.999875 | 0.999903 |
| urgency score | 1.5272 | 1.5284 |
| urgency p(Routine) | 0.0288 | 0.0147 |

Largest probability difference 0.0272; score difference 0.0012. The batched
W4A8 path is what the server serves; single-question parity (0.0056 max Δp
vs batch-of-one) is a separate check.

## 4.1 Architecture profiles (SM89 / SM90 / SM120)

Host-side choices live in `native/src/arch/`: `GpuArch` maps compute capability
to a profile, and `Gpu::new` verifies the embedded cubin's
`DECIDRA_COMPILED_ARCH` against the runtime device. The profiles currently
control the FlashInfer threshold, state-row default, scheduler budget, and GDN
chunk threshold. Shared kernels remain in `native/kernels/kernels.cu`; vendored
Marlin is in `native/kernels/marlin.cu`.

SM120 is a Blackwell workstation ISA, not a small SM100:

- fifth-generation tensor cores with FP4/FP8, but no TMEM, tcgen05, WGMMA,
  thread-block clusters, or distributed shared memory;
- 99 KiB opt-in CTA shared memory and 188 SMs on RTX PRO 6000;
- SM80-style register `mma.sync` plus `cp.async` is the compatible path;
- FP8 and INT8 large-M GEMMs use cuBLAS/cuBLASLt rather than SM100 collectives;
- Blackwell GeForce/WORKSTATION FP8 requires the TN operand layout.

### SM120 FP8 question path

The default is BF16 for state capture and for all rows below
`DECIDRA_FP8_MIN_ROWS` (512). With `DECIDRA_NATIVE_FP8=1` and
`DECIDRA_FP8_MODE=all_linear`, large question passes use E4M3 FP8 GEMMs for
the MLP and projection linears:

- weights are converted once at load to per-output-row E4M3 + f32 scales;
- activations are quantized per token;
- M is padded to 16 and cuBLASLt runs TN FP8 GEMMs;
- a fused epilogue applies the weight-row and activation-token scales and
  writes BF16;
- `capture_state` sets a guard so state capture never uses FP8.

On the 600 W WS host this lowers 64q c1 from **775 ms to 623 ms**
(103 q/s), and on Max-Q from **1,045 ms to 859 ms** (74 q/s). FP8 is
approximate: the measured single-vs-batch probability delta is **0.0362**
(versus 0.0102 for BF16) and the top choice matched on all **64** probe rows. The
all-linear mode is both faster and lower-error than gate/up-only (0.0248) and
all-MLP (0.0497) variants.

### SM120 NVFP4 path (experimental)

The loader also accepts compressed-tensors `nvfp4-pack-quantized`
checkpoints: E2M1 weights packed two per byte, E4M3 group-16 scales,
`weight_global_scale`, and `input_global_scale`. The native path repacks the
checkpoint scales into cuBLASLt's block-scaled VEC16_UE4M3 layout, quantizes
activations dynamically per row and 16-element group, and runs an FP4×FP4
cuBLASLt TN GEMM with BF16 C/D. The routing is per module, so GDN
`linear_attn` and any other ignored projections stay BF16 in the same
checkpoint.

This path is **not recommended for quality-sensitive decisions** and is not
enabled by default. On the 32-row HF reference probe:

| FP4 scope | 64q c1, 600 W | HF top-1 | Max Δp | unsafe flips |
|---|---:|---:|---:|---:|
| gate/up only, down/attention BF16 | 611 ms / 105 q/s | 29/32 | 0.113 | 2 |
| all MLP, attention BF16 | — | 29/32 | 0.284 | 2 |
| all MLP + full attention, GDN BF16 | **500.8 ms / 128 q/s** | 28/32 | 0.279 | 3 |

For comparison, the opt-in FP8 path is 32/32 with max Δp **0.0115** and
623 ms / 103 q/s. FP8 therefore dominates the quality/perf tradeoff on this
model. NVFP4 remains an explicit artifact-level opt-in for experiments; the
repo does not ship an NVFP4 checkpoint. The recipe used 1,024 d3
`prefix_bidir` calibration rows, GPTQ with static act-order, and excluded
`lm_head`, `visual`, `linear_attn`, and `readout`; see
`quant/quantize_nvfp4.py`. Its default `--scope gate_up` also leaves the
full-attention and down projections BF16; `--scope mlp` and `--scope all`
reproduce the table variants. The loader detects NVFP4 per module, so no
environment flag is required beyond supplying the artifact.

Rejected SM120 experiments: fused gate/up + SiLU (616.8 ms but parity
0.0547), FP8 fast accumulation (no gain), an L8 register GDN variant
(correctness failure), W8A8 (parity 0.281), W8A8 transposed (1.88 s), chunked
GDN for short questions (973 ms), GDN vc32/vc64 changes (no gain), and NVFP4
W4A4 for quality-sensitive serving (accuracy gate failed).

The remaining time is dominated by the f32 GDN recurrence (~86 ms at 600 W),
cascade attention (~27 ms), RMSNorm/residual/epilogue kernels, and the state
cache/readout path. The FP8 linears are at roughly 80% of the SM120 FP8 tensor
peak at the sustained clock.

## 5. Performance on one RTX 4090 (450 W cap)## 5. Performance on one RTX 4090 (450 W cap)

Both GPUs are capped at 450 W. The 4090 draws ~449 W at ~2.82 GHz. Defaults:
W4A8 g64, gate/up/down/o/q INT8 group, k/v W4A16, register GDN, suffix-only
question tokenization, 32-row scheduler budget, 16-entry / 5 GiB pinned state cache plus 8 GiB host L2
with a 2 GiB scratch reserve.

| phase / batch | time |
|---|---:|
| preload (15 GB + Marlin repack) | 9.0 s |
| state capture, 2K tokens | ~400–500 ms |
| 64q pass profile at 450 W (3,236 question tokens) | 685 ms regions / 733 ms p50 |
| gate / up / down at 450 W | 108 / 108 / 104 ms |
| GDN recurrent (register) at 450 W | 93 ms |
| cascade attention at 450 W | 39 ms |
| in_proj_qkv / z / out at 450 W | 48 / 31 / 31 ms |
| 1q request, warm / cold | 43 ms / 539 ms |
| 16q c1 / c8 | 236 ms / 1.15 s |
| 64q c1 warm / cold | 936 ms / 1.375 s |
| 64q c8 | 4.18 s (69 q/s) |
| 64q c16 / c32 | 7.91 s / 15.38 s (69 q/s) |

The 2K/64q pass is ~232 TFLOP/s of linear work, ~35% of the 660 INT8 peak at
spec clocks. The 64q c8 throughput is flat from c8 through c32; distinct states
remain serialized by the single CUDA stream. GPU1 draws 449.5 W and is
power-saturated.

Long state at 450 W: 8K/64q **1.02 s (63 q/s at c8)**, 24K/64q **1.29 s
(51 q/s)**. The pinned 16-entry GPU state cache plus an 8 GiB host L2 turns an
8-state set from 0/8 hits (23.5 q/s) with 4 entries into 8/8 hits (68.9 q/s),
16 states into 16/16 hits at 69 q/s, and 32 states into 32/32 hits at 63 q/s
by offloading overflow states over PCIe instead of recapturing them. The 2 GiB
reserve trims pools and evicts before long-state capture; 24K leaves 4.48 GiB
free.

Scheduler work: question suffixes are tokenized separately after the first
question (64q tokenization 95→7 ms), a 32-row budget improves 16q c8 p50 from
1.60→1.15 s without changing q/s, tokenization runs on the blocking pool, the
cuBLAS handle is rebound on the CUDA worker (fixes libcublasLt segfaults under
c16/c32), and `/metrics` exposes tokenize/queue/group/pass/cache counters.

On one **H200 (143 GB, SM90)** with the published BF16 package, native
state-once (FA2 cascade) vs vLLM-SR 0.2.0 engine mode (same package, same H200,
same request):

| 2K state | native warm | native cold | vLLM-SR |
|---|---:|---:|---:|
| 1 question | 36.8 ms | 350 ms | 444 ms |
| 8 questions | 126 ms | 463 ms | 631 ms |
| 64 questions | 800 ms | 1.16 s | 2.90 s |
| 64q, 8 concurrent copies | 5.47 s (94 q/s) | 4.57 s cold (64 q/s) | 13.2 s (22 q/s) |
| 16q, 8 concurrent | 1.32 s (97 q/s) | — | 4.30 s (17 q/s) |
| 8K state, 64q | 1.27 s | — | 4.39 s |
| 24K state, 64q | 2.62 s | — | 9.45 s |

Native BF16 is **2.6–3.6× faster per request** and **2.9–5.8× higher question
throughput** than vLLM-SR on the same H200. The 64-question pass is still the
dominant cost; the measured aggregate is ~411 BF16 TFLOP/s at c8, about 42% of
the H200's dense BF16 peak. A profile of the 2K-state, 64-question pass shows
GDN recurrence 155 ms (32.5%), full attention 58 ms (12%), MLP GEMMs 140 ms
(29%), and the remaining projections/norms/copies. The current per-row ceiling
is ~97 q/s at 16q/c8; the 990-TFLOP BF16 compute ceiling at ~2.7 TFLOP per
question is ~366 q/s, so launch/occupancy and the GDN scan, not matrix math,
set the gap. The 127 q/s target (8 concurrent 2K-state/16-question requests)
was **not reached**: best measured 100 q/s on H100 SXM and 97 q/s on H200,
21–24% short. The register-resident GDN recurrence is now the default for
short blocks; it is 3–4% faster and preserves the 32-row accuracy gate. The next 25% needs the GDN seed/batched-recurrence and batched ragged
cascade work below, not another precision switch.

**SM90 precision choice.** The published BF16 package is the fastest H200 path
and fits the 80 GB H100 (49 GB weights + 2.4 GB embedding + caches). A
re-quantized W4A8/g64 H200 artifact was measured but **failed the 32-row
accuracy gate** (29/32 top-1, two unsafe flips); do not use it. The shipped
15 GB W4A8 v4 artifact is the RTX 4090 production path, where Marlin's
SM80-style MMA is fast; on H200 the same W4A8 path was slower (1.26 s warm /
1.77 s cold for 64q, 58 q/s at c8) than BF16. SM90 W4A8 needs a WGMMA kernel
before it can beat BF16.

FA3 is available but opt-in (`DECIDRA_FLASHINFER_FA3=1`). On the d3 2K–24K
workload the FA2 cascade was equal or faster on H200 (64q: 800 ms vs 834 ms;
24K: 2.62 s vs 3.06 s); FA3 remains for other shapes and H100 experiments.

On one **H100 SXM 80GB (588 W cap, SM90, BF16, FA2)** the same package fits with
30.9 GiB free after preload: 16q/c8 **1.28 s (100 q/s)**, 64q/c1 **824 ms**,
64q/c8 **5.10 s (100 q/s)**, 8K/64q **1.17 s**, 24K/64q **3.10 s**, and the
state-once self-test passes at max Δp 0.0104 with the register GDN recurrence
(default for short blocks). vLLM-SR 0.2.0 on the same H100 takes **2.43 s**
for 64q and **13.66 s (23 q/s)** at c8. The cheaper H100 NVL (400 W cap) does
64q in 916 ms and 80 q/s at c8; SXM is the fastest H100. H100 and H200 are both SM90;
the 80 GB H100 is enough for the 49 GB BF16 package plus the exact-state caches.
The full 1/8/16/32/64-question matrix is in [`BENCHMARKS.md`](BENCHMARKS.md).

The H200 engine preload of the BF16 package took 39 s; H100 33 s.

**Same-machine baseline (450 W historical).** vLLM-SR `vllm-srun` 0.2.0
(native engine, `--max-num-seqs 64 --enable-shared-context`) served the same
15 GB int4 artifact and the same 2,076-token state / 64-question request on
the same RTX 4090 at 450 W: **3.89 s** per request. The native server answered
in **1.65 s cold / 1.17 s warm**, **2.4–3.3× faster**,
with the same top choice and a largest probability difference of 0.005 on the
repeated question. vLLM-SR loads the artifact as W4A16 (int4 unpacked to
BF16); the native path serves the same weights as W4A8 g64 with per-group INT8
scales.

Choice/score `confidence` is the vLLM-SR System One formula (top probability
above an even split for choice; spread around the mode for score), matching
the model card. It is not entropy: a 3-way 0.9996 top gives 0.9994, and the
measured 0.8172 top gives 0.7258.

## 6. Tests

```
ROOT=$(git rev-parse --show-toplevel)
# BF16 parity, 32 rows
CUDA_VISIBLE_DEVICES=1 DECIDRA_D3_MODEL=/srv/models/d3-turbo \
  DECIDRA_D3_PROBE="$ROOT/eval/data/d3_probe_split32.json" \
  cargo test --release -p decidra-native --test d3_eval -- --ignored --nocapture

# state-once / batch parity and negative controls
CUDA_VISIBLE_DEVICES=1 DECIDRA_NATIVE_CACHE=0 DECIDRA_GDN_CHUNKED=1 \
  cargo test --release -p decidra-native --test d3_state_once_negative -- --ignored --nocapture

# 2K state, 64 questions
CUDA_VISIBLE_DEVICES=1 DECIDRA_MARLIN=1 DECIDRA_INT8=1 DECIDRA_INT8_GROUP=1 \
  DECIDRA_PRELOAD=1 DECIDRA_GDN_CHUNKED=1 DECIDRA_GDN_CHUNK_MIN=16 \
  DECIDRA_D3_MODEL=/srv/models/d3-turbo-w4a8-g64-v4 \
  DECIDRA_D3_PROBE="$ROOT/eval/data/d3_once_64_2k.json" \
  cargo test --release -p decidra-native --test d3_perf -- --ignored --nocapture

# artifact dequant gate
python3 quant/d3turbo/verify_quant.py \
  --quant /srv/models/d3-turbo-w4a8-g64-v4 --group 64
```

## 7. Remaining work

1. **Per-row attention cost.** 1,024 FlashInfer launches per 64-question
   pass. FA3 cascade is available on SM90; FA2 measured faster. The next step is a batched ragged
   cascade with LSE merge.
2. **Shared GDN seed.** The single-launch seed broadcast is implemented and
   correctness-verified. The remaining GDN cost is the register/shared
   recurrence itself.
3. **Marlin tuning.** Gate/o moved to INT8 group scales; gate/up/down are now
   the remaining large GEMMs. Concatenating gate/up or q/k/v, larger row
   chunks, and split-K remain candidates.
4. **Larger accuracy set.** 32 rows is a smoke gate. Move to 256–1,024 rows
   with noul/score and ≥2K states before publishing a quality claim.
5. **Cross-state batching.** The GPU cache and host L2 coalesce exact-state copies; distinct
   concurrent states are serialized. A multi-cache ragged question pass is the
   next throughput step.
6. **450 W efficiency.** The GPU is power-limited, so further gains require
   fewer joules per token: batched cascade attention, fused/register GDN, and
   MLP/Marlin fusion. Tokenization, row-budget scheduling, state byte budgets,
   and cuBLAS thread affinity are complete.

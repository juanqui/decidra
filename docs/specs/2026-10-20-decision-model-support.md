# Decision-model support specification

Status: proposed. Targets: `autotrust/AutoTrust-JEV-27B-v1.1`, `vllm-sr/d3`,
`vllm-sr/d3-turbo`, `perplexity-ai/pplx-decider-v1.2` (and the existing JEV
production model). Goal: be the best decision-model inference engine — one
resident runtime, decision-specific optimizations vLLM cannot express, measured
against vLLM on the same GPU and the same quantized artifact.

## 1. Research summary

All four target checkpoints share the same backbone, which changes the problem
from "port four models" to "support one family as data":

| | AutoTrust v1.1 | d3 | d3-turbo | pplx-decider v1.2 |
|---|---|---|---|---|
| Backbone | Qwen3.8-27B, Qwen3_5Model | same | same | same |
| Layers | 64 (48 GDN + 16 full) | same | same | same |
| Hidden / heads | 5120, 24 q / 4 kv, d=256 | same | same | same |
| Vision | 0.46B tower | same | same | same |
| Attention (full layers) | `bidirectional` / `prefix_bidir` | `noncausal_full_attention` | `prefix_bidir` | `noncausal_full_attention` |
| GDN | causal | causal | causal | causal |
| Readout | 255 code rows folded into `lm_head` | separate `readout.safetensors` F32 [255,5120] | same, F32 | separate BF16 [255,5120] |
| Pooling | last | last | last | one marker `<|fim_suffix|>` per question |
| Multi-question | one per forward | one per forward | state-once (one read, many questions) | joint protocol v3, markers |
| Temperature | 1.0 | 1.0 | 1.0 | 1.382 |
| Options | 2..255 | 1..255 | 1..255 (+box/track) | 1..255 |
| Context | 32K | 32K | 32K default / 131K option | 32K validated / 262K max |
| Upstream serving | vLLM 0.31 + StartLux plugin | vLLM main `/v1/systemone` (0.31 lacks it) | vLLM SR Engine | reference package + fast path |

Key contract facts from the checkpoints:

- `decision_config.json` carries `codes` (A..), `token_ids`, `temperature`,
  `attention_mode`, `pooling`, `readout_dtype`, `max_length`/`max_pixels`.
  Readout row `i` corresponds to vocabulary token `token_ids[i]`; the head is a
  separate `[255,5120]` matrix (AutoTrust folds the same rows into `lm_head`).
- The d3/d3-turbo prompt (`d3_format.py`) and the AutoTrust prompt
  (`jev_format.py`) are the same upstream code-readout template: system prompt,
  chat template, coded option lines, readout at the last position. The pplx
  protocol is joint: state, then every question with options and a
  `<|fim_suffix|>` marker; readout at each marker.
- pplx explicitly states: full-attention layers are noncausal over the whole
  request; linear-attention layers keep native causal behavior; default causal
  inference does not reproduce the checkpoint. The d3 card and the AutoTrust
  config state the same requirement in their own terms.
- d3/d3-turbo share the backbone, tokenizer and readout; they differ in
  `attention_mode` and the multi-question behavior. If the weight shards hash
  identical, **one resident weight set can serve both model ids** (hash first,
  §11).
- vLLM support: d3 is native on vLLM main (PR #61565, not in 0.31.0);
  d3-turbo is served by vLLM SR Engine; pplx ships its own reference/fast
  package (packing, varlen FlashAttention, compile, CUDA graphs, last-layer
  pruning, optional FP8). Vendor throughput numbers were measured on MI325X /
  GB200, not on a 4090; they are claims to reproduce, not baselines.

## 2. Current engine: what is true, what is missing

True today (verified in code and production):

- JEV W4A8 production: 28/28 vs the CPU artifact golden, max Δ 0.0176,
  KL 2.7e-4; 21/24/35–39/49–50/53–54 req/s c1–c16; 0.91/1.85/4.18 s at
  4K/8K/16K on the 4090; 21.6 GB peak.
- The engine is prefill-only: one forward, last-token readout, no decode.
- Norm handling already uses the zero-centered `(1+w)` convention for input,
  post-attention, final and q/k norms, and plain `w` for the GDN gated norm.
  Do not change this without evidence.
- A BF16 weight path (`LinearWeight::Bf16`) exists for parity tests.
- Batch 2–8 does **not** use FlashInfer: `flashinfer_for` is only called when
  `batch == 1`; batched prefill falls back to the custom attention kernels.
- `fi_prefill.cu` is a one-shape shim: `kCausal`, 24/4 heads, head_dim 256,
  single prefill. There is no mask parameter and no varlen path.
- RoPE is 1-D. Correct for text-only mRoPE (all three position streams equal),
  wrong once image tokens appear.

Missing for the new family:

1. **Noncausal / prefix-bidirectional full attention.** The full-attention layers
   must attend over the full prompt (or prefix); the GDN layers stay causal.
   The vendored causal mask silently serves the wrong model and produces the
   AutoTrust KL 0.15 class of error.
2. **Separate 255-row readout head** with option-only logits, per-marker
   pooling, temperature and code mapping.
3. **Per-model prompt/protocol adapters**: chat template + coded options
   (AutoTrust/d3/d3-turbo) and the joint marker protocol (pplx).
4. **Qwen3.8 parity isolation** (AutoTrust engine vs HF KL 0.15). The layer-63
   probe result is likely a measurement artifact: HF's last hidden state is
   taken after the final norm; the engine's `Stage::Layer(63)` is pre-norm. That
   comparison cannot establish a graph bug. The P0 harness below replaces it.
5. **Vision path** (0.46B tower, 1.6 MP, 3-D mRoPE) for d3/d3-turbo/pplx.
6. **Quantization calibrated under the real mask**, per family, with
   margin/flip-rate validation at 255-way (24-way JEV evidence does not
   transfer).

## 3. Architecture: one runtime, models as data

```
ModelCard / manifest (TOML, pinned + hashed)
      │ compiler
      ▼
Program (IR, content-hashed)  ──►  Runtime (Rust) ──► ForwardJob(s) ──► kernels
```

The scheduler must never contain `match model_name`. The IR is small:

- `ForwardJob { tokens, pos3 (t,h,w), segments (Prefix/Context/Question(i)/Image(j)),
  mask, vision refs, readout sites, cache_key }`
- `MaskSpec { Causal | Full | PrefixBidir { prefix_len } | Blocks { shared, groups } }`
- `HeadMatrix { file | lm_head[ids], dtype, bias, temperature, code_ids }`
- `Postprocess { choice | noul | score | box | track }`

Rules:

- Adapters render prompts and emit jobs; they never touch CUDA. Backbone code
  never sees prompts.
- The GDN layers ignore `MaskSpec` and stay causal; the mask applies to the 16
  full-attention layers only.
- Fail closed: a model id without a manifest returns 404. Never fall back to
  causal + `lm_head`; that silent fallback is exactly how the AutoTrust
  checkpoint was validated with the wrong mask.
- Program hash = cache key prefix, reproducibility stamp and parity-test id.

Family coverage (no new op needed for any row):

| | Mask (full layers) | Readout sites | Pooling | Head |
|---|---|---|---|---|
| JEV | causal | 1 | last | lm_head rows (24) |
| AutoTrust | bidir/prefix | 1 | last | folded lm_head rows (255) |
| d3 | full | 1/forward | last | separate 255 |
| d3-turbo | prefix_bidir + forks | 1/fork | last | separate 255 |
| pplx | full (joint) | M markers | per-marker | separate 255 |

## 4. Attention

Implement the mask modes with FlashInfer's built-in masks; custom bitmasks use
the slow path.

| Mode | Implementation |
|---|---|
| Causal | `BatchPrefillWithRaggedKVCache`, `kCausal` |
| Full | same call, `kNone` |
| PrefixBidir(P) | two passes: prefix Q[0,P) vs K/V[0,P) with `kNone`; suffix Q[P,L) vs K/V[0,L) with `kCausal` (FlashInfer aligns causal bottom-right, so suffix i sees prefix + suffix ≤ i). Disjoint output rows; no merge. |
| Blocks | partial attentions over shared and group spans, merged with `merge_state` on the LSE |

Required kernel work:

1. **Varlen ragged batched FlashInfer** with `qo_indptr`/`kv_indptr`, one launch
   per mask class per layer, up to 8 sequences / 16K tokens. This also removes
   the non-FlashInfer batch fallback and is a prerequisite for state-once.
2. **Template instantiations**: head_dim 256, GQA 6, bf16, `{kNone, kCausal}`,
   with/without LSE output.
3. **GDN packed batches**: reset conv/recurrent state at `cu_seqlens`; carry
   the 3-token conv tail; keep the recurrent state F32 (BF16 drifts).
4. **1-D RoPE now, 3-D mRoPE with vision**: for images, position ids are
   (t,h,w); continue text after an image from `max(pos)+1`; the interleaved
   [11,11,10] layout over 32 frequency pairs must match HF unit tests.

Correctness traps (ranked):

1. Mask silently differs from training (the AutoTrust bug class). Per-layer
   cliff test in CI on every model; `mask_mode` in every cache key.
2. GDN conv tail dropped at a fork/prefix boundary — first 3 tokens wrong,
   invisible in average KL.
3. Packed GDN conv/state leaking across sequences; test [A,B] packed vs B alone
   bit-identical.
4. Padding entering GDN state or attention keys; use ragged/no padding.
5. Cached prefix K is already rotated; suffix queries use unrotated RoPE.
6. GQA mapping `q_head // 6`, not `% 4`.
7. Split-K nondeterminism vs exact-decision cache; fixed reduction trees for
   the readout GEMM.

Long context: the engine is layer-major and prefill-only; noncausal 131K prefill
is tens of seconds on a 4090 and the MLP intermediate alone is 4.5 GB. Gate at
**32K** (pplx's validated range); 48 GB+ class for longer, and do not advertise
131K/262K on the 24 GB card.

## 5. Readout

```
HeadMatrix { source: lm_head[ids] | readout.safetensors, dtype, bias, T, code_ids }
```

- Apply the final RMSNorm to the readout rows only; gather R rows; fp32 GEMM
  `[R,5120] × [5120,K]`, K ≤ 255. This is megabytes, not the 248K-vocab head.
- Pre-extract JEV/AutoTrust `lm_head` rows at load into a dense `[K,5120]`
  block; avoid a strided dequant gather per request.
- Pooling: `last` = last row; `markers` = each `<|fim_suffix|>` row (assert the
  marker count equals the question count; never search text for the marker).
- Mask option codes ≥ n to −∞ before the softmax; apply `T` at runtime (keep
  the trained temperature separate from post-hoc confidence calibration).
- Outputs: probabilities over valid options, `confidence`, `margin = p1−p2`,
  entropy; `score` gives expected level index; `noul` maps to P(true).
- Fuse final-norm → gather → GEMM into the last block epilogue; for M markers,
  one `[M,5120]×[5120,255]` GEMM and M softmaxes.
- Last-layer pruning: only layer 63's MLP at non-readout rows can be skipped
  (~1–3%); K/V are needed at all rows. Demote it; pplx's gain is packing, not
  pruning.
- Numerics: fp32 accumulation; 255-way margins are much tighter than 24-way.
  Validate with margin-bucket flip rates, not top-1 alone.

## 6. Prompts, adapters, API

Manifest per model, e.g.:

```toml
id = "vllm-sr/d3-turbo"
backbone = { path = "/srv/models/d3", revision = "<sha>", quant = "w4a8-g128" }
attention = { mode = "prefix_bidir", prefix = "until_first_question" }
gdn = "causal"
prompt = { template = "templates/d3.jinja", multi_question = "state_once" }
readout = { file = "readout.safetensors", dtype = "f32", codes = "codes.json",
            bias = false, temperature = 1.0, pooling = "last" }
decision_types = ["choice", "noul", "score", "box", "track"]
max_tokens = 32768
golden = "goldens/d3-turbo-256.jsonl"
```

- `/v1/decide` stays for JEV. Add `/v1/systemone` as a pinned conformance
  clone of vLLM main `/v1/systemone`: record request/response pairs from the
  pinned commit and replay them in CI (the schema is not official yet).
- `/v1/models` exposes `metadata.decidra: {decision_types, options_max,
  mask_mode, max_ctx, cache_scope, protocol}`.
- Startup self-test against `golden`; refuse to serve on KL/top-1 gate failure.
- LiteLLM: route public names to the tinylife port; one backbone per GPU
  process; keep the OpenAI shape and put the decision payload in
  `extra_body.questions` or the existing `/v1/decide` JSON content.

## 7. Cache and batching

Three tiers:

| Tier | Key | Contents |
|---|---|---|
| Decision LRU | program hash, prompt/token hash, image hash, pos-id hash, options, T, **mask mode** | decisions/readouts |
| Prefix (KV+GDN) | program hash, prefix tokens, image hash, pos ids | paged KV blocks + frozen GDN state + conv tails |
| Image encoder | image hash + tower hash | visual embeddings |

Feasibility by mask:

- **PrefixBidir (AutoTrust, d3-turbo): yes**, all layers. Prefix rows never
  attend to the suffix; the 16 full-attention K/V and the 48 GDN states depend
  only on the prefix.
- **Full (d3, pplx): effectively no.** From the first full-attention layer,
  every prefix row depends on the question; only layers 0–2 are reusable (~5%).
- **Blocks/shared: yes** for the shared span, if the shared rows do not see the
  questions.

Costs on this model: 64 KB/token full-attention K/V (16 layers × 4 heads × 256
× 2 × 2 B); GDN state ~151 MB fixed per prefix entry (48 layers × 48 heads ×
128² × 4 B) plus the conv tail. GPU1 has ~1.5 GB headroom; use a host-RAM LRU
(122 GB) and restore over PCIe; a 4K prefix is ~400 MB (~60 ms restore vs
~900 ms recompute).

State-once (d3-turbo): prefix stage produces shared KV blocks + frozen GDN
state; fork with shared page tables and `state_mode = shared_ro`; query stage
uses `prefix_lm` with the cached prefix. Never duplicate the prefix KV; never
fork the GDN state; copy-on-write per row only for the short query recurrence.
Batch 64 rows (raise the cap from 8; tail KV is ~168 MB).

Joint cache scope: pplx's markers may attend to all questions; if so, a
question's answer depends on the whole set and per-question cache entries are
invalid. The IR declares `cache_scope: PerSite | PerStage`; settle the semantics
from the reference code before P3, and include question order in the key.

Batching: ragged/varlen FlashInfer; group jobs by mask class + head; CUDA
graphs bucketed by total tokens (128/256/512/1K/2K) with ragged metadata in
fixed device buffers; cascade (d3-flash/edge → full) only after per-model margin
calibration.

## 8. Quantization

- The four targets are Qwen3.8-27B; on the 4090 only W4A16/W4A8 fit (FP8 27B
  is ~27 GB). JEV's W4A8 g128 recipe is the starting point; the readout stays
  BF16/F32.
- **Recalibrate AWQ/GPTQ under each model's real mask.** The AutoTrust
  calibration ran causal prompts; bidirectional models have different
  activation statistics. A causal-calibrated W4A8 on a bidirectional model is
  silently off-distribution.
- Validate per family at 255-way with margin histograms and flip rates, not
  top-1 agreement alone. The 24-way JEV margin (0.0176) is not evidence for
  255-way.
- Keep the JEV production artifact and gates unchanged; add per-family gates.

## 9. Qwen3.8 / AutoTrust parity program (P0)

The layer-63 result is not yet a diagnosis. The four-number probe and ordered
tests:

1. Dump from HF-BF16 and engine for a fixed 512-token text and a fixed image:
   embedding, hidden at layers 0,1,2,3,7,...,63, `final_norm_out`,
   `readout_logits`. Report `cos`, `relL2`, `‖a‖/‖b‖`, top divergent channels.
2. L2-normalize both tensors and re-measure cos. A jump to ≥0.999 indicates a
   scale/norm or pre/post-norm comparison artifact; ~0.85 indicates a
   directional (mask/attention) bug.
3. Compare the engine's final-norm output with HF's post-norm last hidden;
   compare the engine's layer-63 pre-norm with HF's layer-63 hook directly.
4. Run the engine with BF16 weights against HF-BF16, both causal; then enable
   W4 kernels to isolate the quantized kernels; only then enable the mask.
5. Checks: norm weight mean (≈1.0 plain, ≈0.0 delta ⇒ 1+w); mRoPE unit test at
   positions 1/100/1000; output gate `silu` vs `sigmoid`; GDN A_log/softplus;
   first full-attention layer index (3,7,...).
6. Gates: final KL ≤ 0.01 (target ≤ 3e-3), top-1 ≥ 99.5%, layer-63 cos ≥ 0.999
   after correct indexing; JEV stays 28/28 and max Δ ≤ 0.0176.

Vision (P4): run the 0.46B tower in a sidecar (HF, fp16, ~1 GB) that returns
embeddings; splice at image segments with 3-D mRoPE; cache by image hash. Build
it with the prefix cache, not after. Port natively only if profiles demand it.

## 10. Benchmarks and gates

Same 4090, same quantized artifact, vLLM main (pinned commit, native
`/v1/systemone` for d3) versus Decidra. Never compare across vendor GPUs.

Per model × {1K, 4K, 16K, 32K} × {c1, c8, 64-question}:
- decisions/s, p50/p99 time-to-all-decisions, peak VRAM, TMAC/s (normalize the
  hardware), cache hit/miss and prefix restore-vs-recompute curves;
- accuracy vs the correct-mask BF16 reference: top-1, max |Δp|, KL, and
  margin-bucket flip rates, ≥256 held-out rows per family.

Targets:

| Model | Target |
|---|---|
| JEV | hold 28/28, Δ ≤ 0.0176; ≥1.2× vLLM at c1 and c8 |
| AutoTrust | KL ≤ 0.01 vs correct-mask reference; ≥99.5% top-1; ≥1.2× vLLM |
| d3 | match vLLM `/v1/systemone` within KL ≤ 0.01; ≥1.2× same-hardware |
| d3-turbo | 64-question/photo ≤ 0.10× the 64-prefill reference; bit-identical to 64 independent forwards on 200 photos |
| pplx | joint decisions match reference; 32K; ≥ vLLM fast path normalized |

## 11. Phases and gates

| Phase | Scope | Gate |
|---|---|---|
| P0 Parity | 4-number probe, layer/sublayer dumps, HF hooks; norm/mask/quant isolation; AutoTrust fix | KL ≤ 0.01, layer-63 cos ≥ 0.999, JEV unchanged |
| P1 Attention | Ragged varlen FlashInfer `{kNone,kCausal}`; PrefixBidir two-pass; GDN `cu_seqlens`/conv tail; mask unit tests | dense fp32 reference per mask; pack isolation; JEV no regression |
| P2 Adapters/readout | Manifest/IR, ForwardJob, HeadMatrix, postprocess; AutoTrust + d3 single-question | ≥99% top-1, KL ≤ 3e-3 vs reference on 256 rows |
| P3 Joint/markers | pplx protocol v3, marker readout, cache_scope; last-layer pruning | joint output matches reference; order sensitivity measured |
| P4 State-once/vision | PrefixBidir cache (VRAM + host), vision sidecar, 3-D mRoPE, image cache | cached vs uncached ≤ 1e-5; image goldens |
| P5 Perf/quant | CUDA graphs, quant recalibration under the real mask, margin gates | ≥97% of BF16 quality; same-hardware benchmark table |
| P6 Routing | Margin-calibrated cascade | accuracy cost measured; p99 SLO |

Sequencing warning: P1 before P2; P0 before P4. Building state-once on a broken
mask attributes mask errors to the cache.

## 12. Risks and open questions

Risks: mask semantics silently differ from training (critical); GDN conv tail
dropped (critical); 24 GB cannot serve 131K/262K (high; gate at 32K); 255-way
W4A8 accuracy erodes while 24-way passes (high; margin gates); quant calibrated
under the wrong mask (high); one 27B per GPU (~21.6 GB, 1.5 GB headroom; several
families need several GPUs or hot-swap); GPU1 is production (P0 runs need the
tenant stopped); `/v1/systemone` schema drift (pin + fixtures); vision absent
(high for d3-turbo/pplx); cascade drift (margin calibration only).

Open questions to settle before P2/P3:
1. AutoTrust KL 0.15: mask, final-norm comparison, quant kernel, or modeling
   difference (P0 answers).
2. Do the 16 full-attention layers include layer 3,7,...,63? (Yes per config;
   confirm the first cliff.)
3. pplx markers: do later questions attend to earlier ones? `Full` vs `Blocks`;
   cache scope.
4. GDN chunked-parallel or serial today? If serial, largest remaining kernel
   win.
5. d3/d3-turbo weight shards hash-identical? If yes, one resident weight set.
6. d3-flash/edge are not in the target list; treat them as candidate cascade
   models, not supported ids yet.
7. Box/track: settle the postprocess spec from the d3-turbo code before P4.

## 13. Sources and consultations

- HF model cards/configs/custom code: `vllm-sr/d3-turbo`, `vllm-sr/d3`,
  `vllm-sr/d3-flash`, `vllm-sr/d3-edge`, `perplexity-ai/pplx-decider-v1.2`,
  `pplx-decider-flash-v1.2`, `pplx-decider-v1.1-27b`,
  `autotrust/AutoTrust-JEV-27B-v1.1`.
- vLLM PR #61565 (d3 `/v1/systemone`), vLLM SR Engine docs, decision.vllm-sr.ai
  API docs, pplx `source/src/pplx_decider` reference/fast paths.
- Engine code: `native/src/engine.rs`, `native/src/flashinfer.rs`,
  `third_party/flashinfer/fi_prefill.cu`, `native/src/kernels.cu`,
  `native/src/weights.rs`, `native/src/config.rs`; `docs/JOURNAL.md`.
- Consultations: the requested `codex`/`gpt-astra` and `opus-5.5` were not
  available in this environment (`codex` not installed; Claude rejected the
  model id `opus-5.5`). Reviews were obtained from the harness's default
  completion model (`/tmp/consult-harness.md`) and the available Claude Code
  `opus` alias, which read the engine sources (`/tmp/consult-opus.md`). The
  reviews agree on the ranking: mask/parity first, models as data, then
  state-once/joint/option-only, then perf. Vendor throughput claims must be
  reproduced on the same 4090 before they are used as baselines.

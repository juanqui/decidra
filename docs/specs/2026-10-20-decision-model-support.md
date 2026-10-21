# Decision-model support specification

Status: reviewed against the checkpoints and the engine sources (Claude Code
`claude-opus-5-5` review, 2026-10-20). Targets:
`autotrust/AutoTrust-JEV-27B-v1.1`, `vllm-sr/d3`, `vllm-sr/d3-turbo`,
`perplexity-ai/pplx-decider-v1.2`, plus the existing JEV production model.
Goal: one resident runtime for the family, decision-specific optimizations
vLLM cannot express, measured against vLLM on the same GPU and quantized
artifact.

## 1. Research summary

All four targets share the Qwen3.8-27B hybrid backbone (64 layers: 48 GDN +
16 full at indices 3,7,...,63; hidden 5120; 24 q / 4 kv; head_dim 256;
inter 17408; vocab 248320; GDN 16 k / 48 v heads, 128 dim, conv 4; mRoPE
[11,11,10], partial rotary 0.25; swish output gate; eps 1e-6; 0.46B vision
tower in the checkpoint). The differences are masks, readout sites, prompts and
post-processing:

| | AutoTrust v1.1 | d3 | d3-turbo | pplx-decider v1.2 |
|---|---|---|---|---|
| Full-attention mask | **noncausal full** | noncausal full | **prefix_bidir**: state prefix + bidirectional question block | noncausal full |
| GDN | causal | causal | causal | causal |
| Readout head | 255 bf16 rows folded in `lm_head` (other rows zero) | separate F32 [255,5120] | same F32 | separate BF16 [255,5120] |
| Sites | last | last | last; **Box/track multi-site** | one `<|fim_suffix|>` marker per question |
| Multi-question | one per forward | one per forward | state-once (text/state media); assets/Box/track one row/problem | joint protocol v3; all tokens see all questions |
| Post-process | p(chosen) confidence | package: 1−H/ln n; vLLM: p(chosen) | box QMC confidence + calibration | (p1−1/n)/(1−1/n) |
| Vision | tower present, **serving is text-only** | yes (1.6 MP) | yes; images in options; video | yes (1.6 MP) |
| Context | — | 32K | 32K/131K | 32K validated / 262K max |
| T | 1.0 | 1.0 | 1.0 | 1.382 (post-hoc calibration; overridable) |

Contract facts from the artifacts:

- `decision_config.json` carries `codes`, `token_ids`, `temperature`,
  `attention_mode`, `pooling`, `readout_dtype`. Readout row `i` maps to
  vocabulary token `token_ids[i]`. The code list **skips non-single-token
  pairs** (BQ, CJ, ...); never compute codes base-26.
- AutoTrust's `decision_config.json` says `noncausal_full_attention`; its
  "StartLux prefix-bidirectional" means the whole prefill is the bidirectional
  prefix. It is served with no prefix caching and no chunked prefill. There is
  no state cache for AutoTrust: its state rows depend on the question.
- d3-turbo's `prefix_bidir`: the state (system + media + `State:`) is a
  bidirectional prefix; the question block attends to the state and to itself
  **bidirectionally** (both passes are unmasked). The mask is all-true except
  padding. A causal question block is the wrong model.
- d3-turbo state-once covers text and state-media requests only. Requests with
  `assets`, Box or track run one row per question under the block mask with no
  shared compute; an image token inside a question block errors on the
  state-once path.
- pplx joint: one sequence with the state and every question; every token,
  including the state, sees every question. Order and numbering matter.
- d3-turbo Box/track: prompts append `top:\nbottom:\nleft:\nright:` (track adds
  `visible:` first); readout at each colon token by offset from the end. Edges
  use the first 128 codes, `visible` the first 2; Box confidence uses QMC
  IoU≥0.5 over 512 R4 points with a piecewise-linear calibration.
- pplx readout logits are BF16 in the reference; an fp32 engine will differ
  near ties.
- d3/d3-turbo weight sharing is **unlikely**: different scores, different
  capabilities, different build times; the API alias is not weight sharing.
  Fetch d3's `MODEL_MANIFEST.json` and compare hashes before planning one
  resident weight set.
- Upstream serving: d3 is native on vLLM main (PR #61565; not in 0.31.0) but its
  CUDA W4 path is unverified; pplx ships its own reference/fast package; d3-turbo
  is served by vLLM SR Engine. Vendor throughput numbers are MI325X/GB200
  claims, not 4090 baselines.

## 2. Current engine

Verified: JEV W4A8 production 28/28 vs the CPU artifact golden, max Δ 0.0176,
KL 2.7e-4; 21/24/35–39/49–50/53–54 req/s c1–c16; 0.91/1.85/4.18 s at
4K/8K/16K; 21.6 GB peak. Prefill-only; no decode; exact-decision LRU only.
Norm handling already uses `(1+w)` for input/post/final/q/k norms and plain `w`
for the GDN gated norm. BF16 weights path exists for parity.

Gaps:

1. **Masks.** `fi_prefill.cu` is a one-shape causal shim
   (`native/third_party/flashinfer/fi_prefill.cu`, `kCausal`, 24/4/256,
   SinglePrefill, no LSE/varlen). Batched prefill does not use FlashInfer;
   attention launches per sequence on the native causal kernels. Noncausal
   Full and the bidirectional d3-turbo question block are not expressible.
2. **Readout.** The dynamic 255-row head, per-marker sites, Box/track
   multi-site readout and per-model post-processing are missing. The existing
   `logits_rows` fetches host rows on every call and runs over all sequence
   rows; it needs load-time extraction and an F32 GEMM variant.
3. **GDN state interface.** Ragged batching/conv reset per sequence is already
   done, but there is no initial-state/conv-history input, no final-state/tail
   output, and no cascade attention for d3-turbo state-once.
4. **Prompts/protocols.** Chat templates (`enable_thinking=False`), the pplx
   joint protocol (numbering, `\n\n`, escaping, semantic bisection) and Box/track
   composition are missing.
5. **Config.** `mrope_section`/`mrope_interleaved` are not parsed; 3-D positions
   are needed for images.
6. **Vision** (d3/d3-turbo/pplx; not AutoTrust).
7. **Quantization** recalibration under the real mask; the current AutoTrust
   W4A16 is asymmetric **group 64**, a different and less-tested kernel path than
   JEV's production group 128.

## 3. Architecture: one runtime, models as data

```
ModelCard / manifest (TOML, pinned + hashed)
      │ compiler
      ▼
Program (IR, content-hashed) ──► Runtime ──► ForwardJob(s) ──► kernels
```

IR: `ForwardJob { tokens, pos3, segments, mask, vision, readout sites,
cache_key }`; `MaskSpec { Causal | Full | BlockCascade { shared, group } }` — the
d3-turbo model is `BlockCascade` with **both parts noncausal**, not a causal
suffix; `HeadMatrix { file | lm_head[ids], dtype, F32 GEMM, bias, T, code_ids }`;
`Postprocess` per model. Adapters render prompts and emit jobs; they never touch
CUDA. The GDN ignores the mask and stays causal. Fail closed: no manifest, 404;
never fall back to causal + `lm_head`. Program hash is the cache key and parity
id. Readout sites for pplx come from `joint_config.json`, never from the
`pooling` field (which says `last` despite marker readout).

## 4. Attention

| Mode | Implementation |
|---|---|
| Causal (JEV only) | `kCausal` |
| Full (AutoTrust, d3, pplx) | `kNone` over the whole key range; GDN causal |
| BlockCascade (d3-turbo) | state prefix Q vs state K/V `kNone`; question Q vs [state + own question block] `kNone`; combine with `merge_state` on the LSE. The question block is **bidirectional**, not causal. |

- Ragged varlen FlashInfer `{kNone, kCausal}` with LSE, one launch per mask
  class per layer; this replaces the per-sequence native launches and is the
  prerequisite for state-once.
- GDN: add initial-state and conv-history inputs; final-state and conv-tail
  outputs. The conv tail is the last 3 **pre-conv** `in_proj_qkv` rows
  (10,240 channels, bf16, left-zero-padded under 3 tokens). Each question row
  carries its own state copy (~3.1 MB/layer, transient, ~200 MB for 64 rows);
  "never fork the GDN state" is wrong — the recurrence must be seeded.
- Positions: cached state K is already rotated; question tokens use
  `state_len + j` (or `max(state pos)+1+j` with mRoPE). Split state/questions by
  the full prompt's offset mapping; a straddling token belongs to the question.
- FlashInfer build list: head_dim 256, GQA 6, bf16, `{kNone, kCausal}`, LSE;
  include the cascade pair.
- Full attention cannot skip the upper triangle: it costs ~2× causal. JEV's
  4K/8K/16K timings are not a d3 baseline.
- Correctness traps: mask differs from training; conv tail dropped; packed
  GDN leakage; padding entering state/keys; cached-K RoPE; GQA `q//6`; split-K
  nondeterminism in the readout GEMM. The engine's GQA, GDN `vh/rep` + q/k L2
  norm and NeoX RoPE are already correct — keep them as tests.

## 5. Readout

`HeadMatrix` sources: JEV `lm_head[24]`; AutoTrust folded `lm_head[255]` (bf16
rows; other rows zero; no F32 head ships); d3/d3-turbo F32 file; pplx BF16 file.
At load, pre-extract the rows into a dense `[K,5120]` block; final-norm only the
readout rows; gather sites; F32 GEMM `[R,5120]×[5120,K]`; K ≤ 255.

Sites: `last`; per-marker `<|fim_suffix|>` (assert count == questions); Box/track
colon tokens (4 or 5 sites per question). Option codes ≥ n → −inf/ sliced before
`softmax(z/T)`. Cache pre-T logits; pplx's 1.382 is the calibration T and is
overridable. pplx reference rounds logits to bf16 — measure or emulate near ties.

Post-processing per model (do not genericize):

| | choice confidence | noul | score | box |
|---|---|---|---|---|
| d3 package | 1 − H/ln n | `{noul}` | Σ i·pᵢ, legend, entropy confidence; 2..10 | — |
| d3 vLLM `/v1/systemone` | p(chosen) | adds yes/no probs + confidence | — | — |
| AutoTrust | p(chosen) | `{noul}` | — | — |
| pplx | (p1−1/n)/(1−1/n) | `{noul}` | 1 − E\|i−best\|/baseline; 2..255 | — |
| d3-turbo Box | QMC IoU≥0.5 (512 R4) → calibration | — | — | edges first 128 codes; visible first 2 |

Choose the d3 answer shape per model id (package vs vLLM); they disagree.
Rewrite the existing `logits_rows` path: load-time extraction, F32 GEMM, sites
tensor, no full-sequence GEMM.

## 6. Prompts and protocols

- AutoTrust/d3 share the upstream code-readout strings but not the format id;
  add a byte-level render diff. Hard-code `enable_thinking=False` (the template
  otherwise injects reasoning). Do not tokenize state and question separately;
  split by offsets.
- pplx is a different protocol: different system prompt, `Question {index}:`
  numbering, `\n\n` joins, no closing line, images first, special tokens in user
  text escaped as full-width brackets. Over-length requests are bisected
  semantically; `max_length` is part of the program hash and cache key. Order is
  semantically significant. Add `permutation_average`, video (2 fps, 4–32
  frames, 16K cap), assets/questions images, per-question
  `max_length_exceeded` (d3-turbo) vs whole-request `LengthError` (pplx), and
  usage accounting.
- API: keep `/v1/decide` for JEV; add `/v1/systemone` as a pinned conformance
  clone (record fixtures from vLLM main; do not hand-write the schema);
  `/v1/models` metadata; startup golden self-test; LiteLLM routes per model id.

## 7. Cache

| Mask | Per-question cache | State/prefix cache |
|---|---|---|
| Causal JEV | — | valid (not needed) |
| Full AutoTrust/d3 | valid (one question per forward) | **invalid** (state depends on the question) |
| Full pplx | **never** (all tokens see all questions) | invalid past layer 2 (~5%, not worth it) |
| BlockCascade d3-turbo | valid for text/state-media | **valid** (state rows never see questions); not for assets/Box/track |

pplx caches whole chunks keyed on exact tokens/images/`max_length` and order.
d3-turbo state cache: paged KV + frozen GDN state + conv tails; fork with shared
page tables + per-row transient state copies; question blocks all bidirectional.
Costs: 64 KB/token full-attention K/V; GDN state ~151 MB/entry; host-RAM LRU,
restore ~60 ms/4K prefix vs ~900 ms recompute. Never duplicate the prefix KV.

## 8. Quantization

- On the 4090 only W4A16/W4A8 fit (FP8 27B is ~27 GB). The readout, vision and
  embeddings stay unquantized.
- **Recalibrate AWQ/GPTQ under the real mask** using the reference hooks (d3
  `PromptEncoder`, pplx `enable_noncausal_full_attention`); d3-turbo
  calibration must include short state-once question blocks.
- Isolate the **W4A16 asymmetric group-64** path first on AutoTrust — it is not
  the production g128 JEV path. Validate per family at 255-way with margin
  histograms/flip rates, not top-1 alone.

## 9. Parity program (P0)

The journaled AutoTrust measurement is **engine vs HF, both causal: KL 6.65e-2,
top-1 11/16 on 16 rows** — not 0.15, and the mask cannot explain it because both
sides were causal. Use ≥256 rows.

1. **Teacher-forced per-layer parity on CPU** (BF16 27B needs ~52 GB; the 4090
   cannot hold both): feed HF's captured layer-i input into engine layer i with
   one layer resident; compare relL2 and norm ratio (**not** L2-normalized cos:
   cosine is already scale-invariant and that step is a no-op) against the JEV
   envelope. CPU goldens on the 122 GB host.
2. Isolate the **g64 asymmetric W4A16** kernel against a BF16 single-layer run;
   then enable fused kernels one at a time.
3. Only then compare the HF reference with the correct noncausal hook against
   the engine with the mask on.
4. Gates: ≥256 rows, KL ≤ 3e-3, top-1 ≥ 99.5%, JEV still 28/28 and Δ ≤ 0.0176.

## 10. Benchmarks and gates

Same 4090, same artifact, vLLM main where it actually runs; verify vLLM main
loads the W4 d3 artifact on CUDA and that the d3 CUDA path exists (upstream says
it is unverified). pplx's baseline is the package fast path, not vLLM. Report
TMAC/s so GPU differences cannot hide.

| Model | Gate |
|---|---|
| JEV | hold 28/28, Δ ≤ 0.0176; ≥1.2× vLLM c1/c8 |
| AutoTrust | KL ≤ 3e-3 vs correct-mask reference; ≥99.5% top-1; no state cache |
| d3 | match vLLM `/v1/systemone` within KL ≤ 0.01 (shape chosen per id) |
| d3-turbo | text state-once: identical argmax + max\|Δp\| ≤ 1e-3 vs the per-row path in the same engine (not bit-identical: GDN recurrence and Marlin M differ); Box/track multi-site; no state-once for assets/Box/track |
| pplx | joint decisions match the package; whole-chunk cache; 32K validated |

No free GPU: GPU1 runs JEV production (~21.8 GiB); GPU0 is full. Any 27B test
requires hot-swapping JEV, and that is the owner's call.

## 11. Phases

| Phase | Work | Gate |
|---|---|---|
| P0 | CPU teacher-forced layer goldens; g64 kernel isolation; AutoTrust causal-vs-causal; then correct-mask comparison | KL ≤ 3e-3; JEV unchanged |
| P1 | ragged FlashInfer `{kNone,kCausal}` + LSE; GDN initial/final state + conv tail; BlockCascade merge; mask unit tests | dense fp32 reference per mask; pack isolation; JEV no regression |
| P2 | AutoTrust/d3 text adapters, per-model post-processing, F32/bf16 heads, golden fixtures (package + vLLM) | ≥99.5% top-1, KL ≤ 3e-3 on 256 rows |
| P3 | pplx joint (markers, escaping, numbering, bisection, whole-chunk cache); d3-turbo text-only state-once | joint reference match; state-once argmax/Δp gate |
| P4 | vision sidecar, 3-D mRoPE, images in options, Box/track multi-site + QMC/calibration, video | image/box/track goldens |
| P5 | quant recalibration under the real mask; CUDA graphs; same-hardware benchmarks | ≥97% BF16 quality; benchmark table |
| P6 | margin-calibrated cascade | accuracy cost measured |

## 12. Risks and open questions

Risks: mask semantics (critical); g64 kernel path (high); conv tail/state seeding
(critical); Full attention 2× cost and 24 GB limits (high); 255-way quant
margins (high); pplx order/whole-chunk cache (high); one 27B per GPU / no free
GPU (high); vLLM W4 CUDA unverified (med); vision (high); cascade drift (med).

Resolved: full layers are 3,7,...,63; pplx per-question caching is never valid;
d3/d3-turbo weight sharing is unlikely (verify hashes); pplx readout sites come
from `joint_config`, not `pooling`; GDN state seeding is required.

Open: d3 answer shape (package vs vLLM); d3/d3-turbo manifest hash comparison;
GDN chunked vs serial implementation; the exact vLLM CUDA/W4 d3 support; Box
calibration/`track` window details; whether the 32K cap is acceptable for pplx
answers (bisection changes them).

## 13. Sources and consultations

- Checkpoints and cards: `vllm-sr/d3`, `vllm-sr/d3-turbo`, `vllm-sr/d3-flash`,
  `vllm-sr/d3-edge`, `perplexity-ai/pplx-decider-v1.2`,
  `pplx-decider-flash-v1.2`, `pplx-decider-v1.1-27b`,
  `autotrust/AutoTrust-JEV-27B-v1.1`.
- vLLM PR #61565, vLLM SR Engine docs, decision.vllm-sr.ai API docs, pplx
  `source/src/pplx_decider`.
- Engine: `native/src/engine.rs`, `native/src/kernels.cu`, `native/src/config.rs`,
  `native/third_party/flashinfer/fi_prefill.cu`; `docs/JOURNAL.md`.
- Consultations: Claude Code CLI `claude-opus-5-5` reviewed this spec and the
  checkpoint code (`/tmp/consult-opus2.md`); its C1–C7 corrections are
  incorporated. A second harness-model review is in `/tmp/consult-harness.md`.
  The `codex`/`gpt-astra` CLI is not installed on this host. Vendor throughput
  claims must be reproduced on the same 4090.

# RTX 4090 optimization journal

Date: 2026-10-25. Starting commit: `92c97f5` / research review commit
`5175746`. Host: 2 x RTX 4090, 600 W limits, 32-core CPU, 122 GB RAM.
Production checkpoint: `/srv/models/d3-turbo-w4a8-vision-v2` (W4A8 language,
BF16 vision).

This is the working journal for the review in
`2026-10-24-rtx4090-performance.md`. Every experiment records the exact source
state, flags, raw result, and accept/reject decision. Production GPU1 is
stopped only while a benchmark needs the full 24 GB.

## Rules

- Match power and clocks before comparing; no experiment is accepted on an
  instrumented-region number alone.
- Keep numerical changes isolated; run the 32-row HF gate, state-once
  self-test, and image parity before retaining a change.
- Reject any change below the ≥2% end-to-end or ≥5% targeted-region threshold
  unless it materially improves memory/latency tails.
- Do not compare HTTP, engine, kernel, or instrumented timings as if they were
  the same denominator.

## Work plan

| Stage | Item | Status |
|---|---|---|
| 0 | Distinct-question suite and reproducible baseline | in progress |
| 1 | Readout/metadata cleanup; seeded convolution; disposable GDN state | pending |
| 2 | Selective GDN FMA; Marlin tile/resource experiments | pending |
| 3 | Batched attention; activation reuse/fusion | pending |
| 4 | Short WY GDN; cross-state batching; graphs | pending |
| 5 | Quantization/attention-cache alternatives; vision/host cache | pending |

## Environment baseline

Recorded at the start of the session:

- GPU0: 16,756 MiB used, 468.9 W, 2925 MHz (vLLM tenant).
- GPU1: Decidra v2, 15,407 MiB used, 13.1 W idle, 36 C.
- Both cards: 600 W persistent cap.
- `nvcc`, `ncu`, `nsys` are absent from the host PATH but available in the
  local `nvidia/cuda:13.1.1-devel-ubuntu24.04` container.
- Rust release binary: `target/release/decidra`.

## Experiment log

### E0 — Baseline and fixture audit

Pending: create `d3_distinct_2k_64q.json`, record duplicate-suite and
distinct-suite HTTP/engine numbers, and save raw JSON.


### E0 — Distinct-question baseline (accepted fixture)

Built `eval/data/d3_distinct_2k_64q.json`: state 2,121 tokens, 64 unique
question blocks, 3,334 question tokens (range 48–58). The old
`d3_once_64_2k.json` has 2,048 state tokens and 3,236 question tokens but only
**28 unique** blocks.

Engine-only baseline, production flags, 600 W, SM89, repeat 20:

| fixture | state | rows | tokens | p50 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| duplicate control | 2,048 | 64 | 3,236 | **721.91 ms** | 721.03 | 731.16 |
| distinct questions | 2,121 | 64 | 3,334 | **752.69 ms** | 752.32 | 754.12 |

The duplicate fixture understates the distinct-question pass by **4.3%**;
token count explains about 3.0%. Use the distinct fixture for all subsequent
comparisons.

Fresh `DECIDRA_PROFILE=1` region profile for the distinct pass:

| region | ms | share |
|---|---:|---:|
| MLP gate/up/down | 327.98 | 46.1% |
| GDN recurrence | 95.09 | 13.4% |
| GDN in_qkv/z/out | 112.34 | 15.8% |
| full attention + split/gate | 44.73 | 6.3% |
| embed | 18.30 | 2.6% |
| GDN prepare + norm + conv | 30.60 | 4.3% |
| other measured | 80.36 | 11.3% |
| **sum** | **710.39** | |

The engine p50 of 752.69 ms includes uninstrumented work around the 710.39 ms
region sum. The historical 685 ms profile and the 929 ms HTTP number are
different fixtures/denominators, as the research review warned.

### E1 — GPU readout conversion (rejected)

Moved the post-final-norm BF16→F32 cast and hidden upload off the CPU:
`bf16_to_f32` kernel + F32 GEMM. Distinct p50 **753.04 ms** versus 752.69 ms
baseline (+0.05%, noise), 32/32 text gate unchanged. The final-norm/readout
region is only 0.05 ms in the profile, so the host round-trip is not material.
Reverted.

### E2 — Global FMA compilation (rejected)

Added `DECIDRA_NVCC_FMAD` and built a separate `target-fma` with `-fmad=true`.
Distinct p50 **735.25 ms** versus 734.34 ms for the same GPU-embed baseline;
32/32 gate passed (max Δp 0.0897). No measurable end-to-end win from global
FMA. The static 561→443 instruction reduction in the recurrent kernel does not
survive the wider memory-bound pass. Rejected; keep `-fmad=false` and, if
revisited, target only recurrent arithmetic with a separate specialization.

### E3 — GPU embedding table (accepted, opt-in)

`DECIDRA_GPU_EMBED=1` keeps the 2.37 GiB embedding table resident. Distinct
p50 **734.34 ms** versus 752.69 ms (-2.44%, -18.4 ms); 32/32 gate unchanged.
Device free drops 9.54→7.20 GiB before state/scratch allocation. Retain it as
a production option only if the state cache still fits; it is not made the
unconditional default.

### E4 — Marlin M chunk (rejected)

Whole-M (`DECIDRA_MARLIN_CHUNK=0`, default) remains best:

| chunk | p50 |
|---|---:|
| 0 (whole M) | **734.34 ms** |
| 1,024 | 742.75 ms |
| 512 | 751.16 ms |
| 128 | 824.95 ms |

The persistent whole-M launch wins. Tile/stage/register specializations need
new cubins; this experiment only tests existing launch splits.

### E5 — Direct seeded convolution (accepted)

Added a tiled seeded convolution that reads the state's three history rows
directly and writes compact question rows. It removes 9,216 D2D copy calls per
64-question pass, the gapped buffer, and the compact copy. First non-tiled
version was slower (752.89 ms); tiled version with the correct 16-token grid:

- Distinct p50: **707.82 ms** versus 734.34 ms GPU-embed / 752.69 ms baseline.
- **-3.6%** versus GPU-embed baseline; **-5.96%** versus stock baseline.
- 32/32 text gate unchanged; max Δp 0.0984; mean KL 0.0044.
- `gdn.conv` region remains parallel (16-token tiles, one thread per channel).

Accepted. Next: remove the now-unused non-tiled seeded kernel and keep only the
tiled specialization.

### E6 — Reusable FlashInfer split workspace (small accept)

Cached the `cudaDeviceGetAttribute` result and reused a bounded workspace
instead of `cudaMallocAsync`/`cudaFreeAsync` per question attention call.
Seeded-conv path with GPU embed: **706.73 ms** versus 707.82 ms without the
workspace (-1.1 ms, ~0.15%). Workspace is a few MiB, not the 2.37 GiB embedding
table; retained, but not counted as a major win. Text gate unchanged.

### E7 — Immutable-seed GDN, no final state (accepted)

Question blocks now load the cached GDN seed directly into registers and never
write a final recurrent state. The 64×48×128×128 F32 scratch state (192 MiB at
B=64) and its broadcast are skipped for the short-question path.

- Distinct no-embed p50: **716.38 ms** versus 725.89 ms (-1.4%).
- Distinct GPU-embed p50: **697.39 ms** versus 706.73 ms (-1.3%).
- 32/32 text gate unchanged; max Δp 0.0984.
- Saves **192 MiB scratch plus 192 MiB broadcast writes** per question pass.

Accepted. Longer questions (>256 tokens) keep the state-producing kernels.

### E8 — Compact Q/K at key heads (accepted)

Question preparation now normalizes Q/K once per 16 key heads instead of
duplicating each across three value heads. Q/K buffers drop from
`[rows,48,128]` to `[rows,16,128]` each (about 109 MiB saved at 3,334 rows).
The seeded recurrence indexes `key_head = value_head / 3`.

- Distinct no-embed p50: **709.65 ms** versus 716.38 ms (-0.94%).
- 32/32 gate unchanged.
- Memory savings are material for long-context and image state caches.

### E9 — GDN decay exp hoist (rejected)

Precomputing `exp(g)` in preparation and storing it in the `g` buffer measured
709.46 ms versus 709.65 ms, inside noise. Reverted to keep log-decays and a
single semantic representation.

### E10 — Marlin N=128 specializations (invalid, rejected)

Compiled group-64 INT8 `thread_n_blocks=8` variants (with tmb 1–4). The
uncorrected dispatch reached **401.06 ms** but failed the text gate **6/32**;
outputs are not numerically valid with the current repacked-weight layout.
Simply changing the template tile is not a drop-in optimization. Removed the
variants and dispatch. This is the highest-upside remaining kernel experiment,
but it requires a matching repack/layout and a shape-level correctness test.

### E11 — Marlin M-block cap and chunk sweep (rejected)

Forcing the existing `thread_m_blocks` 1–3 caps makes the pass much slower
(default tmb4 709.56 ms; tmb3 994.37; tmb2 1,154.11). M chunks 128/512/1024 were
also slower than whole-M (824.95/751.16/742.75 vs 734.34 on the GPU-embed
series). Keep whole-M and tmb4.

### E12 — W4A16 MLP variants (rejected)

Excluding MLP gate/up/down from INT8 group W4A8 raised the pass to
**993.42 ms**; excluding only down raised it to **809.97 ms** versus 709.26 ms
W4A8 baseline. The large-M W4A8 path is decisively correct for this workload.

### E13 — GPU embedding, memory tradeoff

GPU embedding saves 18.4 ms (-2.44%) but consumes 2.37 GiB. Under the
large-context/concurrency constraint it remains opt-in only; production keeps
host embedding and the full VRAM for state caches.


### E14 — Final accepted matrix

Distinct-question engine fixture (2,121 state, 64 unique questions,
3,334 rows), production W4A8 flags, repeat 20, 600 W:

| configuration | p50 | vs stock |
|---|---:|---:|
| stock `92c97f5` | 752.69 ms | — |
| accepted, host embedding (production) | **708.91 ms** | **-5.81%** |
| accepted + GPU embedding (opt-in) | **689.63 ms** | **-8.38%** |

Accepted changes: tiled seeded convolution, 64-slot awaited image admission
already in production, cached FlashInfer workspace, state-free seeded GDN,
compact Q/K at key heads. State capture is 416.7 ms for the 2,121-token state.
GPU embedding is not the production default: it costs 2.37 GiB and leaves
7.20 GiB free before state caches, which conflicts with the large-context goal.

HTTP on the same checkpoint, 2k state / 64 duplicate control (3 runs):

| path | p50 | q/s |
|---|---:|---:|
| stock `92c97f5` | 929 ms | 69 |
| accepted | **879.8 ms** | **73** |
| accepted c8 | **3.94 s** | **73** |

Distinct-question HTTP: 901.4 ms c1 / 71 q/s; c8 4.04 s. Image parity remains
**13/13** and the 32-row text gate remains **32/32**, max Δp 0.0984.

Memory reductions accepted:
- question GDN state: 192 MiB scratch + broadcast removed;
- Q/K: about 109 MiB removed;
- gapped convolution buffers/copies removed;
- attention workspace: a few MiB retained instead of per-call alloc/free.

### E15 — Marlin N=128 follow-up (rejected)

After adding tmb1–4 N=128 group-scale templates, the first dispatch produced
401 ms but only 6/32 text-gate decisions. Increasing the persistent lock array
from 128 to 512 did not fix it. A tmb4/N8/K8 variant produced 31/32 with one
unsafe flip, max Δp 0.0874; restricting N=128 to N=17,408, 10,240, 12,288,
6,144, or 5,120 modules all produced the same unsafe flip on
`choice-4dcab9a0`. This is not a valid drop-in: the current repacked weight /
scale layout and the N=128 template do not share the same numerical contract.
The variants and env dispatch were removed. The 1.7× apparent speedup is not
accepted; matching repack/scale layout is the highest-upside future kernel
project.

### E16 — Remaining ideas: measured ceiling / decision

| idea | evidence / ceiling | decision |
|---|---|---|
| Batched full attention | 39.7 ms region, 5.6% of measured pass; 1,024 calls/16 layers. A correct paged/LSE merge is substantial and the direct region ceiling is small at 2K. | deferred; workspace reuse accepted |
| Short WY GDN | recurrence 95.1 ms (13.4%); existing chunked path rejected for short blocks. Requires new triangular-solve kernel and numerical gate. | deferred (high risk, possible ~10–20 ms) |
| Cross-state batching | scheduler serializes distinct states; mixed 1–8q traffic underutilizes linears. Needs per-row cache pointers through attention/GDN. | deferred; architecture work |
| CUDA Graphs | CPU question-pass profile is 153 ms, but 147 ms is host embedding; GPU launch count not yet isolated (nsys importer failed). | deferred; no evidence |
| Layer-63 tail | layer 63 is 1/64 of model work; only final rows needed for logits/state capture K/V. | deferred (≤~1%) |
| Host-state pinned staging | restore-heavy long-context workload; no restore benchmark yet. | deferred |
| Vision feature cache | cold image 88 ms; exact image/state cache 34 ms. Cross-state image reuse needs a memory budget. | deferred under memory constraint |
| Quantized attention/KV | quality risk and head_dim256/GQA port; 2K attention is 5.6%. | deferred |
| Selected W4A16 / coarser activation scales | MLP W4A16 measured 993 ms, down-only 810 ms vs 709 ms. | rejected |
| Multi-replica Rx4090 | GPU0 currently occupied by vLLM; capacity option, not per-GPU efficiency. | not applicable now |


### E17 — Final-layer capture MLP skip (rejected)

Skipping the last layer's post-attention MLP during state capture preserved
K/V and passed the gate, but state capture measured 417.7–420.9 ms versus
416.7 ms in the accepted baseline: no repeatable gain above noise, because
only 1/64 of the MLP work is removed. Reverted to keep the path simple.

### E18 — Long-context check after the accepted changes

Production server, 64 duplicate-control questions, c1 one-shot:

| state | cold request (capture included) | warm exact-state cache | free VRAM after |
|---|---:|---:|---:|
| ~8K | 2,936 ms | **998 ms** | — |
| ~24K | 8,206 ms | **1,242 ms** | 6.1 GiB |

Warm 64q latency is 998 ms at 8K and 1,242 ms at 24K. The accepted kernels
reduce scratch memory and keep the 24K cache resident within the 24 GiB card.
This is the relevant large-context path; the old long-context measurements in
ENGINE §5 were taken under a different release/power configuration.

### E19 — Scoped N=128 attempt (rejected)

Because the N=128 K4 variant was fast but wrong, a K8/N8 tmb4 variant was
tested by layer range. Text-gate results:

| layers using N128 | top-1 | max Δp | unsafe flip |
|---|---:|---:|---|
| 0–15 | 31/32 | 0.0896 | `choice-4dcab9a0` |
| 16–31 | **32/32** | 0.0830 | none |
| 32–47 | 31/32 | 0.1018 | `choice-4dcab9a0` |
| 48–63 | **32/32** | 0.0993 | none |
| 0–31 | 31/32 | 0.0964 | yes |
| 32–63 | 31/32 | 0.0974 | yes |
| all | 31/32 | 0.0874 | yes |
| 16–31 + 48–63 | **32/32** | 0.0780 | none |

The correctness-safe 16–31+48–63 combination measured **726.01 ms**, slower
than the 708.91 ms accepted path. The only fast N=128 variant was the invalid
K4 layout. N=128 is rejected: it is both a layout/quality risk and, in its
validated K8 form, slower on this workload.

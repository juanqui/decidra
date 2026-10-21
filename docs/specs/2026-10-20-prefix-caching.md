# Prefix caching specification

Status: revised after two `claude-opus-5-5` reviews (2026-10-20), both recorded in
§12; the second round's fixes are applied here and the verdict was *approve with
the listed fixes*. Scope: `native/src/engine.rs` and the surrounding server, for
the five decision families in
[`2026-10-20-decision-model-support.md`](2026-10-20-decision-model-support.md).
Design + test plan only; this commit changes no engine code.

## 1. Goals and non-goals

**Goals.**

1. Make the d3-turbo `state-once` program pay: one bidirectional state pass, then
   64 question rows, instead of 64 full prefills. At a 4K state that replaces
   64 × 0.833 s ≈ 53 s of prefill with one state pass plus a 4K-equivalent
   question pass (§7).
2. Make any repeated prompt cheap **regardless of mask** by caching the
   post-final-norm hidden at the readout sites (Tier 1). Changing `head_ids` or
   asking for `--full-logits` on the same prompt must not re-run the backbone.
3. Cache only where the mask proves the prefix output is question-independent,
   and *fail closed* everywhere else (§5). A cache that serves a
   question-dependent prefix is a wrong answer with no visible symptom.
4. Keep the shipped JEV path bit-identical when the feature is off, and keep peak
   memory inside the 21.6 GB production envelope (~1.5 GB headroom on GPU1).

**Non-goals.** Decode/streaming KV growth (there is no decode loop). Chunked
prefill as a latency mechanism — vLLM needs it to interleave decode with prefill;
with no decode phase a chunk is a scheduling choice, not a cache requirement.
Beam search. On-disk or cross-process cache persistence (§4.9). Vision encoding:
the native server rejects image states today (`src/api.rs:268`, `:351`) and
`mrope_section` is unparsed, so Tier 3's keys are reserved and nothing is
implemented.

## 2. Verified engine facts

| Fact | Where |
|---|---|
| Prefill-only; one stream; `Engine` is `Send`, not concurrent | `native/src/engine.rs:309-400`, `native/src/lib.rs:1-17` |
| Buffers come from a size-keyed pool retaining ≤ 2 per size, released to the driver above 12,288 tokens (`low_mem`) | `engine.rs:22-40`, `:114-127` |
| Batching flattens sequences into one `[total, …]` buffer with `offsets`/`lens` | `engine.rs:409-449` |
| RoPE positions are rebuilt per sequence as `0..len` — **no start offset exists** | `engine.rs:1228-1253`, `:449-458` |
| The full-attention mask is a global bool honoured **only** on the FlashInfer branch | `native/src/config.rs:27`, `engine.rs:313`, `:1533`, `:2269` |
| FlashInfer is used only at `batch == 1` **and** `seq ≥ flashinfer_min_seq` (default 1024) | `engine.rs:664-681`, `:1524-1531` |
| The shim hardcodes `qo_len == kv_len == seq` and `lse = nullptr` | `third_party/flashinfer/fi_prefill.cu:20-46`, `:17` |
| FA2 causal mask is `kv_idx + qo_len > kv_len + q_idx`; `kCausal` rejects **`kv_len < qo_len`** and accepts longer KV | `include/flashinfer/attention/prefill.cuh:1707`, `:2790` |
| FA2 writes **log2-domain LSE**: `ptx_log2(d) + m`, `m` in `sm_scale·log2e` units | `prefill.cuh:2722-2730` |
| `attention_tc` works in log2 (`scale_l2`, `exp2f`); `attention_warp` works in natural log (`expf`) | `kernels.cu:2182-2185`, `:486-514` |
| Both native attention kernels assume `kv == qo`: `attention_warp` loops `sx <= qpos`; `attention_tc` sets `nkeys` from `seq` only | `kernels.cu:486`, `:2124-2185` |
| GDN recurrent/shared/split kernels — **including the batched forms** — read their state slot at entry and **write it back at exit**, indexed per row | `kernels.cu:1029-1031`, `:1081-1086`, `:1925-1935`, `:1878-1883` |
| Only the chunked kernels zero the state at entry | `kernels.cu:1155`, `:1492` |
| `gdn_recurrent` (legacy) works **in place in global memory inside the token loop** — it cannot skip its write-back | `kernels.cu:621` |
| Conv is depthwise, kernel 4, causal restart per sequence with zero padding before `t=0`; **no history input** | `kernels.cu:1796-1830`, `:1834-1852` |
| Four conv variants exist: `conv1d_silu_{,tiled_}{,batched_}` | `native/src/kernels.rs:92`, `:159-160` |
| GDN conv width is 10,240 = 2·2048 + 6144; state is 48 heads × 128 × 128 f32 per layer | `engine.rs:1592`, `:1665` |
| Readout uses row `b.offsets[i] + b.lens[i] - 1`; the batched readout uses `lm_head` rows only | `engine.rs:1941`, `:1922`, `:3097` |
| Exact-decision LRU key is `(tokens, head ids)` — no program, path or mask | `src/cache.rs:14-40` |
| `set_*` knobs take `&self` and store atomics; GDN/attention/conv/fused knobs apply on **every** forward | `engine.rs:632-760` |
| `set_lora` takes `&mut self` and has no caller outside `Engine`; LoRA binds per module at load | `engine.rs:590`, `weights.rs:425` |
| Metrics are Prometheus; existing names `decidra_decision_cache_{hits,misses}_total`, `…_entries` | `src/native_backend.rs:205-243` |
| Shipped: 18.83 GiB idle, 20.11 GiB flat at 16K, 4K 0.833 s, 16K 3.740 s | `docs/VLLM-COMPARISON.md:58-69`, `:98-100` |

**Sizes.** Full-attention K/V = 16 layers × 4 kv heads × 256 × 2 (K,V) × 2 B =
**64 KiB/token** (256 MiB at 4K, 1 GiB at 16K, 4 KiB/token/layer). Per-layer K/V
for Σq = 4,096 question rows is **16 MiB**. The frozen GDN state is
48 × 48 × 128 × 128 × 4 B = **144 MiB (151 MB)** per checkpoint, independent of
prefix length. Conv tails are 48 × 3 × 10,240 × 2 B = **2.8 MiB**. A state entry is
`P × 64 KiB + 147 MiB`: 4K = 403 MiB, 16K = 1.14 GiB.

### 2.1 Blocking prerequisite: mask dispatch is wrong today

`noncausal_full` is consulted only inside the FlashInfer branch
(`engine.rs:1533`, `:2269`). The fallback kernels are **causal-only** and take no
mask: `attention_warp_bf16` loops `for (sx = 0; sx <= qpos; …)` (`kernels.cu:486`)
and `attention_tc` applies `col > qpos` unconditionally (`:2182-2185`). So any
AutoTrust, d3 or pplx request that is shorter than 1,024 tokens, co-batched with
another request, or run with a failed `dlopen` is **silently served with a causal
mask**.

Fix, and the smaller one: for a non-causal `MaskSpec`, dispatch **every row
through FlashInfer at any batch size and any length**, in the per-sequence loop
that already exists (`engine.rs:1524-1531` only needs its `batch == 1` and
`seq >= min` conditions relaxed for non-causal masks). The native kernels then
stay causal-only and **assert** that they were not asked for another mask. This
removes the "mask we cannot express" case entirely instead of adding a
`kNone` variant to two tiled kernels.

This is independent of prefix caching and must land first: every cache gate below
compares the engine with itself, so caching wrong-mask hidden states would make
all of them pass.

## 3. What transfers from vLLM, and what does not

vLLM hashes each KV block over `(parent hash, block tokens, extra hashes)` where
the extras carry LoRA ids, multimodal input hashes and an optional `cache_salt`;
only full blocks are cached; blocks are refcounted and evicted LRU from a free
queue, and the request block table is append-only so duplicates persist until
free ([APC design](https://docs.vllm.ai/en/latest/design/prefix_caching/)). For
hybrid Mamba/GDN models it adds a second cache group whose state is stored only
on the Mamba block grid so a hit resumes at a block boundary
(`mamba_cache_mode=align`); `prefix_match_unit` separates hit granularity from
state-store granularity, `prefix_cache_retention_interval` sets periodic
checkpoints, `--enable-mamba-shared-prefix-checkpoint` adds one at the
shared-prefix junction, and `kv_offloading_size/backend` is the host tier
([hybrid Mamba APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/),
[`CacheConfig`](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)).

Transfers: block hashing and the free-queue/refcount eviction discipline **for
causal prefixes only**; the block-grid-versus-state-granularity split; the host
tier behind a byte budget; "only full units are cached". Does not transfer:
decode-time growth (the free queue exists to feed a growing decode KV),
`cache_salt` tenancy in v1, chunked prefill as a cache feature, content-addressed
block dedupe across *bidirectional* prefixes (§4.3), and fp8 KV. vLLM's baseline
ran `--kv-cache-dtype fp8` with 41,441 tokens of capacity
(`docs/VLLM-COMPARISON.md:24-28`), 2× our bytes per entry; JEV's gates
(max Δ ≤ 0.011) leave no room for kv-dtype quantization, so we stay BF16 and
forgo that capacity deliberately.

## 4. Design

### 4.1 Tier map

| Tier | Key | Payload | Placement | Valid masks |
|---|---|---|---|---|
| **T0** exact decision | `program_hash, exec_variant, tokens, head_ids` | final logits `[K]` | host | all |
| **T1** site hidden | `backbone_hash, exec_variant, tokens, image_ids, pos_ids, sites` | post-final-norm hidden at the sites, bf16 `[M,5120]` | host, byte budget | all |
| **T2** prefix K/V + GDN states + conv tails | `backbone_hash, exec_variant, prefix tokens, pos_ids, split` — **exact, whole prefix** | K/V (16 layers) + 48 states + 48 tails | host LRU, optional VRAM | Causal, BlockCascade (whole state only) |
| **T3** image features (reserved) | `vision_hash, image_sha256, resize/patch cfg` | visual embeddings + 3-D patch positions | host LRU | input-side, all masks |

T1 is the cheapest and the broadest: a `last`-site entry is 10 KiB, a 128-marker
pplx entry is 1.3 MiB. A T1 hit still runs the readout (255×5120 = 2.6 MFLOP; a
full-vocab probe is 4.2 GFLOP), so T1 turns a 0.9 s prefill into a readout. T1
stores *hidden*, hence it is independent of `head_ids`, `--full-logits` and `T`.

`exec_variant` is in every key because §4.5 shows the engine is only deterministic
per program **and per batch shape**: the executor path (single-sequence versus
batched) and the `M` class determine which GEMM and which GDN kernel runs. A value
produced by a `state-once` row must never be served to a per-row request.

### 4.2 Keys: `backbone_hash` and `program_hash`

Two hashes, because T1/T2 payloads do not depend on the readout and must not miss
when `T` or the code map changes:

```
backbone_hash = H("decidra-backbone-v1",
      weights_manifest_sha256        // artifact + per-module quant policy, plus config.json
    ‖ adapter_sha256 | "none"        //   and decision_config.json, as loaded
    ‖ tokenizer_sha256               // token ids are the cache currency
    ‖ mask_family                    // Causal | Full | BlockCascade  (NOT the split)
    ‖ numeric_config_hash)           // §4.2.1

program_hash  = H("decidra-program-v1", backbone_hash ‖ readout_sha256
                                        ‖ readout_dtype ‖ index_sha256)
```

All fields are **length-prefixed** before hashing: concatenating variable-length
fields makes the encoding ambiguous, and an ambiguous key is a collision
generator.

- **`mask_family` is mandatory; `split` is not in it.** Identical tokens under
  `kCausal` and `kNone` diverge at layer 3, so the family must be in the hash.
  The split is a per-request quantity and belongs in the T1/T2 keys; putting it in
  `backbone_hash` would make the whole T0 namespace request-dependent.
- **`pos_ids` is hashed, not derived.** Positions are `0..len` today, but 3-D
  mRoPE positions for images are not a function of the tokens, and a resumed
  suffix starts at `P`. `BatchInfo` gains `pos_start: Vec<usize>`.
- **`max_length`** lives in `index_sha256`: it reaches the model only through
  protocol truncation, which changes the tokens.
- **`adapter_sha256`**: `set_lora` has no caller outside `Engine` today, but the
  adapter is part of the function, so it is keyed rather than special-cased.

#### 4.2.1 `numeric_config_hash`

The hash covers every knob that changes logits, and completeness is *tested*: a
`NumericConfig` owns them, is frozen at the end of `preload_all`, and a unit test
fails when a field is added without hashing it. Current set: `marlin`,
`marlin_fp8`, `marlin_int8`, `int8_clip`, `int8_group_scale`, `int8_group_exclude`,
`int8_exclude`, `int8_all_layers`, `fused_w4a16`, `fused_min_n`, `fused_max_m`,
`fused_tile128`, `warp_attention`, `flashinfer_enabled`, `flashinfer_min_seq`,
`gdn_shared`, `gdn_split`, `gdn_split_min_seq`, `gdn_chunks`, `gdn_chunked`,
`gdn_chunked_tc`, `gdn_chunked_min_seq`, `conv_tiled`, `gpu_embed`, plus the
env-only `LazyLock` knobs `DECIDRA_MARLIN_CHUNK` (`engine.rs:233`) and
`DECIDRA_INT8_GROUP_UNIFORM` (`:244`) which no `set_*` call can see.

Two corrections that make "frozen" real rather than aspirational:

- The quant knobs bind at layer load (`engine.rs:852-862`) but the GDN, attention,
  conv and fused knobs are read on **every** forward (`:664`, `:1667-1688`,
  `:1601`). Flipping one after the freeze therefore changes numerics *without*
  changing the hash, which is the direction that must be impossible. So `set_*`
  must return an error once frozen, not merely be documented as load-time.
- The manifest hash must include `config.json` and `decision_config.json`
  (`mrope_section`, `layer_types`, `codes`, `token_ids`), which the engine parses
  at startup and which no existing flag covers.

### 4.3 Storage: contiguous entries; paging only where matching is valid

Under BlockCascade a state row attends bidirectionally to every later state token
from layer 3 on, so the K/V and GDN state of a **strict sub-prefix** of the state
are not a function of that sub-prefix (§5). Two consequences:

- **T2 matches whole prefixes exactly.** A BlockCascade entry is keyed on the
  entire state block `tokens[0..split)`. No partial hit, no promotion to an
  earlier checkpoint, no `RETENTION` checkpoint for BlockCascade.
- **BlockCascade entries are stored contiguous**, one `[P,4,256]` K and V pair per
  layer, with **no block hashing and no content-addressed dedupe**. The page
  table and hash chain are exactly what would reintroduce the bug in storage: two
  states sharing their first 3K tokens would dedupe physical blocks whose contents
  differ, and an exact-key lookup would not notice. Contiguous storage also
  removes the staging copy a launch would otherwise need.

The fork does not need block tables either: §4.6's cross term is one shared K/V
extent, so 64 rows share the arena by pointer, not by page table. Refcounting is
entry-level (one pin per in-flight request).

**Block hashing and paged storage are therefore scoped to Causal only**, where
prefix rows never see the suffix: 64-token matching, retention checkpoints,
promotion-to-nearest-checkpoint and free-queue eviction are all valid there. Value
is near zero today (JEV prompts are 73 tokens), so §9 defers it to P6 behind an
owner decision, and `DECIDRA_PREFIX_CACHE_RETENTION` is refused unless the mask
family is Causal.

K/V is stored **post-RoPE and post-q/k-norm**: capture K after `kernels::rope`
(`engine.rs:1478-1494`) and before `drop(kn)` (`:1561`); capture V from `vv`
(never normed). Capturing at `:1458` — where `kn` is only allocated — would cache
unrotated K, which cross-attending suffixes would then consume as if it were
rotated.

### 4.4 GDN state and conv tails

A checkpoint at `P` carries, **per layer**: the f32 recurrent state (the buffer
`gdn_batch` fills at `engine.rs:1665`), the bf16 conv tail (the last 3 **pre-conv**
`in_proj_qkv` rows × 10,240 channels, captured where `gdn_batch` drops `qkv` at
`:1660-1662`), and the position start `P`. Every GDN layer is seeded from *its own*
checkpoint; reading this as "seed the row once" would make layer ℓ+1 inherit layer
ℓ's state.

Kernel reality, stated correctly this time:

- Every recurrent/shared/split kernel, batched or not, **reads its state slot at
  entry and writes it back at exit** (`kernels.cu:1925-1935` for the batched
  shared form). Seeding therefore needs **no kernel change** — it is a copy into
  the row's slot — but *skipping the write-back* does.
- The chunked kernels zero the state at entry (`:1155`, `:1492`), so they need
  `s0` in place of the zero fill.
- `gdn_recurrent` (the legacy `--legacy-gdn` path) updates global state inside the
  token loop (`:621`) and cannot skip its write-back at all.

So the plan: add `const float* s0` (read-only, **one `[nv·dk·dv]` slice per layer**,
broadcast to all rows) and `float* s_out` to the shared, split and chunked
families and their batched forms. With `s_out == nullptr` a Stage B row keeps its
state in shared memory and never writes, which is correct because a prefill-only
suffix never needs its final state. A fork is then **refused** when
`gdn_shared == false`, i.e. when the selected kernel cannot express it — an error,
not a silent fallback.

Budget honestly: without that kernel change the fork must allocate per-row state
slots, `batch × 3 MiB = 192 MiB` for the current layer (the buffer is allocated
per `gdn_batch` call and recycled by the pool, so it is 192 MiB live, not
9.4 GiB). §4.10 budgets both cases.

**The conv-tail failure mode is not "three wrong tokens".** The three corrupted
conv outputs enter the recurrence at all 48 GDN layers and are carried forward,
then spread further by bidirectional attention. The error is small and decayed, so
it can pass a 1e-3 probability gate while being wrong — which is why §8 tests it
at the kernel level with a bit comparison.

`hist` must be added to **all four** conv variants (`conv1d_silu_bf16`,
`conv1d_silu_batched_bf16`, `conv1d_silu_tiled_bf16`,
`conv1d_silu_tiled_batched_bf16`), or a kernel lacking it must reject a request
that sets `hist`. A state shorter than 3 tokens is left-zero-padded, which is the
same padding the kernels apply at `t = 0` (`kernels.cu:1815-1817`).

### 4.5 Stage A is a solo, fixed-path computation

Cache bytes depend on batch composition today:

- **GEMM shape**: cuBLAS heuristics, Marlin's `pm <= 8` / `tmb` selection
  (`engine.rs:2977-2980`), the fused cut-off (`:3003-3005`) all depend on `M`.
- **GDN kernel choice**: `all_long` / `all_short` / `all_chunked` are computed over
  the whole batch (`:1667-1688`), so a batch-mate's length changes a row's kernel.
- **Attention**: FlashInfer is disabled whenever `batch > 1` (`:1527`).
- **Executor path**: a one-item batch under 1,024 tokens takes the single-sequence
  path (`decision_logits` → `run_layer`/`gdn`, `:3097`, `:2361`), everything else
  the batched path (`src/native_backend.rs:47-62`).

Rules: Stage A always runs as a one-item batch on the batched path; an entry
records the producing `exec_variant`; a hit whose variant differs is a miss.
Prefix-affinity grouping in the batcher (§6.4) is excluded from bit-equality
tests. The invariant, stated honestly: **T0/T1/T2 values are deterministic per
program and per batch shape, not per request.**

### 4.6 Forks: Stage A once, Stage B = cross + self + log2 merge

d3-turbo `state-once` is one Stage A over the state prefix, then up to 64 Stage B
question blocks, where a question token sees the whole state plus its own block
bidirectionally. Per attention layer, Stage B is:

- **Cross**: Q = all question rows flattened (`Σq` rows) against the single state
  K/V extent, `kNone`. One launch — every question row sees *all* state tokens, so
  a shared extent is exactly right.
- **Self**: per row, Q = its own block against its own block, `kNone`.
- **Merge**: `out = merge(o_cross, lse_cross, o_self, lse_self)`, applied **before**
  `sigmoid_gate` and `o_proj` (`engine.rs:1563-1580`), because the gate and the
  output projection act on the merged attention.

Softmax is associative over disjoint key sets, so this equals one `kNone` launch
over `[state ‖ own block]` up to reduction order.

Both terms go through **FlashInfer**, including the 64 tiny self-term launches:
they are one launch each, batch-independent, and this keeps the native kernels
causal-only (§2.1) instead of teaching two tiled kernels a second mask. Cost is
1,024 launches per fork at ~5-10 µs, i.e. 5-10 ms against 53 s saved.

Two traps:

- **FA2's LSE is log2-domain** (`ptx_log2(d) + m`, `prefill.cuh:2722-2730`) with
  `m` already scaled by `sm_scale·log2e`. A merge written with `exp`/`log` weights
  the terms wrongly and still produces a plausible answer. Use `exp2`. And because
  `attention_tc` is log2 while `attention_warp` is natural log
  (`kernels.cu:2182-2185` vs `:486-514`), **every emitter must be pinned to log2 in
  `sm_scale·log2e` units** if either is ever used in a merge.
- **bf16 partial outputs cannot support a tight merge test.** A bf16 `o_cross`
  plus a bf16 `o_self` cannot agree with a single f32-accumulated launch to 1e-5,
  so the tolerance would be loosened until the natural-log negative control stops
  separating. The merge path therefore needs an **f32 partial-output variant** of
  the shim (`SinglePrefillParams<bf16, bf16, float>`), used at least by the test.

**Per-row split.** The sibling spec forbids tokenizing state and question
separately, so `split_r` comes from each row's full-prompt offsets, and BPE can
merge across the boundary for some questions but not others. A row with
`split_r < P`, served from a Stage A whose bidirectional state includes that
token, is silently wrong on exactly those rows. So: compute `(state tokens,
split_r)` per row, fork only exact matches, and route the rest through a second
Stage A or the per-row program. T-SPLITVARY (§8.4) uses a merging boundary.

### 4.7 Host tier, restore versus recompute

Behind VRAM (a couple of entries at most), a host LRU over
`DECIDRA_PREFIX_CACHE_HOST_BYTES` (122 GB available; default 8 GiB → **20** four-K
entries or **7** sixteen-K). Restore is one `cudaMemcpyAsync` of
`P × 64 KiB + 147 MiB` on the engine stream. PCIe Gen4 x4 is ~7.9 GB/s raw, ~6 GB/s
measured — **re-measure and use the measured value in the gate.** At 6 GB/s versus
the measured 4,908 tok/s:

```
restore(P) ≈ P·0.0109 ms + 25.7 ms      (the 147 MiB state dominates below ~2K)
compute(P) ≈ P·0.204 ms
break-even P ≈ 133 tokens               (default MIN_TOKENS = 256)
```

Restore is synchronous on the single stream; the batch worker is already serialized
(`src/batch.rs`), so it costs no concurrency, but it delays what is queued behind
it and must appear in `decidra_prefix_restore_seconds`.

Host memory is pinned with `cudaHostAlloc` (file-backed mappings pin poorly), and
an entry may only be freed after the restore reading it completes: one
`CudaEvent` per restore, free on completion. `evict → memcpy` ordering without the
event is a use-after-free that appears as garbage logits under load.

### 4.8 Site-hidden tier (T1)

Store post-final-norm hidden rows at the requested sites, bf16 — the output of
`final_norm_rows` (`engine.rs:1255`) before the readout GEMM. `sites` is a tensor
of absolute positions, not a loop.

- **Placement and budget**: host memory under a **byte** budget. Entries range
  10 KiB–1.3 MiB (128×), so a count budget is meaningless.
- **Name and default**: `DECIDRA_SITE_HIDDEN_CACHE_BYTES`, default **0 (off)**,
  enabled per model by the manifest. JEV stays 0 until a re-baseline is accepted by
  the owner; every other family enables it once the readout is site-gathered.
- **Bit-identity is conditional.** Today the readout GEMM runs with `M = total`
  (`engine.rs:1935`), deliberately (`:1913`). A readout-only T1 job runs with
  `M = #sites`, which can select a different kernel. T1's bit gate therefore holds
  only where the readout is site-gathered (d3/d3-turbo/pplx, per the sibling
  spec); on JEV's all-rows readout T1 is a tolerance gate at best.
- **Site rule**: the key's `L` is the full sequence length for every non-causal
  mask. "Reject a site inside the cached prefix" is not coherent for an
  exact-token key: if `L` is the full length there is no prefix to be inside.
- **Bisection**: a pplx bisection retry changes the chunk's tokens, so it
  *misses*. What T1 buys is a different option set or `T` on the same tokens.
- **Latency**: a T1 hit still queues behind the FIFO batcher, so §7's 15 ms is a
  service time; report p95.

### 4.9 Eviction, refcounts, lifetime

- T0: LRU by entry count (`DECIDRA_DECISION_CACHE_SIZE`). T1: byte budget.
- T2: entry-level LRU with one pin per in-flight request; block-level refcount and
  free-queue eviction only in the Causal path (P6).
- **All cache memory is raw `CudaSlice`, never `Buf`/`F32Buf`.** `Buf::drop`
  returns buffers to the size-keyed pool (`engine.rs:114`) retaining up to two per
  size, so a cache entry stored in a `Buf` would inflate peak memory invisibly
  after eviction and could be handed back to a normal allocation. Eviction frees to
  the driver.
- Lifetime: process. No cross-restart persistence — on a multi-tenant box that is
  a prompt-content leak, which is why vLLM needs `cache_salt`. If it is ever added
  it requires a salt and a `backbone_hash` check on load.
- Any change to weights or numerics clears T0/T1/T2 in the same call.

### 4.10 Memory budget

| Item | Bytes |
|---|---|
| Resident weights + embedding + scratch (measured) | 18.83 GiB idle, 20.11 GiB flat at 16K |
| Production headroom on GPU1 at 21.6 GB | ~1.5 GB |
| 4K state entry | 256 MiB K/V + 144 MiB state + 2.8 MiB tails = 403 MiB |
| 16K state entry | 1 GiB + 147 MiB = 1.14 GiB |
| Per-layer K/V stage for Σq = 4,096 | 16 MiB |
| Cross-attention output + LSE (Σq = 4,096) | 50 MiB bf16 + 0.4 MiB f32 |
| Per-row state slots, if `s_out` is not implemented | 192 MiB (batch 64, one layer) |
| **Fork marginal over a plain 4K prefill** | **≈ 470 MiB** with the `s0`/`s_out` change, **≈ 660 MiB** without |

Three caveats that matter more than the table:

1. **Stage B is a second full pass** over Σq = 4,096 rows through every GEMM. Its
   activation footprint is that of another 4K prefill, and the measured 20.11 GiB
   peak was taken at 16K **with `low_mem` engaged** (≥ 12,288 tokens), where the
   pool releases instead of retaining. A 4K state plus a 4K question pass is 8,192
   tokens, *below* the threshold, so the pool retains — the mode in which the
   journal recorded 22.8 → 23.9 GiB growth. **The fork must force `low_mem`.**
2. **A 16K state does not fit.** Stage A alone peaks at 20.11 GiB and would hold a
   1.14 GiB entry on top: ≈ 21.25 GiB, over the 21.6 GB envelope with nothing left
   for Stage B. Set and measure a **maximum fork state length** (provisional 8K)
   and refuse longer states loudly.
3. `DECIDRA_PREFIX_CACHE_GPU_BYTES` defaults to 0 (host tier only). If a cache
   allocation fails, the cache degrades — but for `state-once` that silently
   changes the program (to per-row, ~25× slower, different numerics), so it must
   be reported via `decidra_prefix_cache_degraded_total` and a response field, or
   the fork budget reserved at load. Silent degradation is not acceptable.

## 5. Per-mask feasibility

**Reusable depth `d_max`**: the largest `d` such that every prefix row's output at
every layer `< d` is independent of suffix tokens. Validity propagates upward
because a GDN layer's input is the previous layer's output, so `d_max` is the index
of the first layer where a prefix row attends to a suffix token. Full-attention
layers are 3, 7, …, 63; the other 48 are causal GDN.

| Family | Mask | `d_max` | T2 | Note |
|---|---|---|---|---|
| JEV | causal | 64 | valid, **default off** | 73-token prompts never reach break-even; block matching valid here |
| AutoTrust v1.1 | Full | **3** | invalid | T0 + T1 only |
| d3 | Full | **3** | invalid | T0 + T1 only |
| d3-turbo text/state media | BlockCascade | **64** | **valid, whole state only** | primary case: one Stage A, 64 rows |
| d3-turbo assets/Box/track | BlockCascade per-row | 64 | valid, unused | already one row per question |
| pplx v1.2 | Full joint | **3** | invalid | T0 + T1 only, whole chunk |

**Full.** At layer 3 a prefix row's attention output is a function of every
question token, so the prefix hidden entering layer 4 is question-dependent, hence
its K/V at layer 7 is, and so on. Reusable depth is 3/64 ≈ 4.7 % and it is the
*first* three layers, so it buys nothing. The near-miss worth spelling out: prefix
**K/V at layer 3 itself** is prefix-only, because K/V are projections of the layer
input. Caching it saves 3/64 and looks attractive.

**BlockCascade, whole state.** The state block's queries are `kNone` over
`[0, split)`, so no state row at any layer sees a question token; the whole state
is reusable, `d_max = 64`. **But within the state block rows are bidirectional**,
so a strict sub-prefix `[0, C)` with `C < split` has K/V at layers ≥ 7 and states
at layers ≥ 4 that depend on `[C, split)` — a cached sub-prefix under BlockCascade
has `d_max = 3`, not 64. This is the single most dangerous misreading of the
design: two requests sharing the first 3K tokens of their state would produce a
hit, and every same-request gate would still pass. T-SUBPREFIX is the dedicated
negative control, and §4.3 removes the storage-level version of the same bug.

**pplx.** The joint protocol is one sequence in which every token sees every
question, so no row is reusable; the whole-chunk T1 entry, keyed on the entire
question set and its order, is the only valid cache. Its `d_max` of 3 follows from
the three leading GDN layers, but nothing consumes it.

**JEV** is the only mode where enabling the cache changes numerics: cache-off is
one `qo == kv == P+q` launch, cache-on is a state pass plus a suffix pass. Hence
default-off and decision-level gating rather than bits.

## 6. Integration

### 6.1 Engine surface (`native/src/engine.rs`)

```rust
pub enum MaskSpec { Causal, Full, BlockCascade { split: usize } }

/// Chosen by family and request shape, INDEPENDENT of the cache (§4.5, §8.6).
pub enum Program { PerRow, StateOnce }

pub struct ForwardJob {
    pub tokens: Vec<u32>,
    pub pos: Vec<u32>,
    pub mask: MaskSpec,
    pub program: Program,
    pub sites: Vec<u32>,
    pub split_r: Vec<usize>,
}

pub struct BatchInfo {
    offsets: Vec<usize>, lens: Vec<usize>, total: usize,
    offsets_dev: CudaSlice<i32>, lens_dev: CudaSlice<i32>,
    pos_start: Vec<usize>,   // NEW: first RoPE position of each row
    split_r: Vec<usize>,     // NEW: per-row state/question boundary (§4.6)
}

pub struct GdnSeed<'a> {                  // all optional; None = today's behaviour
    pub s0: Option<&'a CudaSlice<f32>>,   // ONE LAYER: [batch or 1][nv*dk*dv], broadcast
    pub s_out: Option<&'a mut CudaSlice<f32>>,  // None = keep in shared memory (§4.4)
    pub hist: Option<&'a CudaSlice<bf16>>,      // ONE LAYER: [batch][3][10240]
    pub hist_out: Option<&'a mut CudaSlice<bf16>>,
}

impl Engine {
    pub fn numeric_config(&self) -> NumericConfig;   // frozen after preload_all
    pub fn backbone_hash(&self) -> [u8; 32];
    pub fn prefix_capture(&self, tokens: &[u32], pos: &[u32], split: usize)
        -> Result<PrefixEntry>;                       // Stage A, solo batch
    pub fn set_prefix_cache(&self, cfg: PrefixCacheConfig);
    pub fn decide(&self, job: &ForwardJob) -> Result<Decision>;
}
```

`GdnSeed` is per **layer**, not `[layers][batch][…]`: the caller slices the
checkpoint before each layer. A `[layers][batch][…]` shape built literally would
be 64 × 144 MiB ≈ 9 GiB.

Call-site changes, in risk order:

1. `BatchInfo::positions()` returns `pos_start[s] + t` (`:449`).
2. `full_attention_batch` (`:1402`) and `full_attention` (`:2269`) take a K/V view
   (`Owned` or `Prefix{arena, P}`), a `MaskSpec`, and emit LSE; non-causal dispatch
   goes to FlashInfer at any batch/length (§2.1).
3. `gdn_batch` (`:1588`) **and** the single-sequence `gdn` (`:2361`) thread
   `GdnSeed`; the pre-conv tail is captured at `:1660` and at its `:2445`
   counterpart. Both paths, or a one-item batch behaves differently (§4.5).
4. `decision_logits_batch` (`:1887`) accepts absolute per-row `sites` and a
   `MaskSpec`. It currently reads `lm_head` only (`:1922`), so the d3/pplx readout
   must be wired before any d3/pplx gate can run.
5. `Program::StateOnce` is executed as **its own executor call** (one Stage A plus
   one fork), never co-batched with another request:
   `BatchExecutor::execute(tokens, ids)` (`src/native_backend.rs:47-62`) cannot
   express a 64-row fork, and `DECIDRA_NATIVE_MAX_BATCH` defaults to 8.
6. The `noncausal_full` bool (`:313`) remains for compatibility; the job's
   `MaskSpec` wins and a mismatch is an **error**, not a fallback.

### 6.2 FlashInfer shim (`native/third_party/flashinfer/fi_prefill.cu`)

Add, leaving the two existing entries byte-untouched so JEV's G1 gate is trivial:

```c
int fi_prefill_bf16_ext(q, k, v, out_f32, lse, qo_len, kv_len, mask_mode, stream);
int fi_merge_state(o_a, lse_a, o_b, lse_b, out, rows, heads, head_dim, stream);
```

One extended entry, not two: `qo_len`/`kv_len`/`mask_mode` cover both the
`kv_len == qo_len` and the `kv_len > qo_len` cases, which `kCausal` explicitly
allows (`prefill.cuh:2790` only rejects the opposite). `out_f32` is the f32
partial-output variant the merge test needs (§4.6). Keep `lse` in log2.

FA2 with `qo_len` = 40, `kv_len` = 4096 is a decode-shaped launch through the
prefill kernel: correct, wasteful. Suffix passes must bypass
`DECIDRA_FLASHINFER_MIN_SEQ` (`engine.rs:664`), because a suffix almost always sits
below 1,024 tokens.

### 6.3 GDN, conv, attention

- Shared / split / chunked GDN families and their batched forms: `s0` + optional
  `s_out` (§4.4). A fork with `gdn_shared == false` is an error.
- Conv: `hist` on all four variants, left-zero-padded under 3 tokens.
- Native attention: **assert** the mask is causal (§2.1). No KV extent, no LSE, no
  second mask — the fork uses FlashInfer for both terms (§4.6).

### 6.4 Server (`src/native_backend.rs`, `src/cache.rs`, `src/batch.rs`)

- `DecisionCache` key becomes `program_hash ‖ exec_variant ‖ tokens ‖ head_ids`.
- `SiteHiddenCache` is consulted before enqueueing; a T1 hit enqueues a
  **readout-only** job rather than returning bytes.
- `Program::StateOnce` gets a dedicated executor entry point; prefix affinity
  (grouping by `backbone_hash ‖ prefix`) is opt-in and excluded from bit gates.

### 6.5 Config

| Variable | Default | Meaning |
|---|---|---|
| `DECIDRA_PREFIX_CACHE` | `0` | master switch; refused under Full |
| `DECIDRA_PREFIX_CACHE_BLOCK` | `64` | tokens per block (Causal matching only, P6) |
| `DECIDRA_PREFIX_CACHE_GPU_BYTES` | `0` | VRAM budget for T2; 0 = host only |
| `DECIDRA_PREFIX_CACHE_HOST_BYTES` | `8589934592` | host LRU budget |
| `DECIDRA_PREFIX_CACHE_RETENTION` | `0` | extra checkpoints; **refused unless Causal** |
| `DECIDRA_PREFIX_CACHE_MIN_TOKENS` | `256` | refuse hits below break-even |
| `DECIDRA_PREFIX_MAX_STATE_TOKENS` | `8192` | refuse a fork above this state length (§4.10) |
| `DECIDRA_SITE_HIDDEN_CACHE_BYTES` | `0` | T1 host budget; off by default |
| `DECIDRA_PREFIX_CACHE_VERIFY` | `off` | `recompute` re-derives a hit (test only) |

### 6.6 Metrics

`decidra_prefix_cache_hits_total{tier}`, `…_misses_total{tier}`,
`…_evictions_total{tier,reason}`, `decidra_prefix_cache_bytes{tier}`,
`…_degraded_total`, `decidra_prefix_restore_seconds` (histogram),
`decidra_prefix_stage_seconds{stage}`, `decidra_prefix_saved_tokens_total`,
`decidra_site_hidden_cache_{hits,misses}_total`.

## 7. Performance model and targets

Assumptions: 4,908 tok/s cold causal prefill (measured); 6 GB/s host→device
(re-measure); state restore ≈ 25.7 ms. **Caveat: 4,908 tok/s is JEV's causal path.
d3-turbo is non-causal, which the engine cannot serve correctly until §2.1 is
fixed, and non-causal attention is ~2× the causal attention term — a small part of
the forward, but a target must be measured, not derived.**

| Scenario | Today | With program + cache | Target |
|---|---|---|---|
| d3-turbo state-once, 4K state, 64 questions | 64 prefills ≈ 53 s | 1 state pass + 64-row question pass | ≥ 20×, < 3 s |
| d3-turbo, 8K state, 64 questions | ~2× that | 1 × 1.711 s + question pass | ≥ 20× |
| Same prompt, different `head_ids` (d3/pplx) | 0.833 s at 4K | T1 hit → readout | ≤ 15 ms service time, p95 |
| pplx joint 4K, different `T`/codes | 0.9 s | T1 hit | ≤ 15 ms |
| Cross-request exact state repeat, host tier | 0.833 s | ~70 ms restore | ≤ 150 ms |
| JEV 73 tokens | 39 ms | unchanged (below break-even) | no regression |

Credit where it belongs: the 53 s saving comes from the **`state-once` program**
(one Stage A plus a fork), which needs LSE merge and GDN seeding but no hashing,
eviction or host tier. T2 is justified separately, by cross-request repeats of the
*same* state, which are probably rare. A benchmark must not attribute the
program's win to the cache.

Against vLLM: cold 4K/8K/16K is 1.302 / 2.660 / 5.533 s versus native's
0.833 / 1.711 / 3.740 s (`docs/VLLM-COMPARISON.md:58-63`), and vLLM's baseline
already runs APC with `--mamba-cache-mode align`, so the honest comparison is
*hits vs hits* on the same repeated state and *cold vs cold* otherwise. Report
TMAC/s and restored-bytes-per-avoided-FLOP so the memory/compute trade is visible.

## 8. Testing

### 8.1 Mask algebra against a dense fp32 reference (CPU, CI)

`decidra-cache::mask` — pure Rust, no CUDA: `visible(MaskSpec, split, q_len,
kv_len) -> Vec<bool>` plus `softmax_reference(q, k, v, mask)` that materialises the
full `[q_len, kv_len]` score matrix.

- **T-MASK-CAUSAL**: `visible` is `kv_idx <= q_idx` for all sizes; tilings agree
  with the dense reference within 1e-6 relative.
- **T-MASK-FULL**: all entries visible.
- **T-MASK-BLOCK**: `split` 37, `q_len` 64 → **state-row queries × question keys
  are `false`** and question-row queries are all `true`. Asserting "all-true" would
  also describe the Full mask, i.e. exactly the §2.1 bug.
- **T-MASK-BLOCK-MERGE**: merge(cross, self) equals the single dense product over
  `[0, split+q_len)` within 1e-5 relative, with a **skewed-LSE fixture**.
- **T-MASK-STRADDLE**: a token at `split` belongs to the question block; neg:
  assigning it to the state must fail.
- **T-MASK-OFFSET**: causal with `kv_len > qo_len` matches the dense reference
  where query `i` sees `kv <= prefix + i`. This is our *definition*; a GPU twin
  (T-OFFSET-GPU) is what pins `prefill.cuh:1707` to it.

These test the Rust reference model, not the engine. Only T-MASKDISPATCH (§8.4)
and the external golden (§8.7) constrain the engine's mask.

### 8.2 Pack isolation

- **T-PACK-ALONE**: a 64-row fork (63 identical questions, one adversarial); each
  row matches a solo run within `max|Δp| ≤ 1e-3` and identical argmax. Tolerance,
  not bits: `M` differs.
- **T-PACK-NEGATIVE (mutation)**: widen the cross-term KV extent to include other
  rows' question blocks; T-PACK-ALONE must fail.
- **T-PACK-PIN**: the fork pins once per request; evicting a finished row does not
  free a live entry. Debug builds fill freed memory with NaN, then run a 16K
  prefill under pressure during Stage B.

### 8.3 Storage integrity versus correctness

- **G1 (bit-identical, cache off)**: with `DECIDRA_PREFIX_CACHE=0` every logit is
  bit-identical to the pre-change binary. Run as a **logit dump from both binaries
  on the 28-row suite, diffed to `max Δ == 0`** — not a `run-eval.sh` tolerance
  check, which measures Δ ≤ 0.011 against the CPU golden and would pass small
  drift. Covers the null-`s0`, null-`hist`, null-`lse` and causal-assert paths.
- **G2 (storage integrity)**: `DECIDRA_PREFIX_CACHE_VERIFY=recompute` versus a warm
  hit of the **same program** must be bit-identical. Comparing against a cache-off
  run would conflate storage with the program selector and silently compare two
  different programs. Preconditions: run-to-run determinism for a fixed program and
  `M` (the shipped config is "deterministic across runs",
  `docs/VLLM-COMPARISON.md:90`) and no numeric conversion on the host round-trip.
  G2 does **not** prove the mask is right.
- **G2b**: same after eviction and a host restore.
- **G2-NEGATIVE (must fail)**: flip one byte of a restored state; restore an entry
  with the wrong `P`. Without this, G2 could be a test that compares nothing.
- **G3 (causal JEV only)**: cache-on vs cache-off differ by construction (§5). JEV's
  golden gate is Δ ≤ 0.011 and the shipped measurement is 0.01027, leaving **7.3e-4**
  of margin. A 1e-2 cache delta would consume it, so G3 is **decision-level**: 28/28
  identical argmax, the logit delta *reported*, and any row whose margin is below
  the observed delta keeps the cache default-off.

### 8.4 Key, storage and dispatch correctness (CPU, CI)

- **T-KEY-MASK / POS / SPLIT / MAXLEN / IMAGE / ADAPTER / VARIANT**: one field
  changed ⇒ different key; unchanged ⇒ identical key (also catches a pointer or
  timestamp leaking into the hash).
- **T-KEY-LENPREFIX**: `("ab","c")` and `("a","bc")` must not collide.
- **T-KEY-NUMERIC**: every `NumericConfig` field, including the two env-only
  `LazyLock` knobs, changes the hash; a field added without hashing fails.
- **T-KEY-FROZEN**: `set_*` after `preload_all` returns an error.
- **T-SUBPREFIX (lookup)**: two BlockCascade states sharing their first 3,000
  tokens must miss.
- **T-SUBPREFIX (forced hit, GPU)**: force the hit and show the outputs change —
  the lookup test alone cannot prove the *consequence*.
- **T-STORAGE-ALIAS**: no physical K/V region is shared between two BlockCascade
  entries (§4.3). Cheap to assert with a set of base pointers.
- **T-SPLITVARY**: a tokenization where BPE merges across the boundary for some
  questions but not others; those rows must be routed away from the shared Stage A.
- **T-MASKDISPATCH**: a Full model at `seq < 1024` and at `batch = 2` must equal the
  FlashInfer `kNone` result. Not "or return an error" — a build that rejects every
  short Full request would satisfy that loophole while still being unusable.
  Neg: today's code fails this.

### 8.5 Kernel-level and layer-level bit tests

- **T-GDN-SEED**: seeded `[0,P) + [P,L)` equals `[0,L)` **bit-for-bit** on the
  shared and split kernels, **with the kernel pinned** on both sides (the engine's
  dispatch switches from split to shared below `gdn_split_min_seq`, so an unpinned
  test compares two kernels and its negative control cannot work). On the chunked
  kernels it holds only when `P` is a multiple of the chunk size; elsewhere use a
  stated relative tolerance. Neg: drop the seed and the test must fail.
- **T-CONV-HIST**: the same `qkv` bytes split at `P`, run with `hist`, must be bit
  identical to one launch over `[0,L)`; without `hist` the three conv rows after
  `P` must differ. Run for every conv variant selected by `conv_tiled`.
- **T-ROPE-POS**: a suffix resumed at `P` equals the uncached run; restarting RoPE
  at 0 must not.
- **T-LAYERWISE (capture points)**: with kernels pinned, question-row hidden after
  layers 0–3, fork versus per-row, at ~1e-5 relative. Negatives that must fail:
  capture the **post-conv** rows instead of pre-conv, and capture K **before** RoPE.
  This is the only test that constrains the capture *location* rather than the
  arithmetic — G2 uses the same capture code on both sides and T-CONV-HIST is
  handed a correct `hist`, so neither can see a wrong capture point.
- **T-MERGE (GPU)**: `fi_merge_state(cross, self)` versus one `kNone` launch over
  the concatenated K/V, **f32 partials**, 1e-5, skewed-LSE fixture. Neg: a
  natural-log merge must fail. bf16 partials cannot support this bound and would
  force a loosened tolerance that the negative control cannot separate.
- **T-OFFSET-GPU**: the FA2 causal offset at `kv_len > qo_len` against a dense
  reference.
- **T-READOUT-M**: a site row's bytes at `M = 1` versus `M = total`. Decides the T1
  bit-gate question in §4.8 with data instead of assumption.
- **T-PATH**: the same request alone, co-batched, and through both executor paths;
  record which combinations are bit-equal and make §4.5's invariant explicit.

### 8.6 Program-level gates

- **T-ONCE**: 64 questions over a fixed state; `state-once` versus the **per-row
  BlockCascade program**. That program does not exist yet — today d3-turbo runs
  under the global `noncausal_full`, i.e. Full, where state rows see the question.
  Building it is P1 work, and T-ONCE against the wrong mask would be worse than no
  test. Gate: identical argmax for all 64, `max|Δp| ≤ 1e-3` (never bit-identical —
  the GDN recurrence and Marlin's `M` differ).
- **T-ONCE-EXTERNAL**: `state-once` versus the reference runtime's own `state-once`
  output on the same requests, KL ≤ 3e-3.
- **T-ONCE-ASSETS**: `assets`, Box or track must select the per-row program;
  assert the selector, not just the output.
- **T-ONCE-DEGRADED**: force a cache-allocation failure; the degradation must be
  reported (§4.10), not silent.

### 8.7 Quality gates, including an external reference

- **External golden for the per-row program**: logits from the reference stack on
  64 real requests; KL ≤ 3e-3, top-1 ≥ 99.5 %. This is the only gate that can
  detect a wrong mask contract, and it is why T-ONCE's reference must be built
  rather than assumed.
- Pin the mask contract textually to `d3_runtime.py` and correct the sibling spec,
  which currently says both "all-true except padding" (`:44`) and "state Q sees only
  state K/V". §§4–5 of this document are conditional on the second reading.
- **JEV**: 28/28, mean KL ≤ 1e-3, mean MAE ≤ 1e-3, max Δ ≤ 0.011
  (`docs/VLLM-COMPARISON.md:75-86`), cache off (bit-identical to the shipped
  binary) and on. Latency within 2 % of 37.5 ms / 834 ms / 3,740 ms.
- **Margin histograms** before/after for every family; a flip is a failure even when
  argmax agrees.

### 8.8 Microbenchmarks

1. **Prefix length sweep**: P ∈ {256, 512, 1K, 2K, 4K, 8K, 16K} — cold compute, VRAM
   restore, host restore; ms and break-even.
2. **Restore vs recompute**: measured break-even within 2× of §4.7's 133 tokens,
   else re-derive `MIN_TOKENS`.
3. **Fork scaling**: 1/8/16/32/64 questions at a fixed 4K state, `low_mem` forced;
   Stage A + linear in rows; peak memory recorded, and the 8K state cap confirmed.
4. **Hit/miss at c1 and c16** (`scripts/bench-batch.sh`) with T0 and T1 disabled so
   they do not mask T2's effect.
5. **Memory**: 16K peak with `GPU_BYTES` 0 / 512 MiB / 1 GiB; must reproduce the
   flat 20.11 GiB profile at 0.

### 8.9 Commands and CI placement

The native crate cannot be built without a CUDA toolchain: `native/build.rs`
compiles `kernels.cu`, `marlin.cu` and `fi_prefill.cu` with `nvcc` or falls back to
`docker run` of a CUDA image; cudarc's `dynamic-loading` does not change that. So
the pure logic lives in a **CUDA-free workspace member `crates/decidra-cache`**
(`mask`, `key`, `page_table`, `lru`), added to `[workspace] members` in
`Cargo.toml:11` with `Cargo.lock` updated in the same commit (or `--locked` fails),
and depended on by both `decidra` and `decidra-native`.

CPU (CI; the root crate already builds without a CUDA toolchain because `native`
is an optional feature, `Cargo.toml:14-18`):

```sh
cargo test --locked -p decidra-cache              # T-MASK-*, T-KEY-*, T-SUBPREFIX, T-STORAGE-ALIAS
cargo test --locked -p decidra-cache lru          # eviction/pin ordering in isolation
cargo test --locked                               # root crate, MockBackend, existing API tests
cargo fmt --all -- --check
cargo clippy --all-targets -p decidra -p decidra-cache -- -D warnings
```

`native/tests/*` targets are in the **`decidra-native`** package, so they need
`-p decidra-native --test <name>`; the root `--features native --test <name>`
pattern only reaches the root crate's own `tests/native_smoke.rs`.

GPU (operator-run on GPU1, `scripts/prefix-gates.sh`; never in CI):

```sh
# §2.1 first: mask dispatch must be correct before any cache gate means anything
CUDA_VISIBLE_DEVICES=1 cargo test --release -p decidra-native --test mask_dispatch -- --ignored

# G1: cache off is bit-identical to the shipped binary (dump-and-diff, not a tolerance check)
CUDA_VISIBLE_DEVICES=1 DECIDRA_PREFIX_CACHE=0 decidra-native logits --tokens suite28.json --out /tmp/pc/g1.json
diff-logits --bitwise /tmp/pc/g1-baseline.json /tmp/pc/g1.json     # max Δ must be 0.0

# G2 / G2-NEGATIVE / T-LAYERWISE / T-CONV-HIST / T-ROPE-POS / T-GDN-SEED
CUDA_VISIBLE_DEVICES=1 cargo test --release -p decidra-native --test prefix_seed -- --ignored
# T-ONCE / T-PACK (needs the per-row BlockCascade program)
CUDA_VISIBLE_DEVICES=1 cargo test --release -p decidra-native --test d3_fork -- --ignored
# T-MERGE / T-OFFSET-GPU (f32 partials)
CUDA_VISIBLE_DEVICES=1 cargo test --release -p decidra-native --test fi_merge -- --ignored
# break-even, fork scaling, memory
CUDA_VISIBLE_DEVICES=1 ./scripts/bench-prefix.sh --lengths 256,512,1024,2048,4096,8192,16384
```

New targets: `native/tests/{mask_dispatch,prefix_seed,d3_fork,fi_merge}.rs`
(`native/tests/` currently holds `autotrust_*`, `d3_eval`, `gdn_*`, `marlin_*`,
`pplx_eval`). CI: the existing workflow is Docker-only and path-ignores `docs/**`;
add `.github/workflows/ci.yml` running the CPU block above.

## 9. Phases

| Phase | Work | Gate |
|---|---|---|
| **P-1** | §2.1: non-causal dispatch always through FlashInfer; native kernels assert causal | T-MASKDISPATCH green; Full correct at `seq < 1024` and `batch = 2` |
| **P0** | `crates/decidra-cache`: mask algebra, dense reference, keys, LRU. No engine change | §8.1, §8.4 green in CI |
| **P1a** | **Entry gate**: mask contract pinned to `d3_runtime.py`; sibling spec `:44` corrected | written contract, no code |
| **P1b** | Per-row BlockCascade program (the reference T-ONCE needs); `pos_start`, per-row `split_r`, `hist` on all four conv variants, GDN `s0`/`s_out`, LSE+merge. **No cache yet** | T-LAYERWISE, T-GDN-SEED, T-CONV-HIST, T-ROPE-POS, T-MERGE, T-OFFSET-GPU, external golden |
| **P1c** | `state-once` program on top of P1b (Stage A + fork) | T-ONCE, T-ONCE-EXTERNAL, T-PACK-* |
| **P2** | Stage A capture/store/restore for BlockCascade (whole state, exact key, contiguous) | G2, G2-NEGATIVE, T-SUBPREFIX (both), T-STORAGE-ALIAS, T-SPLITVARY |
| **P3** | T1 site-hidden tier, readout-only job path, server wiring, metrics | T1 hit bit-identical where the readout is gathered; JEV 28/28; T-READOUT-M |
| **P4** | Host LRU, PCIe restore, break-even gate, pinned-region lifetime | G2b, break-even within 2×, `MIN_TOKENS` from measurement |
| **P5** | T3 image features + 3-D mRoPE (blocked on vision) | image/position key tests |
| **P6** | Causal-only block matching, retention checkpoints, promotion, free queue (JEV; owner-approved, low value) | causal 64-token hit correctness; JEV default unchanged |

P-1 first is not busywork: a cache over a wrong-mask engine is a cache of wrong
answers and would make §8 pass. P1b before P1c is deliberate: `state-once` is
meaningless without the per-row reference, and the headline win needs no cache at
all.

## 10. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Full mask silently served causal** (current bug) | **critical** | P-1 first; T-MASKDISPATCH without the "or error" loophole |
| **Sub-prefix hit inside a bidirectional state** | **critical** | whole-state T2 keys; contiguous BlockCascade storage, no dedupe; `RETENTION` refused unless Causal; T-SUBPREFIX lookup **and** forced-hit T-STORAGE-ALIAS |
| Per-row `split_r` divergence | **critical** | fork only exact state+split matches; T-SPLITVARY |
| Conv tail dropped → decayed error through the state | **critical** | pre-conv capture at `:1660`; T-CONV-HIST plus T-LAYERWISE negative control |
| Mask contract itself unresolved (sibling spec contradicts itself) | **critical** | P1a entry gate; external golden |
| Log2 LSE merged with `exp`/`log` | high | `exp2`; f32 partials; T-MERGE with a natural-log negative control |
| Wrong capture point (pre-RoPE K, post-conv tail) | high | T-LAYERWISE with both negative controls |
| Stage A cached from a batched path with a different `M` | high | Stage A solo, `exec_variant` in the key, T-PATH |
| Fork needs per-row state slots (192 MiB) or a kernel change | high | `s0`/`s_out` on the shared/split/chunked family; refuse a fork under `--legacy-gdn`; budget both cases |
| Fork exceeds the envelope (8K+ state, pool retention below `low_mem`) | high | force `low_mem`; `PREFIX_MAX_STATE_TOKENS`; measure peak at 4K/8K |
| Cache memory via `Buf` pollutes the pool | high | raw `CudaSlice`, free to driver; 16K memory bench |
| Capture-before-RoPE / `pos_start` omitted | high | T-ROPE-POS and T-LAYERWISE, both directions |
| Host entry freed while a restore is queued | high | `CudaEvent` per restore |
| Cache-allocation failure silently degrades the program | high | degraded counter/field, or a reserved fork budget |
| Self-referential gates all pass on a wrong mask | high | external per-row golden (§8.7) |
| `NumericConfig` drifts; a `set_*` after freeze changes numerics | medium | frozen struct, erroring setters, completeness test |
| JEV cache-on changes numerics | medium | default off; G3 decision-level + margin arithmetic |
| Cross-request T2 hit rate near zero | medium | measure before P6; the fork is where the value is |
| Tenant leak if persistence is added | medium | process-lifetime only; salt required if revisited |

## 11. Open questions

1. Does d3-turbo's hosted API ever send `assets` together with a text state on one
   request? If so the selector must fail closed to per-row.
2. Is the whole-state T2 hit rate on real traffic non-zero at all? If not, T2 is
   dead weight and only T1 ships.
3. What maximum state length does an 8K-capped fork actually hold at peak? Depends
   on `low_mem` and on whether the `s0`/`s_out` change lands; measure before fixing
   the default.
4. Which d3 answer shape (package vs vLLM) the external golden targets — inherited
   from the sibling spec.

## 12. Verification

Two reviews with `claude-opus-5-5` (the id pinned in `~/.claude/settings.json`),
each on the spec as it then stood. Briefs `/tmp/prefix-review-{1,2}.md`, transcripts
`/tmp/prefix-opus-{1,2}.md`.

### Round 1 — 2026-10-20. Verdict: *approve with the listed fixes; #1–#9 must be
resolved before P2.* Fourteen findings, each verified against the code.

| # | Finding | Fix |
|---|---|---|
| 1 | **Critical**: 64-token matching is invalid inside a bidirectional state; a shared first-3K-tokens hit would be silently wrong and no same-request gate would see it | T2 matches whole states exactly; block matching confined to Causal; §5 adds the sub-prefix `d_max = 3` theorem; T-SUBPREFIX |
| 2 | **Critical, live bug**: `noncausal_full` is honoured only on the FlashInfer branch; the fallback kernels are causal-only, so Full models are wrong below 1,024 tokens and at `batch > 1` | new §2.1, phase P-1, T-MASKDISPATCH |
| 3 | **Critical**: G2's "cold" reference was undefined, so it tested neither storage nor correctness | program selector split from cache switch; G2 relabelled storage integrity; correctness rests on kernel bit tests + external golden; the 53 s credited to `state-once` |
| 4 | **High**: `split_r` can differ per row through BPE merges at the boundary | per-row `(state, split_r)`; fork only exact matches; T-SPLITVARY |
| 5 | **High**: every d3-turbo gate compared the engine with itself | external per-row golden in §8.7; T-MASK-BLOCK asserts state×question = `false` |
| 6 | **High**: T1 bit-identity conflicted with G1 because the readout GEMM uses `M = total` | T1 bit gate conditioned on the gathered readout; JEV T1 off by default; T-READOUT-M |
| 7 | **High**: cached bytes depend on batch-mates and the executor path | Stage A solo fixed path; `exec_variant` recorded; T-PATH; affinity excluded from bit gates |
| 8 | **High**: "kernels zero or ignore incoming state" was wrong — the recurrent family is in-place | corrected §2/§4.4; seeding is a copy for recurrent, `s0` only for chunked |
| 9 | **High**: K capture was cited before RoPE | capture after `:1494`, before `:1561`; V from `vv` |
| 10 | Medium: the conv-tail error is decayed through the state, not confined to 3 tokens | mechanism corrected; T-CONV-HIST is a kernel-level bit test |
| 11 | Medium: FA2 LSE is log2-domain | `exp2` merge; T-MERGE with a natural-log negative control |
| 12 | Medium: `program_hash` over-keyed and incomplete | `backbone_hash`/`program_hash` split; `NumericConfig` with a completeness test; env-only knobs listed |
| 13 | Medium: memory budget omitted the resident checkpoint and Stage B activations, overstated question K/V, cited a fixed 16K regression | §4.10 re-derived with every component and three caveats |
| 14 | Medium: the CI plan did not build and named a non-existent test | `crates/decidra-cache`; real targets and commands |

Also applied in round 1: T1 sized by bytes; the incoherent "site inside a prefix"
rule replaced by `L = full length`; the bisection claim corrected; the `kCausal`
assertion direction fixed (it rejects `kv_len < qo_len`); pplx `d_max` unified at
3; copy-cost arithmetic corrected; break-even 133 tokens; "64 prefills", not 65;
length-prefixed hashing; refcount pinned once per request.

### Round 2 — 2026-10-20. Verdict: *approve with the listed fixes; P-1 and P0 may
start immediately; P1/P2 must not start until findings 1–5 are fixed and §4.10 is
re-derived.* Twenty-one findings, all verified.

| # | Finding | Fix |
|---|---|---|
| 1 | **Critical**: "`s_out` null, so no per-row state buffer" is false — the batched shared kernel reads and writes a per-row slot (`kernels.cu:1925-1935`), so 64 rows need 192 MiB, and a literal `[layers][batch]` `s0` would be 9 GiB | §4.4 now states that seeding needs no kernel change but skipping the write-back does; `s0`/`s_out` added to the shared/split/chunked families; `s0` is one layer slice; `gdn_recurrent` excluded from forks; both budgets given |
| 2 | **High**: the 419 MiB fork budget was too low; the 20.11 GiB bound was measured with `low_mem` on; a 16K state cannot fit | §4.10 re-derived (≈ 470/660 MiB) and three caveats added: force `low_mem`, cap the state length, report degradation |
| 3 | **High**: page tables plus hash chains could recreate the sub-prefix bug in storage via dedupe; the fork does not need them | §4.3: contiguous BlockCascade entries, no dedupe; hashing and paging scoped to Causal (P6); T-STORAGE-ALIAS |
| 4 | **High**: the native kernels need more than a mask argument — separate `qo`/`kv` lengths, padding-key masking, and a different LSE base each | the fork runs **both** terms through FlashInfer; native kernels assert causal only; every emitter pinned to log2 in `sm_scale·log2e` units |
| 5 | **High**: the mask-contract contradiction is still open and did not block P1 | P1a entry gate: pin the contract to `d3_runtime.py` and correct the sibling spec |
| 6 | Medium: `backbone_hash` contained the per-request `split` | mask *family* only; `split` in the T1/T2 keys |
| 7 | Medium: T0/T1 were not keyed on program or execution path | `exec_variant` added to T0/T1/T2 keys |
| 8 | Medium: no program selector was defined; the batcher cannot express a 64-row fork | `Program::{PerRow, StateOnce}` in `ForwardJob`; `state-once` is its own executor call |
| 9 | Medium: the per-row BlockCascade reference does not exist, so T-ONCE would compare against the wrong mask | P1b builds it; T-ONCE-EXTERNAL checks against the reference runtime |
| 10 | Medium: T1's config name/default contradicted the text | one name, `…_BYTES`, default 0 |
| 11 | Medium: `hist` was added to one conv kernel of four | all four, or reject; left-zero padding stated |
| 12 | Medium: "frozen" `NumericConfig` was unenforced; quant knobs bind at load, others per forward | setters error after the freeze; `config.json`/`decision_config.json` added to the manifest hash |
| 13 | Medium: the CI commands would not run as written (wrong package for `native/tests`, clippy scope, workspace member, lockfile, G1 not a bit check) | §8.9 rewritten with `-p decidra-native`, `-p decidra-cache`, the workspace/lockfile note, and G1 as a dump-and-diff |
| 14 | Low: T-MASKDISPATCH's "or return an error" was a loophole | dropped; non-causal dispatch goes to FlashInfer at any size and batch |
| 15 | Medium: T-MERGE cannot pass at 1e-5 with bf16 partials, and the native emitters use another base | f32 partial-output shim variant; skewed-LSE fixture; base pinned everywhere |
| 16 | **High**: a wrong capture point is visible to no gate | T-LAYERWISE with post-conv and pre-RoPE negative controls |
| 17 | Medium: T-SUBPREFIX tested only the lookup, and in the CPU section | split into a CPU lookup test and a GPU forced-hit test; T-STORAGE-ALIAS |
| 18 | External golden is the only mask check | acknowledged; §8.7 makes it the reference for T-ONCE |
| 19 | Low: G3's 1e-2 exceeded JEV's remaining margin (7.3e-4); T-GDN-SEED compared two kernels; T-MASK-OFFSET cannot pin the kernel from the CPU | G3 made decision-level with the margin arithmetic; T-GDN-SEED pins the kernel; T-OFFSET-GPU added |
| 20 | Low: entry counts, status line, `config.rs` path, two identical shim entries, merge point unspecified | 20/7 entries; status updated; `native/src/config.rs`; one extended shim entry; merge before `sigmoid_gate`/`o_proj` |

Both rounds' transcripts are retained; nothing in either review was left
unaddressed.

## 13. Sources

- Engine (read for this spec): `native/src/engine.rs` (`BatchInfo`, `embed`,
  `forward_dev`, `batch_forward`, `run_layer_batch`, `full_attention_batch`
  `:1402`, `gdn_batch` `:1588`, `gdn` `:2361`, `decision_logits_batch` `:1887`,
  `decision_logits` `:3097`, `gemm` dispatch `:2977-3005`), `native/src/flashinfer.rs`,
  `native/src/config.rs`, `native/src/kernels.rs`, `native/kernels/kernels.cu`
  (`gdn_chunked_f32`, `gdn_chunked_tc_body`, `gdn_recurrent*` incl. batched,
  `attention_tc*`, `attention_warp`, all four `conv1d_silu*`), `native/build.rs`,
  `native/third_party/flashinfer/fi_prefill.cu`,
  `native/third_party/flashinfer/include/flashinfer/attention/prefill.cuh`
  (`:1707` causal offset, `:2790` the `kv_len < qo_len` assertion, `:2722-2730`
  log2 LSE), `src/cache.rs`, `src/native_backend.rs`, `src/batch.rs`,
  `src/config.rs`, `src/api.rs`, `native/src/weights.rs`.
- Measurements: `docs/VLLM-COMPARISON.md`, `docs/RESULTS.md`, `docs/JOURNAL.md`.
- Contracts: `docs/specs/2026-10-20-decision-model-support.md`.
- Prior reviews: `/tmp/consult-opus.md` §6, `/tmp/consult-opus2.md` (C1–C7),
  `/tmp/consult-harness.md` §2.2–2.3 and §4.
- vLLM: [APC design](https://docs.vllm.ai/en/latest/design/prefix_caching/),
  [APC feature docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/),
  [`vllm.config.cache`](https://docs.vllm.ai/en/latest/api/vllm/config/cache/).

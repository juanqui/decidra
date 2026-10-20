# jev-native

Native CUDA inference engine for the JEV-27B checkpoint. The runtime is Rust +
custom CUDA kernels (+ cuBLAS for BF16 GEMM). It does not import or link an
external inference runtime, PyTorch, or Python.

## Layout

```text
native/
  build.rs            compiles kernels/*.cu to a sm_89 cubin
  kernels/kernels.cu  RMSNorm, RoPE, causal GQA attention (warp + legacy), GDN
  kernels/marlin.cu   vendored Marlin W4A16 launcher + repack (third_party/marlin)
  src/cuda.rs         driver context, cubin module, cuBLAS handle
  src/tensor.rs       mmap'd multi-shard safetensors reader
  src/config.rs       Qwen3.5 text config
  src/weights.rs      weight sources (merged BF16 today; packed W4A16 next)
  src/kernels.rs      typed kernel launch wrappers
  src/engine.rs       decoder forward, layer/stage dumps, golden compare hooks
  src/main.rs         CLI
```

## Build

The host does not need a CUDA toolkit. `build.rs` uses `nvcc` when present and
otherwise runs the CUDA compiler from a container:

```sh
cd native
cargo build --release          # default: docker run nvidia/cuda:13.1.1-devel-ubuntu24.04 nvcc -cubin
JEV_NVCC=/usr/local/cuda/bin/nvcc cargo build --release   # if nvcc is installed
JEV_CUDA_IMAGE=nvidia/cuda:13.1.1-devel-ubuntu24.04 cargo build --release
```

Runtime requirements (no toolkit needed):

- NVIDIA driver exposing `libcuda.so.1` (CUDA >= 13.0 bindings).
- `libcublas.so.13` and `libcublasLt.so.13` (CUDA 13 runtime), found via
  `LD_LIBRARY_PATH`/`ldconfig`. The stable host copy is
  `/srv/models/jev-awq/cudalibs` (568 MB, not committed):

  ```sh
  export LD_LIBRARY_PATH=/srv/models/jev-awq/cudalibs
  ```

  `JEV_CUDA_LIBS` names this directory for scripts; the binary itself only needs
  `LD_LIBRARY_PATH` because cuBLAS is loaded with `dlopen`.
- RTX 4090 / sm_89.

## Run

GPU 1 only (GPU 0 is left alone):

```sh
export CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/tmp/cudalibs

./target/release/jev-native self-test

./target/release/jev-native layer \
  --model /tmp/jev-native-ref/tiny \
  --tokens '[313,288,307,312,259,311,471,16]' \
  --layer 3 --stage final_norm --out /tmp/final.f32

./target/release/jev-native compare \
  --ref /tmp/jev-native-ref/tiny/ref/final_hidden.f32 --got /tmp/final.f32
```

`--tokens` accepts inline JSON or a path to a JSON file. `--stage` is one of
`embed`, `attn`, `mlp`, `layer`, `final_norm`, `logits`. `--last-token` trims the
output to the final position (for lm_head logits). `--input <f32>` starts from an
external `[seq, hidden]` state to isolate a single layer. `--dump <dir>` writes
every intermediate as raw f32 for golden comparison. `--preload` keeps all
decoder layers resident; `--no-cache` streams one layer at a time (needed for the
51 GiB merged-BF16 checkpoint on a 24 GiB card).

## Profiling

`JEV_PROFILE=1` records per-stage CUDA events. The debug variables
`JEV_MARLIN_DEBUG` and `JEV_GEMM_DEBUG` now only log (they no longer call
`cudaStreamSynchronize`); per-GEMM timing needs the explicit
`JEV_GEMM_DEBUG_SYNC=1` opt-in. Do not leave debug variables exported while
benchmarking: a single synchronize per GEMM serializes the stream and inflates
the enclosing event windows. Use `env -u JEV_MARLIN_DEBUG -u JEV_GEMM_DEBUG
-u JEV_PROFILE` for clean timings.

## Library API

```rust
use jev_native::Engine;

let engine = Engine::new(model_dir)?;   // validates config, opens shards, CUDA ctx
engine.preload_all()?;                  // 12.4 GiB packed weights resident, ~5.8 s
let hidden = engine.forward_last_hidden(&tokens)?;            // [hidden]
let logits = engine.decision_logits(&tokens, &[3721, 1802])?; // [2], last position
```

`Engine` is `Send` (compile-time asserted) but not intended for concurrent
forwards: move it to one blocking worker (`spawn_blocking`) and serialize calls.
`decision_logits` stores the per-call head ids, so do not call it concurrently.
CLI commands remain available for validation (`self-test`, `layer`, `compare`,
`info`, `dequant`, `matmul`).

## Numerics conventions

- Qwen3.5 `Qwen3_5RMSNorm` is zero-centered: stored weight `w` is applied as
  `(1 + w)`. The tiny reference stores exactly 0.0 for `input_layernorm`,
  `post_attention_layernorm`, `q_norm`, `k_norm`, and `model.norm`; `(1+w)` is
  therefore required. The GDN `RMSNormGated` and every linear use plain weights.
- Conv, beta, and the attention gate reproduce torch's bf16 rounding points;
  the delta-rule state, GDN decay, and softmax accumulate in f32.
- Norm weights and linears must come from the same checkpoint: AWQ checkpoints
  store smoothing-adjusted norms, so they are never mixed with merged-bf16.

## Attention

Three causal GQA kernels share the same graph:

- **tiled FA2-style** (default): 64-query CTAs, 32-key K/V tiles, cp.async
  double buffering, `ldmatrix` + `mma.sync` m16n8k16 BF16, online softmax in
  registers. Fixed entry points for head_dim 64/128/256; dynamic shared
  27.6/52.2/99 KiB.
- **warp-per-query** (`--legacy-attention`): one warp per query, no block
  synchronization in the key loop. Kept as the fallback.
- **block-per-query**: the original kernel, still compiled for A/B but not
  exposed on the CLI.

For **single-request, seq >= 1024** full-attention layers the engine loads
vendored FlashInfer 0.7.0.post1 headers (`third_party/flashinfer`) through a
small `fi_prefill.cu` wrapper (`libfi_prefill.so`, built by `build.rs` and
dlopen'd at runtime). 24Q/4KV head_dim 256 BF16 prefill uses FA2 split-K and
runs 1.80x faster than the native tiled d256 kernel: 16K attention
567 -> 315 ms over the 16 layers, full 16K forward 6.25 -> 6.00 s. 8K saves
2.2%, 4K 1.3%, 1K 0.5%. Batched requests, other shapes, and load failures use
the native tiled/warp kernels. Disable with `JEV_FLASHINFER_ATTENTION=0`,
change the threshold with `JEV_FLASHINFER_MIN_SEQ` (default 1024). See
`native/third_party/flashinfer/INTEGRATION.md`.

Long-context latency (Marlin weights, resident, SM89 RTX 4090, mixed W4A8-INT8):

| tokens | engine CLI (warm) | resident server | peak GPU mem |
|---|---|---|---|
| 4K (4,090) | 0.916 s | 0.913 s | 18.83 GiB |
| 8K (8,178) | — | 1.871 s | 20.17 GiB |
| 16K (16,383) | 4.005 s | 4.051 s | 20.17 GiB |
| 32K (32,763 tok) | not rerun | not rerun | — |

The pre-chunked W4A16 stack measured 2.12-2.18 s (4K), 4.23-4.32 s (8K) and
8.62-8.85 s (16K); the current stack is ~2.2-2.3x faster. The remaining time is
dominated by the Marlin GEMMs; attention is <1% at 1K and still small at 16K.
Long prefill holds 100% GPU utilization at ~420-430 W; peak memory is flat
across repeated 16K requests (low-memory pool from 12,288 tokens). The embedding
table is resident on the GPU (2.37 GiB, row gather on-device).

Long-context quality gate against the CPU AWQ/SDPA probes
(`/tmp/jev-native-ref/long-context/probes`, Marlin + tiled attention):

| probe | final hidden max / mean / rel_rms / cosine | head logits diff (3721, 1802) | two-way prob delta | top-1 |
|---|---|---|---|---|
| 4K (4090 tok) | 0.0625 / 4.60e-3 / 0.808% / 0.999967 | [0.0156, 0.0313] | 3.5e-4 | true |
| 8K (8178 tok) | 0.0996 / 4.28e-3 / 0.795% / 0.999968 | [0.0156, 0.0313] | 3.0e-4 | true |
| 16K (16383 tok) | 0.0469 / 4.01e-3 / 0.692% / 0.999976 | [0.0078, 0.0313] | 6.7e-4 | true |
| 32K (32763 tok) | 0.0469 / 4.18e-3 / 0.727% / 0.999974 | [0.0078, 0.0313] | 6.0e-4 | true |

The probe `probs` include the decision-head transform (bias/temperature); the
raw two-way softmax of the head logits is within 5.6e-4 of the golden raw
values. Deltas are BF16 accumulation noise and the expected choice is `true` at
every length. The warp kernel has slightly smaller final-hidden deltas
(0.72/0.76/0.69% rel_rms) but is slower; both pass. Causally, the first 36
positions of the 76/1K/4K/8K/16K prompts also match the 36-token golden head
logits within max 0.051-0.086 / mean 0.014-0.018.

K/V are transient per layer (bf16, no persistent KV cache, no FP8 KV); the GDN
state is f32. 16K peaks at 18.9 GiB including the 13.1 GiB resident weights.
32K (32,763-token CPU-AWQ probe, expected `true`, golden P(true)=0.98315) is the
verified maximum: 18.6-19.0 s full forward, 19.1 GiB peak, 100% utilization,
~455 W, ~4.4 GiB headroom on 24 GiB. For `seq >= 24576` the buffer pool releases
dropped large tensors instead of retaining them and large MLP/attention/GDN
intermediates are dropped as soon as they are consumed; 1K/16K timings are
unchanged (486 ms / 8.86 s).

## Weight sources

`weight_packed` is detected per module, so the graph is identical for both
sources:

- **merged BF16** (`/srv/models/jev-awq/merged-bf16`): graph/kernel validation.
- **AWQ W4A16** (`/srv/models/jev-awq/awq-w4a16-v2`): compressed-tensors
  pack-quantized, group 64, asymmetric. Layout: `weight_packed` I32
  `[out, ceil(in/8)]` with nibble `k` at bits `4k` covering input `8j+k`;
  `weight_zero_point` I32 `[ceil(out/8), in/64]` packed along the output dim;
  `weight_scale` BF16 `[out, in/64]`; `weight_shape` I64 `[out, in]`;
  `w = (q - zp) * scale`. The native dequantizer reproduces
  `compressed_tensors.PackedQuantizationCompressor.decompress()` bit-exactly
  (max_abs 0.0 on four sampled tensors).
- No `weight_g_idx` in this checkpoint, so the weights are in column order.
- **Marlin W4A16** (`--marlin` / `Engine::set_marlin(true)` before
  `preload_all`): the vendored vLLM/Marlin kernel (Apache-2.0, see
  `third_party/marlin/NOTICE.md`). Weights are repacked from the CT layout to
  Marlin's `[K/16, 2N]` layout on the GPU; the repacked qweight, scales, and
  zero points match vLLM's `gptq_marlin_repack` / `marlin_permute_scales` /
  `marlin_zero_points` **bit-exactly** (verified for a 17408x5120 module).
  The kernel uses `kU4` with asymmetric zero points, `kBFloat16` activations,
  group 64 (group_blocks=4), thread_m_blocks 1..4, 128 CTAs.

Packed decoder weights are uploaded once by `--preload` / `preload_all()` and
stay resident (12.22 GiB packed + 0.18 GiB dequant workspace). Correctness mode
dequantizes a module into the workspace and runs cuBLAS BF16 GEMM; a fused
W4A16 kernel is the next performance step. Embedding rows are gathered from the
mmap (a few KiB per request); lm_head rows are gathered/streamed so the 2.5 GiB
head is never required in full.

## Status

Validated against `/tmp/jev-native-ref/tiny` (4 layers, GDN + full attention):

| tensor | max_abs | mean_abs | rel_rms |
|---|---|---|---|
| embedding | 0 | 0 | 0 |
| layer 0 (GDN) | 9.8e-4 | 7.4e-5 | 0.42% |
| layer 1 (GDN) | 9.8e-4 | 7.6e-5 | 0.37% |
| layer 2 (GDN) | 9.8e-4 | 4.9e-5 | 0.26% |
| layer 3 (full attn) | 1.5e-3 | 1.1e-4 | 0.33% |
| final norm | 4.7e-2 | 1.0e-2 | 1.3% |
| last-token logits | 1.2e-2 | 3.0e-3 | 1.2% |

Per-layer isolated errors are at the bf16 rounding floor (<=1 bf16 ulp for most
elements).

AWQ 27B against `/tmp/jev-native-ref/golden-27b` (36-token prompt, CPU golden
from `compressed_tensors` dequant):

| tensor | max_abs | mean_abs | rel_rms | cosine |
|---|---|---|---|---|
| layer 0 | 0.125 | 1.5e-4 | 0.35% | 0.9999942 |
| layer 1 (isolated) | 0.125 | 1.4e-4 | 0.18% | 0.9999983 |
| layer 3 (full stack) | 0.25 | 6.2e-4 | 0.38% | 0.9999928 |
| layer 4 (isolated) | 0.125 | 1.8e-4 | 0.11% | 0.9999994 |
| layer 63 (full stack) | 32 | 6.4e-2 | 0.96% | 0.9999539 |
| final norm | 0.375 | 9.6e-3 | 1.25% | 0.9999221 |
| head logits (3721/1802) | 0.0977 | 0.0177 | 1.18% | 0.9999306 |
| full logits (last) | 0.0625 | 3.0e-3 | 0.26% | 0.9999969 |

Top-1 token/last logit are identical (`867`, `3.578125`); two-way softmax is
0.8568 vs the golden 0.8559 (abs diff 9.6e-4); full-vocab log-sum-exp differs by
9.4e-4. `max_abs` at large activations is exactly one BF16 ulp.

Resident (`--preload`, GPU 1 freed, 12.66 GiB used / 10.86 GiB free):

| prompt | reference (dequant+cuBLAS) | hybrid fused | Marlin |
|---|---|---|---|
| 36 tokens | 174.7 ms | 84.8 ms | **45.1 ms** (3.87x) |
| 76 tokens | 202.2 ms | 126.4 ms | **73.0 ms** (2.77x) |
| 1024 tokens | 1261.2 ms | 1261.2 ms | **486.1 ms** (2.59x) |

Marlin preload is 10.1 s (repack) and uses 13.1 GiB; the other paths preload in
5.8 s / 12.7 GiB. Init 0.2 s. GPU samples: reference/fused ~184-231 W,
Marlin ~45 ms at 36 tokens with 100% utilization. Marlin golden checks: layer 0
mean 1.58e-4 / rel_rms 0.347%; final norm rel_rms 1.23%, cosine 0.999924.

At 1K the gain is 2.59x (1261 -> 486 ms) after the GDN recurrence work below.
Marlin alone was 1.32x: that case is compute-bound and cuBLAS+dequant was
already efficient. Two 128-thread large-M tile variants (`thread_n_blocks=8/4`,
`thread_k_blocks=4/8`) were compiled and A/B'd at 1K: 982.3 ms and 997.1 ms
versus 984.3 ms for the default, with bit-identical outputs. There is no win,
so the default 256-thread config stays.

### GDN recurrence

`gdn_recurrent_split` keeps the f32 recurrent state in dynamic shared memory and
assigns four lanes to each value column; each lane owns `dk/4` state rows and
the two dot products are reduced with two `__shfl_xor` steps. Grid is
`heads * (dv/32)`, block is 128 threads, shared is `dk*(vc+1)*4` (16.9 KiB for
dk=128). Per-layer cost at 1K drops from 13.35 ms to 3.41 ms (3.9x) and the
end-to-end 1K forward from 958 ms to 486 ms. The decay + `kv` accumulation are
fused into one pass.

The split reduction is only used for **seq >= `gdn_split_min_seq` (default
512)**. Short prompts use the accurate single-column shared recurrence: the
four-way f32 summation reorders the state dot products enough to shift a rare
short-prompt logit. Reproduced on suite row `r22` (54 tokens):
split raw logits [2.7344, 1.7813] vs CPU golden [2.6719, 1.7891] (two-way delta
1.43e-2, fails the suite gate); shared raw logits [2.6875, 1.7734] (delta
6.43e-3, passes; the pre-split Marlin suite max was 6.03e-3). Configure with
`--gdn-split-min-seq N`; `--gdn-shared` forces the accurate path and
`--legacy-gdn` the original global-state kernel.

### Chunked prefill on short prompts

The chunked TF32 kernel is also used for **seq >= `gdn_chunked_min_seq`
(default 64)** when chunked prefill is enabled; the server enables it by
default (`JEV_NATIVE_GDN_CHUNKED=true`), and shorter prompts keep the shared
fallback. Set `--gdn-chunked-min-seq N` on the CLI or
`JEV_NATIVE_GDN_CHUNKED_MIN_SEQ=N` for the server (0 forces every length).

Measured at 54 tokens on GPU1 (suite `r22`): shared [2.6875, 1.7734],
chunked TF32 [2.7031, 1.7891], CPU golden [2.6719, 1.7891]. Suite28 with the
chunked path on every row: top-1 28/28, max delta 6.37e-3, MAE 6.48e-4
(shared: 1.03e-2 / 9.97e-4), so `r22` stays inside the 0.011 gate. The
36-token golden head logits are max 0.109, top-1 36/36; the 4K/8K/16K probes
are bit-identical to the previous chunked kernel and keep their documented
deltas.

The chunked kernel uses 128-bit vector loads/stores, a 4-column register-tiled
triangular solve without per-row barriers, and a 4x4 register-tiled state
update (64 registers, no spills). It requires `dk=dv=128`; other head dims
fall back to the shared/split recurrence.

Full forward, warm, RTX 4090 / GPU1:

| tokens | shared GDN | chunked TF32 | GDN stage (shared -> chunked) |
|---|---|---|---|
| 76 | 69.5 ms | **52.0 ms** (1.34x) | 27.6 -> 7.3 ms |
| 128 | 97.8 ms | **64.9 ms** (1.51x) | 46.4 -> 9.8 ms |
| 1K | 478 ms | **355 ms** (1.35x) | 75.3 -> 30.6 ms |
| 16K | 8.26 s | **7.53 s** (1.10x) | 1200 -> 485 ms (2.47x) |

The 1K and 16K rows are same-session A/Bs against the pre-optimization chunked
kernel (old: 400 ms / 8.26 s); absolute latency tracks GPU clocks.

1K chunked breakdown (452 ms sum of regions, 355 ms warm forward): MLP GEMMs
211 ms (47%), `gemm.b` 84 ms (host launch gap), GDN recurrence 32 ms (7%),
`in_proj_qkv` 35 ms, `out_proj` 19 ms, `in_proj_z` 19 ms, attention 4.2 ms,
MLP activation 6.8 ms.

The batched path launches one chunked CTA grid `(dv/VC, heads, batch)` over the
flattened offsets/lens (`gdn_chunked_tc_batched`) whenever every sequence is
>= the threshold; mixed batches keep the per-sequence loop. Outputs and state
are bit-identical to one launch per sequence (covered by the `gdn_chunked`
test). Batch of 8 x 73 tokens: GDN recurrence 54.6 -> 19.6 ms and warm forward
247 -> 212 ms (37.7 req/s, 2.5k tok/s). The remaining 212 ms is MLP GEMMs
119 ms (at ~165 TFLOP/s, i.e. tensor-core peak), non-MLP projections ~40 ms,
attention 2.4 ms; `gemm.b` in the profile is host-side launch gap, not GPU
work.

Shared-path 1K breakdown (486 ms forward, historical): MLP GEMMs 204 ms (42%),
GDN recurrence 164 ms (34%), other GEMMs 88 ms (18%), GDN conv 8.8 ms, MLP
activation 6.7 ms, attention 4.0 ms, norms/residuals ~5 ms. (The profiler's
`gemm.b` entry is an event-gap artifact; standalone it is 0.017 ms.)

Fallbacks: `--gdn-shared` (single-column shared state, 856 ms at 1K) and
`--legacy-gdn` (original global-state kernel). Resident outputs are
bit-identical to the streaming (`--no-cache`) run. Golden after residency
(hybrid): final norm rel_rms 1.07% (reference 1.25%), full logits rel_rms 0.24%,
top-1 identical, two-way softmax identical at f32 print precision.

### Fused W4A16 status

`gemm_w4a16_bf16` dequantizes packed weights directly into padded shared memory
and runs bf16 tensor-core WMMA (BM=BN=64, BK=64) with double-buffered staging and
register prefetch. The padded shared stride (72 halfs) removes the 128-byte
stride bank conflicts that made the first prototype slower than cuBLAS. One warp
covers four output rows: threads 0..7 load the 8 packed words (32 B, coalesced)
of a row, and each thread writes one aligned `uint4` (8 bf16) into shared. The
next group's packed word, scale, zero point, and activation tile are prefetched
into registers while the current group multiplies. 72 registers, 45 KiB shared
(2 CTAs/SM), no spills.

Policy: use the fused kernel when `N >= 5120` and `M <= 256`, otherwise fall
back to dequant+cuBLAS (`--fused-min-n`, `--fused-max-m`). Small-N projections
(`k/v/a/b`) and large prefills are faster on the reference path; the 1K-token
case is exactly the reference latency.

Kernel-only averages (36 rows, ms, reference -> fused):
gate_proj 0.630 -> 0.170; down_proj 0.625 -> 0.231; q_proj 0.447 -> 0.119;
in_proj_z 0.193 -> 0.083; out_proj 0.187 -> 0.097; in_proj_qkv 0.357 -> 0.113;
k_proj 0.039 -> 0.082; in_proj_a/b 0.014 -> 0.076.

GEMM outputs are recycled through a small per-size buffer pool (`Buf`), which
removes per-call `cuMemAlloc` jitter (a fresh allocation of some sizes costs
~0.6 ms on this driver). At most two buffers per size are retained.

Correctness: layer 0 vs AWQ golden max 0.125 mean 9.96e-5 rel_rms 0.306%, and
full logits max 0.0625 mean 2.57e-3 rel_rms 0.238%, top-1 identical.

A 128x128 tile variant (`--fused-tile128`, 16 warps, 90 KiB dynamic shared, one
M tile for M <= 128) was implemented and correctness-checked (65-token layer 0
mean 1.2e-4 vs the reference path), but it is **slower**: 180 ms at 76 tokens and
1395 ms at 1K, versus 126 ms and 1270 ms. With 90 KiB shared and 98 registers
only one CTA fits per SM, and `grid.x = N/128` leaves SMs idle (e.g. 40 blocks
for `down_proj`). Forcing the 64x64 kernel at 1K is 1336 ms vs 1270 ms for the
reference path, so the `M > 128` fallback stays.

At 76 rows the winning 64x64 kernel still needs two M tiles (128 padded rows),
which is why the speedup there is 1.6x rather than 2x. Fusing the remaining
RMSNorm/GDN/attention work and a pipelined 128-row tile with 2-CTA occupancy are
the next steps toward the ~60 ms ideal.

### W4A8-INT8 status (shipped, mixed scale granularity)

The model runs Marlin W4A8-INT8 (`kS8 x kU4`) on every linear except the residual
attention writers `self_attn.o_proj/v_proj/k_proj` outside layers 56-63. Two activation
scale granularities are compiled:

- **per-token** (`quant_int8_per_token_bf16`): one scale per row, int32 accumulation.
  Fastest. Used for `mlp.gate_proj`.
- **per-K-group-64** (`quant_int8_group_bf16` + the `group_a_scale` Marlin variant): one
  scale per row per 64 K elements, f32 accumulation per group. Used for GDN `z/b/a`,
  `mlp.up_proj`, `mlp.down_proj`, `in_proj_qkv`, `out_proj`, `q_proj`, and the all-linear
  layers 56-63. It costs ~35-50% versus the per-token epilogue but still runs 2.1-2.4x
  faster than W4A16 on the same module.

Server suite (batch=1, CPU AWQ golden): **28/28, KL 3.095e-5, MAE 8.642e-4, max
7.511e-3** — every gate passes, deterministic across three runs. Engine forward:
4K 916 ms, 16K 4.01 s (from 1231 ms / 5.17 s on the W4A16-only selective stack).
Server: c1/c8/c16 **24.7 / 49.5 / 53.4 req/s**; long 4K/8K/16K **0.913 / 1.871 /
4.051 s**, all at 100% GPU for long prompts.

The environment knobs are `JEV_NATIVE_INT8` (default true),
`JEV_NATIVE_INT8_EXCLUDE` (modules kept on W4A16),
`JEV_NATIVE_INT8_ALL_LAYERS` (layer subset that runs all-linear INT8),
`JEV_NATIVE_INT8_GROUP_SCALE` (default true),
`JEV_NATIVE_INT8_GROUP_EXCLUDE` (modules kept on per-token scales) and
`JEV_NATIVE_INT8_CLIP` (abs-max clip factor, default 1.0). The CLI equivalents
are `--marlin-int8`, `--int8-exclude`, `--int8-all-layers`, `--int8-group-scale`,
`--int8-group-exclude`, `--int8-clip`, and `--gdn-chunked-min-seq` for the GDN
threshold. `--marlin-int8` without exclusions runs all-linear INT8 (group-64 by
default in the CLI only if `--int8-group-scale` is passed).

Marlin launches the whole M in one kernel by default (`JEV_MARLIN_CHUNK=0`,
the kernel's internal `parallel` loop); `JEV_MARLIN_CHUNK=N` caps each launch
at roughly N rows. Full-M removes the outer 1024-row split (16 launches at 16K
become 2: 16320 + tail) and measures 16K 5991 -> 5935 ms, 4K 1432 -> 1425 ms;
the 4K/8K/16K probe deltas stay inside the W4A16 envelope and suite28 is
bit-identical (short rows are a single launch either way).

The environment knobs are `JEV_NATIVE_INT8` (server default true),
`JEV_NATIVE_INT8_EXCLUDE` (comma-separated module substrings),
`JEV_NATIVE_INT8_ALL_LAYERS` (layer subset that runs all-linear INT8) and
`JEV_NATIVE_INT8_CLIP` (abs-max clip factor, default 1.0). The CLI equivalents
are `--marlin-int8`, `--int8-exclude`, `--int8-all-layers`, `--int8-clip`, and
`--gdn-chunked-min-seq` for the GDN threshold. `--marlin-int8` without
exclusions runs all-linear INT8.

Layer-wise expansion (all-linear INT8 for one layer subset, current selective
set elsewhere): `first:N` and `alt:N` fail suite28 for N=8..32; the largest
passing subset is **`last:8`** (layers 56-63): 28/28, KL 2.37e-5, MAE 6.91e-4,
max 7.48e-3. On the 500-row holdout it agrees with the selective config on
481/500 top-1; every flip is a 0.5/0.5 or <0.008-margin tie. This subset is
the **server default** (`JEV_NATIVE_INT8_ALL_LAYERS=last:8`); it improves the
suite margin and costs ~4 ms at 16K. Set it empty to disable.

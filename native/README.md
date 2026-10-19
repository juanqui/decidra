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

Long-context latency (Marlin weights, resident, SM89 RTX 4090):

| tokens | tiled FA2 | warp (`--legacy-attention`) | peak GPU mem |
|---|---|---|---|
| 1K | 0.959 s | 0.959 s | 13.1 GiB resident |
| 4K | 3.99-4.04 s | 4.65 s | ~17.2 GiB |
| 8K | 7.95-8.05 s | 9.88 s | ~17.6 GiB |
| 16K | 16.10-16.32 s | 22.88 s | 18.9 GiB |

The standalone attention kernel is 7-12x faster than the warp kernel, but the
full forward is now **GEMM-bound**: ~0.96 ms/token (1K 0.96 s, 4K 4.0 s, 8K
8.0 s, 16K 16.3 s). Attention is no longer the quadratic term; reaching
<2 s at 4K requires a faster large-M W4A16 path, not a faster attention kernel.

Long-context quality gate against the CPU AWQ/SDPA probes
(`/tmp/jev-native-ref/long-context/probes`, Marlin + tiled attention):

| probe | final hidden max / mean / rel_rms / cosine | head logits (3721, 1802) | two-way prob delta | top-1 |
|---|---|---|---|---|
| 4K (4090 tok) | 0.0938 / 4.41e-3 / 0.783% / 0.999970 | [1.2344, 5.0] vs [1.2734, 5.0313] | 1.7e-4 | true |
| 8K (8178 tok) | 0.0781 / 4.26e-3 / 0.805% / 0.999967 | [1.2109, 5.1563] vs [1.2188, 5.1563] | 1.5e-4 | true |
| 16K (16383 tok) | 0.0703 / 4.14e-3 / 0.731% / 0.999974 | [1.2422, 5.2188] vs [1.2422, 5.25] | 5.6e-4 | true |

The probe `probs` include the decision-head transform (bias/temperature); the
raw two-way softmax of the head logits is within 5.6e-4 of the golden raw
values. Deltas are BF16 accumulation noise and the expected choice is `true` at
every length. The warp kernel has slightly smaller final-hidden deltas
(0.72/0.76/0.69% rel_rms) but is slower; both pass. Causally, the first 36
positions of the 76/1K/4K/8K/16K prompts also match the 36-token golden head
logits within max 0.051-0.086 / mean 0.014-0.018.

K/V are transient per layer (bf16, no persistent KV cache, no FP8 KV); the GDN
state is f32. 16K peaks at 18.9 GiB including the 13.1 GiB resident weights, so
24 GiB fits with headroom.

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
| 1024 tokens | 1261.2 ms | 1261.2 ms | **958.5 ms** (1.32x) |

Marlin preload is 10.1 s (repack) and uses 13.1 GiB; the other paths preload in
5.8 s / 12.7 GiB. Init 0.2 s. GPU samples: reference/fused ~184-231 W,
Marlin ~45 ms at 36 tokens with 100% utilization. Marlin golden checks: layer 0
mean 1.58e-4 / rel_rms 0.347%; final norm rel_rms 1.23%, cosine 0.999924.

At 1K the gain is 1.32x: that case is compute-bound and cuBLAS+dequant was
already efficient; the fixed 64-row M tile and re-reading W per M tile limit
Marlin's large-M benefit. Two 128-thread large-M tile variants
(`thread_n_blocks=8/4`, `thread_k_blocks=4/8`) were compiled and A/B'd at 1K:
982.3 ms and 997.1 ms versus 984.3 ms for the default, with bit-identical
outputs. There is no win, so the default 256-thread config stays and the
variants were removed. Resident outputs are
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

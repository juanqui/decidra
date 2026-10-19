# Native kernel stack

The shipped runtime is Rust + custom CUDA kernels, with cuBLAS/cuBLASLt used for BF16
matrix multiplies. There is no external model server, PyTorch, Triton, FlashInfer, or
Python in the process.

Target: RTX 4090 / sm_89. `native/build.rs` compiles `native/kernels/kernels.cu` to a
cubin; `cudarc` loads the CUDA driver, the cubin module, and cuBLAS dynamically. The host
therefore needs no CUDA toolkit at runtime, only the driver and cuBLAS runtime libraries.

## Kernels

| Kernel | Used for |
|---|---|
| RMSNorm | zero-centered Qwen3.5 norms (`(1 + w)`); plain norm for GDN gated norm |
| RoPE | partial rotary factors on full-attention Q/K |
| Tiled FA2-style causal GQA | full-attention layers (default; 64-query CTAs, 32-key tiles, cp.async, BF16 WMMA) |
| Warp-per-query causal GQA | full-attention fallback (`--legacy-attention`) |
| Block-per-query causal GQA | original kernel, compiled for A/B only |
| GDN conv / prepare / rule | gated delta-network linear-attention layers |
| GDN recurrent (split-dk) | four lanes per value column, f32 state in shared memory; 3.9x faster than the shared-state kernel |
| SiLU × gate | MLP activation |
| Add | residual connections |
| Dequant W4A16 | expand packed `int4` to a BF16 workspace for cuBLAS |
| Self-test | launch validation |

Numeric conventions (see [`native/README.md`](../native/README.md)):

- Conv, beta, and the attention gate reproduce torch BF16 rounding points.
- Delta-rule state, GDN decay, and softmax accumulate in f32.
- Norm weights and linears must come from the same checkpoint: AWQ norms are
  smoothing-adjusted.

## Quantized matmuls

The shipped checkpoint is compressed-tensors pack-quantized, asymmetric int4 (AWQ/GPTQ),
group size 64:

- `weight_packed`: I32 `[out, ceil(in/8)]`; nibble `k` at bits `4k` covers input `8j+k`.
- `weight_zero_point`: I32 `[ceil(out/8), in/64]`, packed along the output dim.
- `weight_scale`: BF16 `[out, in/64]`.
- `w = (q - zp) * scale`.

The dequantizer reproduces `compressed_tensors` decompression bit-exactly on sampled
tensors. There is no `weight_g_idx`, so weights are in column order.

### Marlin W4A16 (shipped default)

`JEV_NATIVE_MARLIN=true` repacks the compressed-tensors layout on the GPU into Marlin's
`[K/16, 2N]` layout during `preload_all` and runs the vendored Marlin W4A16 kernel
(Apache-2.0, `native/third_party/marlin`). Repacked qweight, scales, and zero points match
the reference GPTQ-Marlin repack / scale / zero-point permutations bit-exactly for a
17408×5120 module verified by the native worker.

| prompt | dequant + cuBLAS | hybrid fused | Marlin |
|---|---|---|---|
| 36 tokens | 174.7 ms | 84.8 ms | **45.1 ms** (3.87x) |
| 76 tokens | 202.2 ms | 126.4 ms | **73.0 ms** (2.77x) |
| 1024 tokens | 1261.2 ms | 1261.2 ms | **488 ms** (2.58x) |

Marlin resident weights use **13.1 GiB**; preload/repack takes **10.1 s**. Tiled FA2
attention plus gated split-dk GDN makes long context **GEMM-bound** at ~0.49-0.58 ms/token:

| tokens | full forward (tiled FA2 + split GDN) | previous warp | peak GPU mem |
|---|---|---|---|
| 1K | **0.488 s** | 0.959 s | 18.15 GiB |
| 4K | **2.12-2.18 s** | 4.65 s | ~17.2 GiB |
| 8K | **4.23-4.32 s** | 9.88 s | ~17.6 GiB |
| 16K | **8.62-8.97 s** | 22.88 s | 18.86 GiB |
| 32K (32,763 tok) | **18.60-19.03 s** | - | 19.09 GiB |

The attention kernel itself is 7-12x faster than warp on identical buffers (kernel-only
1K 0.276 ms, 4K 2.85 ms, 8K 9.68 ms). The split-dk GDN recurrence is 3.9x faster on its
hot kernel (1K per-layer 13.35 -> 3.41 ms; 1K forward 958 -> 488 ms). Remaining time is
dominated by W4A16 GEMMs; attention is <1% at 1K and still small at 16K/32K.

Split GDN is used only for **seq >= `gdn_split_min_seq` (default 512)**. Short prompts
use the accurate single-column shared recurrence: the four-way f32 reduction reorders
the state dot products enough to shift a rare short-prompt logit. Suite row `r22`
(54 tokens) measured split two-way delta **1.43e-2** (fails) versus shared **6.43e-3**
(passes). Long sequences keep the 3.9x split path; 16K remains 8.62-8.97 s and passes.

For **seq >= 24,576** the engine enables a low-memory buffer-pool mode: dropped large
tensors are released instead of retained, and large MLP/attention/GDN intermediates are
freed as soon as they are consumed. This is what makes **32K fit and run on 24 GB**
(19.09 GiB peak, ~4.4 GiB headroom); 1K/16K timings are unchanged.

Long-context correctness against CPU AWQ/SDPA probes (Marlin + tiled FA2 + split GDN):

| probe | final rel_rms | cosine | two-way prob delta | top-1 |
|---|---|---|---|---|
| 4K | 0.808% | 0.999967 | 3.5e-4 | true |
| 8K | 0.795% | 0.999968 | 3.0e-4 | true |
| 16K | 0.692% | 0.999976 | 6.7e-4 | true |
| 32K | 0.727% | 0.999974 | 6.0e-4 | true |

All long-context suite cases are correct, and **32K is the verified maximum context** on
24 GB.

### Fallback paths (`JEV_NATIVE_MARLIN=false`)

Dequantize one module into a reusable BF16 workspace (largest module 17408×5120) and run
cuBLAS BF16 GEMM. With `JEV_NATIVE_FUSED_W4A16=true`, the hybrid fused kernel dequantizes
into padded shared memory and runs BF16 WMMA (BM=BN=64, BK=64) with double-buffered
staging for `N >= 5120 && M <= 256`; otherwise it falls back to dequant + cuBLAS. The
fused path ignores the Marlin-specific repacked layout.

## Residency

`preload_all()` repacks/upload every packed decoder layer once and keeps it resident.
With Marlin the repacked layout uses **13.1 GiB** and preload takes **10.1 s**; the
dequant fallback uses ~12.7 GiB / 5.8 s. Embedding rows are gathered from the
mmap (a few KiB per request); lm_head rows are gathered/streamed so the 2.5 GiB head is
never required in full.

The alternative `--no-cache` CLI mode streams one layer at a time (needed for the 51 GiB
merged-BF16 validation checkpoint on a 24 GiB card). It is not the serving path.

## Decision read-out

`Engine::decision_logits(tokens, head_ids)` runs the full decoder, final norm, and the
lm_head rows for the requested verbalizer ids at the last position. It returns raw `f32`
logits aligned with `head_ids`. The server applies bias, per-kind temperature, optional
fitted calibration, and softmax; a constant logit shift (the softmax normalizer) cancels.

The engine is `Send` but owns one CUDA context/stream and is not safe for concurrent
forwards. The server wraps it in `Arc<Mutex<Engine>>` inside `spawn_blocking`, so the async
reactor stays free while forwards are serialized.

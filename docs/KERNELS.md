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
| Causal GQA attention | full-attention layers |
| GDN conv / prepare / rule | gated delta-network linear-attention layers |
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
| 1024 tokens | 1261.2 ms | 1261.2 ms | **1144.9 ms** (1.10x) |

Marlin resident weights use **13.1 GiB**; preload/repack takes **10.1 s**. Long context is
attention/prefill-bound: 4K **4.65 s**, 8K **9.88 s**, 16K **22.88 s** (18.9 GiB peak).
The 1K gain is limited by the fixed 64-row M tile; a persistent M-loop is the next step.

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

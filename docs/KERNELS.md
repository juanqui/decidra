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

Default correctness path: dequantize one module into a reusable BF16 workspace
(largest module 17408×5120) and run cuBLAS BF16 GEMM.

The fused W4A16 kernel (`JEV_NATIVE_FUSED_W4A16=true`, now the default) dequantizes into
padded shared memory and runs BF16 WMMA (BM=BN=64, BK=64) with double-buffered staging.
The hybrid policy uses it when `N >= 5120 && M <= 256` and falls back to dequant + cuBLAS
otherwise. On a 36-token prompt this is 84.8 ms vs 174.7 ms (2.06x); at 1K tokens it falls
back and provides no benefit. Small-N projections and large prefills stay on cuBLAS.

## Residency

`preload_all()` uploads every packed decoder layer once (~12.22 GiB packed plus a
~0.18 GiB dequant workspace) and keeps it resident. Embedding rows are gathered from the
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

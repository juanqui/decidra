# Serving kernels on SM89 (RTX 4090)

This is the kernel stack vLLM 0.31.0 selects for `JEV-27B-VL` AWQ W4A16 on the Ada
RTX 4090 (SM89). The choices below were read from the live startup log and verified by
the tuning matrix; they are the best available correct options on this GPU, not merely
defaults.

Host GPU selection is **GPU 1 only** (`--gpus device=1`, or `CUDA_VISIBLE_DEVICES=1` for a
native launch). Inside the container CUDA exposes one device, so vLLM log lines that say
`device 0` refer to that container-visible device, not to host GPU 0.

## Model shape

- Architecture: `Qwen3_5ForConditionalGeneration` (`model_type=qwen3_5`), a hybrid model:
  full-attention layers plus GDN linear-attention layers, with a separate vision tower.
- Compute dtype: BF16 activations; quantized weights.
- Quantization: compressed-tensors `pack-quantized`, **W4A16**, int4, group size 64,
  asymmetric (`symmetric: false`), static activation ordering. `lm_head` and the visual
  tower are not quantized.
- Decode is one token per decision read-out; latency is prefill-dominated.

## Selected kernels

| Stage | Kernel | Startup evidence |
|---|---|---|
| Quantized linear (mixed-precision and `CompressedTensorsWNA16`) | **MarlinLinearKernel** | `Using MarlinLinearKernel for mixed-precision linear`; `Using MarlinLinearKernel for CompressedTensorsWNA16` |
| GDN prefill | **Triton/FLA** (`head_k_dim=128`) | `Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)` |
| GDN decode | **CUDA** | `GDN decode kernel: cuda` |
| Full-attention decode/prefill | **FlashInfer** | `Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN']`; `FlashInfer resolved ... decode_backend=flashinfer-native, kv_cache_dtype=torch.float8_e4m3fn, arch=sm89` |
| Vision encoder attention | **FlashAttention** | `Using backend AttentionBackendEnum.FLASH_ATTN for vit attention`; `Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention` |
| Sampler | **FlashInfer** | `Using FlashInfer for top-p & top-k sampling` |
| Model runner | **V2** | `Using V2 Model Runner` |
| KV cache layout | **LBNHC**, block **1,568 tokens** | `Using LBNHC KV cache layout`; `Setting attention block size to 1568 tokens to ensure that attention page size is >= mamba page size` |

Related cache setup:

- FP8 KV cache (`--kv-cache-dtype fp8`, `torch.float8_e4m3fn`).
- Mamba page size padded by 0.13% so attention and mamba pages are exactly equal.
- Prefix caching enabled; chunked prefill enabled (default in 0.31.0).
- `--logprobs-mode processed_logprobs` returns the processed distribution the decision
  read-out expects.
- `--trust-request-chat-template` is required for the image path's per-request raw template.
- `--max-num-seqs 8` is the verified correctness cap for System 1.

## Why these are the best available on SM89

### Marlin W4A16

Marlin is vLLM's high-performance mixed-precision GEMM for weight-only quantized linear
layers. It supports **int4, group size 64, asymmetric** weights with BF16 activations,
which is exactly this artifact, and it requires only SM75+. It is the correct `auto` pick
for `CompressedTensorsWNA16` on SM89.

Alternatives in the vLLM 0.31 registry:

| Alternative | Why it is rejected here |
|---|---|
| **Machete** | Requires compute capability **9.0 (Hopper/SM90)**. RTX 4090 is SM89. |
| **Cutlass** (`CutlassW4A8LinearKernel`) | SM90 and **W4A8** only. This model is W4A16. |
| **Exllama** | Supports **FP16 activations only**; this model runs BF16. Using it would require `--dtype float16`, changing numerics and accuracy. Its handling of group-64 asymmetric int4 would also need verification. |
| **Triton W4A16** | Generic fallback; not faster than Marlin for this shape. |
| **Torch/emulation** | Unquantized or dequantize-per-forward; far slower and not the served format. |

So `--linear-backend auto` selecting Marlin is not a fallback: it is the only correct,
high-throughput W4A16 kernel available on SM89.

Marlin startup warning:

```text
WARNING [marlin_utils.py:239] Marlin requires thread-tile padding for some weight shapes
in this model. Activations and/or outputs of the padded layers are padded/sliced on every
forward; performance may be degraded.
```

Some layer dimensions are not multiples of Marlin's thread tile. vLLM pads the activation
and output buffers and slices them on every forward. This is a **performance** warning,
not a correctness warning. Its cost is included in every number reported in
[`RESULTS.md`](RESULTS.md): the measured run still reaches **3,077-3,188 prompt tok/s** on
long prefill and **34.2 req/s** on short prompts at concurrency 8 on this GPU. There is no
supported alternative on SM89 that avoids the padding while remaining correct and faster;
switching backends to "remove the warning" would either fail (Machete/Cutlass), change
dtype/accuracy (Exllama), or slow down (generic Triton).

### GDN prefill: Triton/FLA

The model's linear-attention layers use gated delta networks with `head_k_dim=128`. vLLM's
FLA/Triton chunked prefill kernel parallelizes the delta-rule recurrence across the chunk
and is the fastest available prefill path for this head size on Ada. The log explicitly
reports `requested=auto`, so this is the automatic choice, not a forced override.

### GDN decode: CUDA

For the one-token decode step, vLLM uses the CUDA GDN decode kernel (`GDN decode kernel:
cuda`). The recurrent state update is memory/launch-bound at batch 1; the CUDA kernel
avoids the Triton launch overhead of the prefill path. There is no faster SM89 kernel for
this operation in the 0.31 stack.

### Decode/full attention: FlashInfer

vLLM chooses FlashInfer from `['FLASHINFER', 'TRITON_ATTN']`. With paged KV in LBNHC
layout and FP8 E4M3 cache, FlashInfer's native decode backend is the fastest available
paged-attention path on SM89; `TRITON_ATTN` is the fallback. The resolved line confirms
`arch=sm89`, BF16 query dtypes, and FP8 KV.

### Vision attention: FlashAttention

The ViT and multimodal encoder use `FLASH_ATTN`. For the 512x512 single-image path, this
is the standard high-throughput encoder attention and avoids materializing the full
attention matrix. The image path was measured at 0.176 s and 312 prompt tokens for the
functional check.

### Sampler: FlashInfer

`Using FlashInfer for top-p & top-k sampling` covers the one-token sample. The decision
read-out itself uses the returned logprobs for the requested head token ids; the sampled
token is not what determines the decision distribution. FlashInfer is nevertheless the
best sampler path available and keeps the OpenAI passthrough efficient.

### V2 runner, LBNHC, block 1,568

- The V2 model runner is the current vLLM execution path for this model and supports the
  hybrid attention/mamba cache bookkeeping.
- **LBNHC** is the KV layout required by FlashInfer for this configuration.
- The attention block size is forced to **1,568 tokens** so the attention page size is at
  least the mamba page size; mamba pages are padded by 0.13% to match exactly. This is a
  cache-allocation constraint of the hybrid model, not a tunable performance knob in this
  deployment.
- The 1,568-token block interacts with `max_model_len=16384`; long-context requests are
  served within that cap.

## Numerics and batch invariance

Kernel choice is about speed, but it also fixes the low-order bits:

- Same process, same prompt, direct vLLM logprobs vs Rust `/v1/decide`: **28/28**, max
  delta **4.44e-16**, mean KL **3.53e-18**. The Rust head math is exact.
- The hybrid GDN model is **not batch-invariant** at the `5e-4` level, and up to about
  `5e-3` when prefill chunking and concurrent requests change batch composition. Top-1
  decisions stay stable.
- The saved production-v2 baseline was produced by an older serving build. Comparing
  current Rust against it yields mean KL `1.505e-5`, mean MAE `6.83e-4`, max delta
  `4.933e-3` (`r22`), deterministically. This is vLLM build/config drift (FP8 KV,
  chunked prefill, SM89 kernels), not a Rust arithmetic difference. Use the documented
  `0.01` cross-build tolerance; keep strict gates for same-process comparisons.

Do not interpret the Marlin padding warning, FP8 KV, or the FlashInfer/Triton selection as
accuracy defects. They are the correct Ada serving configuration; the measured decision
top-1 and exactness results are in [`RESULTS.md`](RESULTS.md).

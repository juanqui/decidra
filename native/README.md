# jev-native

Native CUDA inference engine for the JEV-27B checkpoint. The runtime is Rust +
custom CUDA kernels (+ cuBLAS for BF16 GEMM). It does not import or link vLLM,
PyTorch, or Python.

## Layout

```text
native/
  build.rs            compiles kernels/*.cu to a sm_89 cubin
  kernels/kernels.cu  RMSNorm, RoPE, causal GQA attention, GDN conv/prepare/rule
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
cargo build --release          # default: docker run vllm/vllm-openai:v0.31.0 nvcc -cubin
JEV_NVCC=/usr/local/cuda/bin/nvcc cargo build --release   # if nvcc is installed
JEV_CUDA_IMAGE=nvidia/cuda:13.1.1-devel-ubuntu24.04 cargo build --release
```

Runtime requirements (no toolkit needed):

- NVIDIA driver exposing `libcuda.so.1` (CUDA >= 13.0 bindings).
- `libcublas.so.13` and `libcublasLt.so.13` (CUDA 13 runtime), found via
  `LD_LIBRARY_PATH`/`ldconfig`. For development these were copied out of the
  CUDA image to `/tmp/cudalibs`.
- RTX 4090 / sm_89.

## Run

GPU 1 only (GPU 0 and the vLLM oracle on :8002 are left alone):

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

| prompt | reference path (dequant+cuBLAS) | fused `--fused-w4a16` |
|---|---|---|
| 36 tokens | 176.6 ms (first 267 ms) | 405 ms |
| 76 tokens | 202.9 ms | 460 ms |
| 1024 tokens | 1277.6 ms | 2689 ms |

Preload 5.8 s, init 0.2 s. GPU sample during the 36-token reference run: 100%
utilization, ~231 W peak; the fused kernel is latency-bound at ~156 W. Resident
outputs are bit-identical to the streaming (`--no-cache`) run, and the golden
checks are unchanged (final norm rel_rms 1.25%, full logits rel_rms 0.26%,
top-1 identical, two-way softmax delta 9.6e-4).

### Fused W4A16 status

`gemm_w4a16_bf16` dequantizes packed weights directly into shared memory and
runs bf16 tensor-core WMMA (BM=BN=64, BK=64). It is numerically correct (layer 0
mean_abs 1.0e-4 vs golden; full logits rel_rms 0.22%, top-1 identical) but **not
faster** than dequantize-to-BF16 + cuBLAS on this workload: kernel-only averages
are gate_proj 0.67 vs 0.63 ms, in_proj_qkv 1.20 vs 0.80 ms, out_proj 0.55 vs
0.19 ms, down_proj 1.54 vs 0.63 ms, in_proj_a/b 0.40 vs 0.014 ms. The reference
path is bandwidth-efficient and cuBLAS is already good at these small M; the
naive fused schedule serializes dequant with MMA and re-dequantizes W per M
tile. It stays selectable (`--fused-w4a16`, `--fused-min-n`) but is off by
default. A Marlin-style pipeline (double-buffered cp.async, W tile reused across
M tiles) is required to beat the reference path.

# Decidra d3-turbo deployment

Decidra is one process: the HTTP server and the native CUDA engine. The model
package is mounted read-only; weights are never baked into the image.

## Prerequisites

- NVIDIA RTX 4090 / L40S (SM89), H100 / H200 (SM90), or RTX PRO 6000
  Blackwell (SM120), CUDA 13.x driver.
- 24 GB VRAM for the W4A8 g64 v4 package on a 4090. The current host caps
  both GPUs at **450 W**; the SM89 numbers in the benchmark report reflect
  that. A 16-entry / 5 GiB pinned GPU state cache, an 8 GiB host L2, and a
  2 GiB scratch reserve keep 24K states from exhausting the card.
- On H100 80 GB / H200 use the published BF16 package (49 GB) for best
  performance: it leaves ~31 GiB free on H100 after preload for the
  exact-state caches. A re-quantized W4A8 H200 artifact must pass the 32-row
  accuracy gate before use; the SM80-style Marlin path was slower than BF16 on
  H200.
- On RTX PRO 6000 Blackwell (96 GB) use the BF16 package. It passes the
  state-once parity gate (max Δp 0.0102) and is the recommended Blackwell path.
  W4A8 fits and is slightly faster, but its single-vs-batch parity is 0.081 on
  this stack; do not use it where per-row equivalence is required.
- `libcublas.so.13` / `libcublasLt.so.13` at runtime.
- The d3-turbo package: `config.json`, `decision_config.json` with
  `attention_mode: prefix_bidir`, `tokenizer.json`, `model.safetensors`,
  `readout.safetensors`. Anything else is rejected at startup.

## Build

```bash
# SM89 (4090/L40S)
cargo build --release --locked --features native

# SM90 (H100/H200)
DECIDRA_CUDA_ARCH=sm_90 cargo build --release --locked --features native

# SM120 (RTX PRO 6000 Blackwell)
DECIDRA_CUDA_ARCH=sm_120 cargo build --release --locked --features native

# SM90 + FA3 (optional; FA2 is the measured default for d3)
DECIDRA_FA3=1 \
DECIDRA_CUTLASS_INCLUDE="/path/cutlass/include:/path/cutlass/tools/util/include" \
DECIDRA_CUDA_ARCH=sm_90 cargo build --release --locked --features native
```

The Docker image is SM89 by default. Build an H100 image with
`--build-arg CUDA_ARCH=sm_90`, or an RTX PRO 6000 image with
`--build-arg CUDA_ARCH=sm_120`. A binary embeds one cubin and rejects a GPU from
another SM family at startup. FA3 requires CUTLASS headers and is built from
source outside the default image.

## Run

```bash
# RTX 4090 / L40S (SM89 W4A8)
CUDA_VISIBLE_DEVICES=0 \
DECIDRA_CHECKPOINT=/srv/models/d3-turbo-w4a8-g64-v4 \
DECIDRA_NATIVE_MODEL=/srv/models/d3-turbo-w4a8-g64-v4 \
DECIDRA_NATIVE_GPU_EMBED=false \
DECIDRA_STATE_CACHE_ENTRIES=16 \
DECIDRA_STATE_CACHE_BYTES=5368709120 \
DECIDRA_HOST_STATE_CACHE_BYTES=8589934592 \
DECIDRA_BATCH_ROWS=32 \
DECIDRA_BIND_ADDR=0.0.0.0:8899 \
./target/release/decidra

# RTX PRO 6000 Blackwell (SM120, BF16)
CUDA_VISIBLE_DEVICES=0 \
DECIDRA_CHECKPOINT=/srv/models/d3-turbo \
DECIDRA_NATIVE_MODEL=/srv/models/d3-turbo \
DECIDRA_NATIVE_MARLIN=0 \
DECIDRA_NATIVE_GPU_EMBED=true \
DECIDRA_BATCH_ROWS=32 \
DECIDRA_BIND_ADDR=0.0.0.0:8899 \
./target/release/decidra

# H100 80 GB / H200 (SM90 BF16)
CUDA_VISIBLE_DEVICES=0 \
DECIDRA_CHECKPOINT=/srv/models/d3-turbo \
DECIDRA_NATIVE_MODEL=/srv/models/d3-turbo \
DECIDRA_NATIVE_MARLIN=0 \
DECIDRA_NATIVE_GPU_EMBED=true \
DECIDRA_STATE_CACHE_ENTRIES=16 \
DECIDRA_STATE_ONCE_MAX_ROWS=128 \
DECIDRA_BIND_ADDR=0.0.0.0:8899 \
./target/release/decidra
```

`--check-config` prints the package contract without touching the GPU.
`--native-check` preloads the engine and runs the state-once self-test
(state capture + one question + per-row block comparison).

## Docker

```bash
docker build -t decidra:0.2.0 .
# H100/H200:
docker build --build-arg CUDA_ARCH=sm_90 -t decidra:0.2.0-h100 .

docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
  -p 8899:8080 \
  -v /srv/models/d3-turbo-w4a8-g64-v4:/models:ro \
  decidra:0.2.0
```

The image embeds the cubins and FlashInfer library, so it needs no build tree
or CUDA toolkit at runtime. The host driver provides `libcuda`; the image
copies the real cuBLAS/cuBLASLt files.

## Environment

| variable | default | meaning |
|---|---|---|
| `DECIDRA_BIND_ADDR` | `0.0.0.0:8080` | HTTP bind |
| `DECIDRA_MODEL` | `vllm-sr/d3-turbo` | client-facing model id |
| `DECIDRA_CHECKPOINT` | `/srv/models/d3-turbo-w4a8-g64-v4` | package |
| `DECIDRA_NATIVE_MODEL` | `$DECIDRA_CHECKPOINT` | engine package |
| `DECIDRA_TOKENIZER` | `<checkpoint>/tokenizer.json` | tokenizer |
| `DECIDRA_NATIVE_CACHE` | `true` | resident packed weights |
| `DECIDRA_NATIVE_MARLIN` | `true` | Marlin W4A8 |
| `DECIDRA_NATIVE_INT8` | `true` | INT8 per-token activations |
| `DECIDRA_NATIVE_INT8_GROUP_SCALE` | `true` | per-64-group scales |
| `DECIDRA_NATIVE_INT8_EXCLUDE` | `self_attn.v_proj,self_attn.k_proj` | W4A16 modules |
| `DECIDRA_NATIVE_INT8_GROUP_EXCLUDE` | empty | modules using per-token scales |
| `DECIDRA_NATIVE_GDN_CHUNKED` | `true` | chunked GDN prefill |
| `DECIDRA_NATIVE_GDN_CHUNKED_MIN_SEQ` | `16` | chunked threshold |
| `DECIDRA_NATIVE_GPU_EMBED` | `false` | GPU embedding table (2.4 GiB) |
| `DECIDRA_MAX_QUEUE` | `32` | queued requests before 503 |
| `DECIDRA_BATCH_REQUESTS` | `32` | max requests per pass |
| `DECIDRA_BATCH_ROWS` | `32` | soft question-row budget per pass |
| `DECIDRA_BATCH_WINDOW_US` | `2000` | coalescing window |
| `DECIDRA_STATE_CACHE_ENTRIES` | `16` | pinned exact-state entries |
| `DECIDRA_STATE_CACHE_BYTES` | `5368709120` | state K/V/GDN byte budget |
| `DECIDRA_STATE_CACHE_PIN` | `1` | keep the first N states resident |
| `DECIDRA_HOST_STATE_CACHE_BYTES` | `8589934592` | host L2 for overflow states |
| `DECIDRA_STATE_ONCE_MAX_ROWS` | `64` | coalesced question rows per pass |
| `DECIDRA_FLASHINFER_FA3` | off | opt-in Hopper FA3 (FA2 is faster for d3) |
| `DECIDRA_METRICS_ENABLED` | `true` | Prometheus `/metrics` |

## Health

`GET /health` returns `{"status":"ready","model":"vllm-sr/d3-turbo","head":"d3-code-readout"}`.
The runtime image also carries `LICENSE`, `LICENSE-MIT`, `LICENSE-APACHE`, and
`NOTICE` under `/usr/share/licenses/decidra/`.
The server refuses to start unless the package is `prefix_bidir`, the F32
readout exists, and the state-once self-test agrees with the per-row block
path. Images and videos are rejected by the text path with HTTP 400.

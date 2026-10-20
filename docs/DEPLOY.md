# Deployment

## Topology

```text
GPU 1 ── native CUDA engine (in-process, packed AWQ W4A16 resident) ──┐
                                                                      │
CPU   ── decidra HTTP server ── 0.0.0.0:8501 ── clients / proxy ───┘
```

One process, one GPU. The server process owns the engine: the HTTP layer is
async, inference runs through `decidra_native::Engine` on GPU 1 and keeps packed
weights resident. There is no external model server, Python, or PyTorch in the
process.

## Ports and devices

| Resource | Default | Knobs |
|---|---|---|
| HTTP listen | `0.0.0.0:8080` (script: `0.0.0.0:8501`) | `DECIDRA_BIND_ADDR` |
| GPU | GPU 1 | `CUDA_VISIBLE_DEVICES=1` before process start |
| Checkpoint | `/srv/models/jev-awq/awq-w4a16-v2` (W4A16) | `DECIDRA_NATIVE_MODEL`, `DECIDRA_CHECKPOINT` |
| cuBLAS | loader path | `DECIDRA_CUDA_LIBS` prepended to `LD_LIBRARY_PATH` |

Device selection is process-level: the native engine opens ordinal 0 of the visible set,
so `CUDA_VISIBLE_DEVICES=1` is the supported way to pin GPU 1. `DECIDRA_NATIVE_DEVICE` is not
read.

Port `8501` is chosen so the existing Python front on `8500` and its production backend on
`8001` remain untouched.

## LiteLLM

`/v1/chat/completions` and `/v1/completions` are OpenAI-compatible decision routes, so
LiteLLM can proxy the server directly:

```yaml
model_list:
  - model_name: jev-decisions
    litellm_params:
      model: openai/jev-awq
      api_base: http://127.0.0.1:8501/v1
      api_key: unused
```

The user content / `prompt` must be a JSON decision request. The assistant message is the
compact decision JSON (`choice`, `probabilities`, `confidence`, `usage`), so downstream
code can parse it as JSON rather than reading free-form text. Streaming is supported
(SSE `chat.completion.chunk` events followed by `data: [DONE]`). A plain text prompt is a
400; it is never silently coerced into a decision.

## Prerequisites

Build:

- Stable Rust (edition 2024; see `rust-toolchain.toml`).
- `nvcc` for the native kernels, or Docker so `native/build.rs` can compile them in a
  container. Override with `DECIDRA_NVCC=/path/to/nvcc` or
  `DECIDRA_CUDA_IMAGE=nvidia/cuda:13.1.1-devel-ubuntu24.04`.

Runtime:

- NVIDIA driver exposing `libcuda.so.1`.
- `libcublas.so.13` and `libcublasLt.so.13`. Set `DECIDRA_CUDA_LIBS` if they are not on the
  loader path.
- RTX 4090 / sm_89.
- ~13.1 GiB free VRAM for Marlin's resident repacked weights plus workspace; peak is
  **18.86 GiB at 16K** and **19.09 GiB at 32K** (24 GB card, ~4.4 GiB headroom).
- Marlin + tiled FA2 + gated split GDN full-forward latency: 1K **488 ms**, 4K
  **2.12-2.18 s**, 8K **4.23-4.32 s**, 16K **8.62-8.97 s**, 32K **18.60-19.03 s**.
  These are GEMM-bound (~0.49-0.58 ms/token); a 16,444-token `/v1/decide` completed in
  8.69 s and the 32,763-token probe in 18.66 s.
- **32K is the verified maximum context** on 24 GB. Split GDN is gated to sequences
  >= **512 tokens**; shorter prompts use the accurate shared recurrence. For
  **seq >= 24,576** a low-memory buffer-pool mode releases dropped tensors and frees
  large intermediates early (1K/16K timings unchanged).

`curl` is used by the health check and smoke script. `python3` is optional and only used
by the development evaluation tooling.

## Build and start

```bash
# Mock-backend tests (no GPU) and the shipped binary.
cargo test
cargo build --release --features native

# Validate tokenizer/head/calibration without loading the engine.
./target/release/decidra --check-config

# One real forward on GPU 1, then exit.
CUDA_VISIBLE_DEVICES=1 DECIDRA_CUDA_LIBS=/opt/cuda-libs \
  ./target/release/decidra --native-check

# Serve.
CUDA_VISIBLE_DEVICES=1 \
DECIDRA_CUDA_LIBS=/opt/cuda-libs \
DECIDRA_NATIVE_MODEL=/srv/models/jev-awq/awq-w4a16-v2 \
DECIDRA_CHECKPOINT=/srv/models/jev-awq/awq-w4a16-v2 \
DECIDRA_BIND_ADDR=0.0.0.0:8501 \
DECIDRA_SIGNING_KEY="$(cat /run/secrets/jev-signing-key)" \
./target/release/decidra
```

`scripts/run-native.sh` performs the build (unless `DECIDRA_SKIP_BUILD=1`), sets
`CUDA_VISIBLE_DEVICES=1`, and execs the binary. It sets `LD_LIBRARY_PATH` only when
`DECIDRA_CUDA_LIBS` is non-empty.

Startup loads the tokenizer/head/calibration, validates the A-P verbalizer parity, opens
the native checkpoint, and preloads/re-packs the decoder layers. Readiness requires all of
those steps to succeed; there is no partial serving.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `DECIDRA_BIND_ADDR` | `0.0.0.0:8080` | HTTP listen address. |
| `DECIDRA_MODEL` | `jev-awq` | Model id in `/v1/models`. |
| `DECIDRA_CHECKPOINT` | `/srv/models/jev-awq/awq-w4a16-v2` | Tokenizer/head/calibration artifact. |
| `DECIDRA_NATIVE_MODEL` | `$DECIDRA_CHECKPOINT` | Checkpoint opened by the native engine. |
| `DECIDRA_NATIVE_CACHE` | `true` | Keep packed decoder weights resident. |
| `DECIDRA_NATIVE_MARLIN` | `true` | Repack to the Marlin W4A16 layout during preload and use Marlin. |
| `DECIDRA_NATIVE_FUSED_W4A16` | `true` | Fallback-only hybrid fused policy; ignored when Marlin is on. |
| `DECIDRA_NATIVE_BATCH` | `true` | Native request batcher. `false` means one request per forward and window 0. |
| `DECIDRA_NATIVE_MAX_BATCH` | `8` | Max requests per batch. Throughput saturates at batch 4; 8 lets a burst drain in one pass, while 12/16 add p50 without throughput. |
| `DECIDRA_NATIVE_BATCH_WINDOW_US` | `2000` | Batch window in microseconds (max `50000`). The 2 ms default keeps a lone request at ~62 ms p50; 5/10 ms fill wider batches but did not raise throughput. |
| `DECIDRA_NATIVE_MAX_BATCH_TOKENS` | `16384` | Max total prompt tokens per batch (min `512`); a larger single request runs alone. |
| `DECIDRA_NATIVE_GDN_CHUNKED` | `true` | Chunked TF32 WMMA GDN prefill for sequences `>= 512` tokens; shorter prompts keep the shared/split recurrence. `false` uses the legacy path. |
| `DECIDRA_DECISION_CACHE_SIZE` | `256` | Exact-decision logits cache entries (LRU, exact token/head-id key); `0` disables. Successful forwards only; exact prompts only, no prefix caching. |
| `DECIDRA_SIGNING_KEY` | unset | HMAC-SHA256 response signing. |
| `DECIDRA_METRICS_ENABLED` | `true` | Prometheus recorder. |
| `DECIDRA_CUDA_LIBS` | unset | Directory with `libcublas.so.13`/`libcublasLt.so.13`. |
| `CUDA_VISIBLE_DEVICES` | `1` in scripts | GPU pin. |
| `RUST_LOG` | `decidra=info` | Log filter. |

## W4A8 checkpoint

The W4A8 artifact (`aioutfitters/JEV-27B-VL-AWQ-INT4-W4A8`) uses group-128
symmetric INT4, per-token INT8 activations, and a rank-32 PEFT LoRA. Run it
with:

```bash
export DECIDRA_CHECKPOINT=/srv/models/jev-w4a8
export DECIDRA_NATIVE_MODEL="$DECIDRA_CHECKPOINT"
export DECIDRA_TOKENIZER="$DECIDRA_CHECKPOINT/tokenizer.json"
export DECIDRA_LORA_ADAPTER="$DECIDRA_CHECKPOINT/adapter_vllm"
export DECIDRA_NATIVE_INT8_EXCLUDE=none
export DECIDRA_NATIVE_INT8_GROUP_SCALE=false
export DECIDRA_NATIVE_GPU_EMBED=false      # 21.2 GiB peak; GPU table is 23.3 GiB
```

The adapter directory must contain `adapter_config.json` and
`adapter_model.safetensors`. The engine applies the LoRA to every target linear
and to the `lm_head`; the scale is `lora_alpha / r` from the adapter config.

## Docker

The image builds the native kernels with `nvcc` from a CUDA devel image. The
runtime is Ubuntu 24.04 plus the server binary and the real `libcublas.so.13` /
`libcublasLt.so.13` files (versioned files plus stable symlinks) copied from the
devel builder into `/opt/decidra/cudalibs/`; the file also sets `LD_LIBRARY_PATH`.
The CUDA driver comes from the host. Weights are mounted, never baked in; Marlin
is the default (`DECIDRA_NATIVE_FUSED_W4A16` is not set in the image). Kernel
cubins and the FlashInfer prefill library are embedded in the binary.

```bash
docker build -t decidra:0.1.0 .

docker run --rm \
  --gpus '"device=1"' \
  --user "$(id -u):$(id -g)" \
  -p 8501:8080 \
  -v /srv/models/jev-awq/awq-w4a16-v2:/models:ro \
  -e DECIDRA_SIGNING_KEY \
  decidra:0.1.0
```

`--user` is needed because the checkpoint files are usually mode 600. The image
defaults the checkpoint, native model, and tokenizer to `/models`.

No-GPU image checks:

```bash
docker run --rm decidra:0.1.0 --help
docker run --rm --user "$(id -u):$(id -g)" \
  -v /srv/models/jev-awq/awq-w4a16-v2:/models:ro \
  decidra:0.1.0 --check-config
docker run --rm --entrypoint sh decidra:0.1.0 \
  -c "ls -l /opt/decidra/cudalibs && test -f /opt/decidra/cudalibs/libcublas.so.13 && test -f /opt/decidra/cudalibs/libcublasLt.so.13"
```

To use the fallback path instead of Marlin, set
`-e DECIDRA_NATIVE_MARLIN=false -e DECIDRA_NATIVE_FUSED_W4A16=true`.

Inside the container the runtime exposes the reserved device as ordinal 0, so do not set
`CUDA_VISIBLE_DEVICES` yourself.

Compose reserves GPU 1, mounts the model directory read-only, and matches the
host UID for the model files:

```bash
docker compose up -d --build
docker compose logs -f
```

The health check starts after a 180 s grace period.

## systemd

See the unit example in the README. Keep `CUDA_VISIBLE_DEVICES=1` and `DECIDRA_CUDA_LIBS` in
the unit environment or an `EnvironmentFile`; keep `DECIDRA_SIGNING_KEY` in a mode-0600 file.
`Restart=on-failure` is safe: each start re-preloads the checkpoint.

## Health and metrics

- `GET /health` returns `{"status":"ready", ...}` once the engine is loaded.
- `GET /metrics` exposes HTTP and backend counters/histograms.
- `GET /v1/decide/info` reports protocol, option limits, temperatures, and default
  strategy without running a forward.

A liveness probe should use `/health`. There is no separate readiness state: the process
does not bind until the engine is ready.

## Shutdown

SIGTERM/Ctrl-C triggers axum's graceful shutdown. In-flight decision passes finish; the
process then exits and the CUDA context is released. `DECIDRA_NATIVE_CACHE=false` is available
for debugging streaming mode but re-reads packed layers per forward and should not be used
in production.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `libcublas.so.13: cannot open shared object file` | Set `DECIDRA_CUDA_LIBS` to the cuBLAS directory or add it to `ldconfig`. |
| `CUDA driver version is insufficient` | Driver too old for the CUDA 13 engine build. |
| Startup blocks for ~10 s with disk/GPU work | `DECIDRA_NATIVE_CACHE=true` repacks/uploads ~13.1 GiB once; wait for the preload log line. |
| `/v1/decide` returns 502 | Native forward failed (check logs); no fallback backend exists. |
| `/v1/decide` returns 400 for image states | Expected: native vision is not implemented. |
| `/v1/chat/completions` returns 400 | Expected: the user message must be a JSON decision request (`{"kind","state","question","options"}`). |
| OOM during preload | Free GPU 1; do not run another model process there. |

Never expose the Python shim on `8500` or the production backend on `8001` as part of this
deployment.

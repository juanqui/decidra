# Deployment

## Topology

```text
GPU 1 ── native CUDA engine (in-process, packed AWQ W4A16 resident) ──┐
                                                                      │
CPU   ── jev-engine HTTP server ── 0.0.0.0:8501 ── clients / proxy ───┘
```

One process, one GPU. The axum server is CPU-side; inference runs through
`jev_native::Engine` on GPU 1 and keeps packed weights resident. There is no external
model server, Python, or PyTorch in the process.

## Ports and devices

| Resource | Default | Knobs |
|---|---|---|
| HTTP listen | `0.0.0.0:8080` (script: `0.0.0.0:8501`) | `JEV_BIND_ADDR` |
| GPU | GPU 1 | `CUDA_VISIBLE_DEVICES=1` before process start |
| Checkpoint | `/srv/models/jev-awq/awq-w4a16-v2` | `JEV_NATIVE_MODEL`, `JEV_CHECKPOINT` |
| cuBLAS | loader path | `JEV_CUDA_LIBS` prepended to `LD_LIBRARY_PATH` |

Device selection is process-level: the native engine opens ordinal 0 of the visible set,
so `CUDA_VISIBLE_DEVICES=1` is the supported way to pin GPU 1. `JEV_NATIVE_DEVICE` is not
read.

Port `8501` is chosen so the existing Python front on `8500` and its production backend on
`8001` remain untouched.

## Prerequisites

Build:

- Stable Rust (edition 2024; see `rust-toolchain.toml`).
- `nvcc` for the native kernels, or Docker so `native/build.rs` can compile them in a
  container. Override with `JEV_NVCC=/path/to/nvcc` or
  `JEV_CUDA_IMAGE=nvidia/cuda:13.1.1-devel-ubuntu24.04`.

Runtime:

- NVIDIA driver exposing `libcuda.so.1`.
- `libcublas.so.13` and `libcublasLt.so.13`. Set `JEV_CUDA_LIBS` if they are not on the
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
./target/release/jev-engine --check-config

# One real forward on GPU 1, then exit.
CUDA_VISIBLE_DEVICES=1 JEV_CUDA_LIBS=/opt/cuda-libs \
  ./target/release/jev-engine --native-check

# Serve.
CUDA_VISIBLE_DEVICES=1 \
JEV_CUDA_LIBS=/opt/cuda-libs \
JEV_NATIVE_MODEL=/srv/models/jev-awq/awq-w4a16-v2 \
JEV_CHECKPOINT=/srv/models/jev-awq/awq-w4a16-v2 \
JEV_BIND_ADDR=0.0.0.0:8501 \
JEV_SIGNING_KEY="$(cat /run/secrets/jev-signing-key)" \
./target/release/jev-engine
```

`scripts/run-native.sh` performs the build (unless `JEV_SKIP_BUILD=1`), sets
`CUDA_VISIBLE_DEVICES=1`, and execs the binary. It sets `LD_LIBRARY_PATH` only when
`JEV_CUDA_LIBS` is non-empty.

Startup loads the tokenizer/head/calibration, validates the A-P verbalizer parity, opens
the native checkpoint, and preloads/re-packs the decoder layers. Readiness requires all of
those steps to succeed; there is no partial serving.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `JEV_BIND_ADDR` | `0.0.0.0:8080` | HTTP listen address. |
| `JEV_MODEL` | `jev-awq` | Model id in `/v1/models`. |
| `JEV_CHECKPOINT` | `/srv/models/jev-awq/awq-w4a16-v2` | Tokenizer/head/calibration artifact. |
| `JEV_NATIVE_MODEL` | `$JEV_CHECKPOINT` | Checkpoint opened by the native engine. |
| `JEV_NATIVE_CACHE` | `true` | Keep packed decoder weights resident. |
| `JEV_NATIVE_MARLIN` | `true` | Repack to the Marlin W4A16 layout during preload and use Marlin. |
| `JEV_NATIVE_FUSED_W4A16` | `true` | Fallback-only hybrid fused policy; ignored when Marlin is on. |
| `JEV_NATIVE_BATCH` | `true` | Native request batcher. `false` means one request per forward and window 0. |
| `JEV_NATIVE_MAX_BATCH` | `8` | Max requests per batch. Throughput saturates at batch 4; 8 lets a burst drain in one pass, while 12/16 add p50 without throughput. |
| `JEV_NATIVE_BATCH_WINDOW_US` | `2000` | Batch window in microseconds (max `50000`). The 2 ms default keeps a lone request at ~62 ms p50; 5/10 ms fill wider batches but did not raise throughput. |
| `JEV_NATIVE_MAX_BATCH_TOKENS` | `16384` | Max total prompt tokens per batch (min `512`); a larger single request runs alone. |
| `JEV_MAX_CONCURRENCY` | `8` | Accepted and logged, but not enforced as a semaphore in this build. The batch worker executes one batch at a time. |
| `JEV_SIGNING_KEY` | unset | HMAC-SHA256 response signing. |
| `JEV_METRICS_ENABLED` | `true` | Prometheus recorder. |
| `JEV_CUDA_LIBS` | unset | Directory with `libcublas.so.13`/`libcublasLt.so.13`. |
| `CUDA_VISIBLE_DEVICES` | `1` in scripts | GPU pin. |
| `RUST_LOG` | `jev_engine=info` | Log filter. |

## Docker

The image builds the native kernels with `nvcc` from a CUDA devel image. The runtime image
is not guaranteed to ship cuBLAS, so the build copies the real `libcublas.so.13` and
`libcublasLt.so.13` (versioned files plus stable symlinks) from the devel builder into
`/opt/jev/cudalibs/` and prepends that directory to `LD_LIBRARY_PATH`; it does not
overwrite the base image's CUDA libraries or require `ldconfig`. Weights are mounted,
never baked in; Marlin is the default (`JEV_NATIVE_FUSED_W4A16` is not set in the image).

```bash
docker build -t jev-engine:0.2.0 .

docker run --rm \
  --gpus '"device=1"' \
  -p 8501:8080 \
  -v /srv/models/jev-awq:/models:ro \
  -e JEV_NATIVE_MODEL=/models/awq-w4a16-v2 \
  -e JEV_CHECKPOINT=/models/awq-w4a16-v2 \
  -e JEV_SIGNING_KEY \
  jev-engine:0.2.0
```

No-GPU image checks:

```bash
docker run --rm jev-engine:0.2.0 --help
docker run --rm -v /srv/models/jev-awq:/models:ro \
  -e JEV_NATIVE_MODEL=/models/awq-w4a16-v2 \
  -e JEV_CHECKPOINT=/models/awq-w4a16-v2 \
  jev-engine:0.2.0 --check-config
docker run --rm --entrypoint sh jev-engine:0.2.0 \
  -c "ls -l /opt/jev/cudalibs && test -f /opt/jev/cudalibs/libcublas.so.13 && test -f /opt/jev/cudalibs/libcublasLt.so.13"
```

To use the fallback path instead of Marlin, set
`-e JEV_NATIVE_MARLIN=false -e JEV_NATIVE_FUSED_W4A16=true`.

Inside the container the runtime exposes the reserved device as ordinal 0, so do not set
`CUDA_VISIBLE_DEVICES` yourself.

Compose reserves GPU 1 and mounts the model directory read-only:

```bash
docker compose up -d --build
docker compose logs -f
```

The health check starts after a 300 s grace period because startup uploads the packed
weights.

## systemd

See the unit example in the README. Keep `CUDA_VISIBLE_DEVICES=1` and `JEV_CUDA_LIBS` in
the unit environment or an `EnvironmentFile`; keep `JEV_SIGNING_KEY` in a mode-0600 file.
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
process then exits and the CUDA context is released. `JEV_NATIVE_CACHE=false` is available
for debugging streaming mode but re-reads packed layers per forward and should not be used
in production.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `libcublas.so.13: cannot open shared object file` | Set `JEV_CUDA_LIBS` to the cuBLAS directory or add it to `ldconfig`. |
| `CUDA driver version is insufficient` | Driver too old for the CUDA 13 engine build. |
| Startup blocks for ~10 s with disk/GPU work | `JEV_NATIVE_CACHE=true` repacks/uploads ~13.1 GiB once; wait for the preload log line. |
| `/v1/decide` returns 502 | Native forward failed (check logs); no fallback backend exists. |
| `/v1/decide` returns 400 for image states | Expected: native vision is not implemented. |
| `/v1/chat/completions` returns 501 | Expected: native generation is not implemented. |
| OOM during preload | Free GPU 1; do not run another model process there. |

Never expose the Python shim on `8500` or the production backend on `8001` as part of this
deployment.

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
- ~13.1 GiB free VRAM for Marlin's resident repacked weights plus workspace; long context
  with tiled FA2 peaks at ~18.86 GiB at 16K.
- Marlin + tiled FA2 full-forward latency: 4K **3.99-4.04 s**, 8K **7.95-8.05 s**,
  16K **16.10-16.32 s**. These are GEMM-bound (~0.96 ms/token), not attention-bound.

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
| `JEV_MAX_CONCURRENCY` | `8` | Outstanding decision passes. Forwards are serialized. |
| `JEV_SIGNING_KEY` | unset | HMAC-SHA256 response signing. |
| `JEV_METRICS_ENABLED` | `true` | Prometheus recorder. |
| `JEV_CUDA_LIBS` | unset | Directory with `libcublas.so.13`/`libcublasLt.so.13`. |
| `CUDA_VISIBLE_DEVICES` | `1` in scripts | GPU pin. |
| `RUST_LOG` | `jev_engine=info` | Log filter. |

## Docker

The image builds the native kernels with `nvcc` from a CUDA devel image and runs on a CUDA
runtime image (driver API + cuBLAS). Weights are mounted, never baked in.

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

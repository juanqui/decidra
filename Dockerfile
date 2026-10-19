# Server image. Model weights are never baked into the image.
#
# The Rust HTTP server is CPU-side; the in-process `jev-native` engine needs one
# NVIDIA GPU (GPU 1 in this deployment). Build compiles the native CUDA kernels
# with nvcc from the CUDA devel image; the runtime image carries the CUDA driver
# API and cuBLAS/cuBLASLt.
#
# Build:
#   docker build -t jev-engine:0.2.0 .
# Run (GPU 1, model dir mounted read-only):
#   docker run --rm --gpus '"device=1"' -p 8501:8080 \
#     -v /srv/models/jev-awq:/models:ro \
#     -e JEV_NATIVE_MODEL=/models/awq-w4a16-v2 \
#     -e JEV_CHECKPOINT=/models/awq-w4a16-v2 \
#     jev-engine:0.2.0

FROM nvidia/cuda:13.1.1-devel-ubuntu24.04 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain 1.85.0
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY native ./native
RUN cargo build --release --locked --features native

FROM nvidia/cuda:13.1.1-runtime-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && (ldconfig -p | grep -q 'libcublas\.so\.13' \
        || { echo "error: CUDA runtime image does not provide libcublas.so.13" >&2; exit 1; }) \
    && groupadd --system --gid 10001 jev \
    && useradd --system --uid 10001 --gid jev \
        --home-dir /nonexistent --shell /usr/sbin/nologin jev

COPY --from=builder /app/target/release/jev-engine /usr/local/bin/jev-engine

ENV JEV_BIND_ADDR=0.0.0.0:8080 \
    JEV_NATIVE_MODEL=/models/awq-w4a16-v2 \
    JEV_CHECKPOINT=/models/awq-w4a16-v2 \
    JEV_NATIVE_CACHE=true \
    JEV_NATIVE_FUSED_W4A16=true \
    JEV_MAX_CONCURRENCY=8 \
    JEV_METRICS_ENABLED=true

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

USER jev
ENTRYPOINT ["/usr/local/bin/jev-engine"]

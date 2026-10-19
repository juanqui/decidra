# Server image. Model weights are never baked into the image.
#
# The Rust HTTP server is CPU-side; the in-process `jev-native` engine needs one
# NVIDIA GPU (GPU 1 in this deployment). Build compiles the native CUDA kernels
# with nvcc from the CUDA devel image; cuBLAS/cuBLASLt are copied from that
# builder into the runtime image because the CUDA runtime tag may omit them.
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
    | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY native ./native
RUN cargo build --release --locked --features native

# Stage the real cuBLAS libraries (follow symlinks, then recreate the version
# names) in a private directory so we do not overwrite the base image's CUDA libs.
RUN mkdir -p /staging/cudalibs \
    && for lib in libcublas libcublasLt; do \
         target="$(readlink -f /usr/local/cuda/lib64/${lib}.so.13)"; \
         test -f "$target" || { echo "missing ${lib}.so.13 in builder" >&2; exit 1; }; \
         base="$(basename "$target")"; \
         cp "$target" "/staging/cudalibs/$base"; \
         ln -s "$base" "/staging/cudalibs/${lib}.so.13"; \
         ln -s "${lib}.so.13" "/staging/cudalibs/${lib}.so"; \
       done

FROM nvidia/cuda:13.1.1-runtime-ubuntu24.04

# The CUDA runtime image is not guaranteed to ship cuBLAS/cuBLASLt, so bring
# them in from the devel builder under a private prefix.
COPY --from=builder /staging/cudalibs/ /opt/jev/cudalibs/

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && test -f /opt/jev/cudalibs/libcublas.so.13 \
    && test -f /opt/jev/cudalibs/libcublasLt.so.13 \
    && groupadd --system --gid 10001 jev \
    && useradd --system --uid 10001 --gid jev \
        --home-dir /nonexistent --shell /usr/sbin/nologin jev

COPY --from=builder /app/target/release/jev-engine /usr/local/bin/jev-engine

ENV JEV_BIND_ADDR=0.0.0.0:8080 \
    JEV_NATIVE_MODEL=/models/awq-w4a16-v2 \
    JEV_CHECKPOINT=/models/awq-w4a16-v2 \
    JEV_NATIVE_CACHE=true \
    JEV_NATIVE_MARLIN=true \
    JEV_MAX_CONCURRENCY=8 \
    JEV_METRICS_ENABLED=true \
    LD_LIBRARY_PATH=/opt/jev/cudalibs:/usr/local/cuda/lib64

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

USER jev
ENTRYPOINT ["/usr/local/bin/jev-engine"]

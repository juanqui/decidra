# syntax=docker/dockerfile:1.7
#
# Decidra runtime image. The model is NOT in the image: mount it read-only at
# /models. The kernels (cubins) and the FlashInfer prefill library are embedded
# in the binary at build time, so the runtime needs no build tree and no CUDA
# toolkit. Only cuBLAS/cuBLASLt and the NVIDIA driver are required at run time;
# the driver comes from the host through the NVIDIA container toolkit.

ARG CUDA_ARCH=sm_89
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS builder
ARG CUDA_ARCH

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config libssl-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH \
    CARGO_HOME=/usr/local/cargo

WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml README.md \
     LICENSE LICENSE-MIT LICENSE-APACHE NOTICE ./
COPY src ./src
COPY native ./native

# Compile the server with the in-process native backend. build.rs drives nvcc
# for both the engine kernels and vendored Marlin, and links the FlashInfer
# prefill library; all of it is embedded into the binary.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    DECIDRA_CUDA_ARCH=${CUDA_ARCH} \
    cargo build --release --locked --features native \
    && mkdir -p /out \
    && cp target/release/decidra /out/decidra

# Copy real cuBLAS files (following symlinks) under a private prefix.
RUN mkdir -p /out/cudalibs \
    && for lib in libcublas libcublasLt; do \
         target="$(readlink -f /usr/local/cuda/lib64/${lib}.so.13)"; \
         test -f "$target" || { echo "missing ${lib}.so.13" >&2; exit 1; }; \
         base="$(basename "$target")"; \
         cp "$target" "/out/cudalibs/$base"; \
         ln -s "$base" "/out/cudalibs/${lib}.so.13"; \
       done

FROM ubuntu:24.04

ARG CUDA_ARCH=sm_89

LABEL org.opencontainers.image.title="Decidra" \
      org.opencontainers.image.description="State-once inference engine for vllm-sr/d3-turbo" \
      org.opencontainers.image.source="https://github.com/juanqui/decidra" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0" \
      ai.decidra.cuda-arch="${CUDA_ARCH}"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 decidra \
    && useradd --system --uid 10001 --gid decidra \
        --home-dir /nonexistent --shell /usr/sbin/nologin decidra

COPY --from=builder /out/cudalibs/ /opt/decidra/cudalibs/
COPY --from=builder /out/decidra /usr/local/bin/decidra
COPY LICENSE LICENSE-MIT LICENSE-APACHE NOTICE /usr/share/licenses/decidra/

ENV DECIDRA_BIND_ADDR=0.0.0.0:8080 \
    DECIDRA_NATIVE_MODEL=/models \
    DECIDRA_CHECKPOINT=/models \
    DECIDRA_TOKENIZER=/models/tokenizer.json \
    DECIDRA_NATIVE_CACHE=true \
    DECIDRA_NATIVE_MARLIN=true \
    DECIDRA_NATIVE_INT8=true \
    DECIDRA_NATIVE_INT8_GROUP_SCALE=true \
    DECIDRA_NATIVE_INT8_EXCLUDE=self_attn.v_proj,self_attn.k_proj \
    DECIDRA_NATIVE_GDN_CHUNKED=true \
    DECIDRA_NATIVE_GDN_CHUNKED_MIN_SEQ=16 \
    DECIDRA_NATIVE_GPU_EMBED=false \
    DECIDRA_STATE_CACHE_ENTRIES=16 \
    DECIDRA_STATE_CACHE_BYTES=5368709120 \
    DECIDRA_STATE_CACHE_PIN=1 \
    DECIDRA_HOST_STATE_CACHE_BYTES=8589934592 \
    DECIDRA_BATCH_REQUESTS=32 \
    DECIDRA_BATCH_ROWS=32 \
    DECIDRA_BATCH_WINDOW_US=2000 \
    DECIDRA_STATE_ONCE_MAX_ROWS=64 \
    DECIDRA_METRICS_ENABLED=true \
    LD_LIBRARY_PATH=/opt/decidra/cudalibs

USER decidra
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/decidra"]

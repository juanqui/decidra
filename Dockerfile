# syntax=docker/dockerfile:1.7
#
# Decidra runtime image. The model is NOT in the image: mount it read-only at
# /models. The kernels (cubins) and the FlashInfer prefill library are embedded
# in the binary at build time, so the runtime needs no build tree and no CUDA
# toolkit. Only cuBLAS/cuBLASLt and the NVIDIA driver are required at run time;
# the driver comes from the host through the NVIDIA container toolkit.

FROM nvidia/cuda:13.1.1-devel-ubuntu24.04 AS builder

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
COPY Cargo.toml Cargo.lock rust-toolchain.toml README.md LICENSE-MIT LICENSE-APACHE ./
COPY src ./src
COPY native ./native

# Compile the server with the in-process native backend. build.rs drives nvcc
# for both the engine kernels and vendored Marlin, and links the FlashInfer
# prefill library; all of it is embedded into the binary.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
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

LABEL org.opencontainers.image.title="Decidra" \
      org.opencontainers.image.description="Native inference engine for decision models" \
      org.opencontainers.image.source="https://github.com/juanqui/decidra" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0"

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

ENV DECIDRA_BIND_ADDR=0.0.0.0:8080 \
    DECIDRA_NATIVE_MODEL=/models \
    DECIDRA_CHECKPOINT=/models \
    DECIDRA_TOKENIZER=/models/tokenizer.json \
    DECIDRA_NATIVE_CACHE=true \
    DECIDRA_NATIVE_MARLIN=true \
    DECIDRA_METRICS_ENABLED=true \
    LD_LIBRARY_PATH=/opt/decidra/cudalibs

USER decidra
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8515/health || exit 1

ENTRYPOINT ["/usr/local/bin/decidra"]

# Server image only. Model weights are never baked into the image.
#
# Build:
#   docker build -t jev-engine:0.1.0 .
# Run (host vLLM on 127.0.0.1:8002, model dir mounted read-only):
#   docker run --rm -p 8501:8080 \
#     -v /srv/models/jev-awq:/models:ro \
#     -e JEV_VLLM_URL=http://host.docker.internal:8002 \
#     -e JEV_CHECKPOINT=/models/awq-w4a16-v2 \
#     --add-host host.docker.internal:host-gateway \
#     jev-engine:0.1.0

FROM rust:1.85 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 jev \
    && useradd --system --uid 10001 --gid jev \
        --home-dir /nonexistent --shell /usr/sbin/nologin jev

COPY --from=builder /app/target/release/jev-engine /usr/local/bin/jev-engine

ENV JEV_BIND_ADDR=0.0.0.0:8080 \
    JEV_VLLM_URL=http://127.0.0.1:8002 \
    JEV_CHECKPOINT=/models/awq-w4a16-v2 \
    JEV_MAX_CONCURRENCY=8 \
    JEV_REQUEST_TIMEOUT_SECS=600 \
    JEV_METRICS_ENABLED=true

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

USER jev
ENTRYPOINT ["/usr/local/bin/jev-engine"]

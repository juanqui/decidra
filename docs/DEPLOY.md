# Deployment

This stack has two processes with separate responsibilities:

```text
GPU 1 ── vLLM 0.31.0 (AWQ W4A16, container jev-awq-test) ── 127.0.0.1:8002
                                                                    ▲
                                                          JEV_VLLM_URL
                                                                    │
CPU ── jev-engine (Rust API front) ── 0.0.0.0:8501 ── clients / proxy
                                └──── /metrics (Prometheus)
```

- **GPU 1 is the only GPU used by this stack.** The Rust server uses no GPU at all.
- **GPU 0 is not used by this stack.** These scripts never select, mount, or address GPU 0.
- The production stack on ports **8001** and **8500** is not touched or replaced.
- The Rust server never starts, stops, or supervises vLLM.

## Ports

| Component | Default | Override |
|---|---|---|
| vLLM HTTP | `127.0.0.1:8002` | `JEV_VLLM_BIND_HOST`, `JEV_VLLM_HOST_PORT` |
| jev-engine | `0.0.0.0:8501` | `JEV_BIND_ADDR` (native), `JEV_SERVER_PORT` (Compose) |
| Prometheus | `8501/metrics` | same as jev-engine |

Port `8501` is chosen so the existing Python front on `8500` and its production vLLM on
`8001` are left alone.

## Prerequisites

- Docker with the NVIDIA container toolkit (`docker run --gpus ...` works).
- The AWQ artifact at `/srv/models/jev-awq/awq-w4a16-v2` (read-only mount).
- For native builds: stable Rust with edition 2024 support. `Cargo.lock` is pinned so the
  `rust:1.85` builder used by the Dockerfile resolves ICU 1.x; build with `--locked`.
- `python3` and `curl` for `scripts/smoke.sh`.

## 1. Start the GPU 1 test backend

```bash
./scripts/run-vllm.sh
```

Defaults:

| Variable | Default | Meaning |
|---|---|---|
| `JEV_VLLM_IMAGE` | `vllm/vllm-openai:v0.31.0` | Backend image. |
| `JEV_VLLM_CONTAINER` | `jev-awq-test` | Disposable container name. |
| `JEV_VLLM_GPU` | `1` | Numeric GPU device id. Keep this at `1`. |
| `JEV_VLLM_BIND_HOST` | `127.0.0.1` | Host bind address for port mapping. |
| `JEV_VLLM_HOST_PORT` | `8002` | Host port mapped to container port 8000. |
| `JEV_MODELS_DIR` | `/srv/models/jev-awq` | Mounted read-only at `/models`. |
| `JEV_VLLM_HEALTH_TIMEOUT` | `900` | Seconds to wait for `/health`. |

The script launches `vllm/vllm-openai:v0.31.0` with:

```text
--model /models/awq-w4a16-v2
--served-model-name jev-awq
--logprobs-mode processed_logprobs
--max-model-len 16384
--enable-prefix-caching
--mamba-cache-mode align
--limit-mm-per-prompt {"image":1}
--trust-request-chat-template
--max-num-seqs 8
--gpu-memory-utilization 0.95
--kv-cache-dtype fp8
--port 8000
```

It waits for `http://127.0.0.1:8002/health` (override with
`JEV_VLLM_HEALTH_HOST`). On failure it prints recent logs and stops only the
`jev-awq-test` container. The container is started with `--rm` and a fixed name, so a
second launch refuses to proceed while it exists.

If running vLLM natively instead of in Docker, select GPU 1 explicitly:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /srv/models/jev-awq/awq-w4a16-v2 \
  --served-model-name jev-awq \
  --host 127.0.0.1 --port 8002 \
  --logprobs-mode processed_logprobs \
  --max-model-len 16384 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image":1}' \
  --max-num-seqs 8 \
  --trust-request-chat-template \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8
```

## 2. Run the Rust server natively

```bash
./scripts/run-server.sh
```

Defaults exported by the script:

```bash
JEV_VLLM_URL=http://127.0.0.1:8002
JEV_VLLM_MODEL=jev-awq
JEV_CHECKPOINT=/srv/models/jev-awq/awq-w4a16-v2
JEV_BIND_ADDR=0.0.0.0:8501
JEV_MAX_CONCURRENCY=8
JEV_REQUEST_TIMEOUT_SECS=600
JEV_METRICS_ENABLED=true
```

Any of these can be overridden in the environment. The script builds with
`cargo build --release --locked` unless `JEV_SKIP_BUILD=1`. Arguments after the script name
are passed to the binary, for example:

```bash
./scripts/run-server.sh --check-config
```

Set `JEV_SIGNING_KEY` in the environment to enable response signing; leave it unset/empty for
unsigned responses. Do not put the key on the command line.

## 3. Run the server in Docker

Compose (server only; vLLM stays on the host at `127.0.0.1:8002`):

```bash
docker compose up -d --build
docker compose logs -f jev-engine
docker compose ps
```

`docker-compose.yml`:

- Mounts `/srv/models/jev-awq` read-only at `/models`.
- Points `JEV_VLLM_URL` at `http://host.docker.internal:8002` by default.
- Adds `host.docker.internal:host-gateway` for Linux.
- Publishes `8501:8080`.
- Reserves **no GPU** for the server.
- Reads `JEV_SIGNING_KEY` from the environment / `.env`; an unset value runs unsigned.

Direct `docker run`:

```bash
docker build -t jev-engine:0.1.0 .
docker run --rm -p 8501:8080 \
  -v /srv/models/jev-awq:/models:ro \
  -e JEV_VLLM_URL=http://host.docker.internal:8002 \
  -e JEV_CHECKPOINT=/models/awq-w4a16-v2 \
  -e JEV_MAX_CONCURRENCY=8 \
  -e JEV_SIGNING_KEY="$JEV_SIGNING_KEY" \
  --add-host host.docker.internal:host-gateway \
  jev-engine:0.1.0
```

The image contains only the Rust binary and CA certificates; it never contains model
weights.

## 4. Health checks

```bash
# Backend
curl -fsS http://127.0.0.1:8002/health

# Rust server
curl -fsS http://127.0.0.1:8501/health
curl -fsS http://127.0.0.1:8501/v1/models
curl -fsS http://127.0.0.1:8501/v1/decide/info

# One real decision (requires the backend)
make smoke
# or
JEV_SMOKE_URL=http://127.0.0.1:8501 ./scripts/smoke.sh

# Prometheus
curl -fsS http://127.0.0.1:8501/metrics
```

`make smoke` fails on a non-2xx response, a missing/malformed `probabilities` vector, a
non-normalised distribution, an invalid `choice`, or missing usage fields.

Useful metrics:

- `jev_http_requests_total{method,path,status}`
- `jev_upstream_requests_total{endpoint,status}`
- `jev_upstream_request_duration_seconds`
- `jev_decide_total{kind,strategy}`
- `jev_upstream_tokens_total{direction}`

## 5. Restart and upgrade

```bash
# Restart the Rust server
docker compose restart jev-engine
# or, natively: Ctrl-C / SIGTERM, then ./scripts/run-server.sh

# Restart only the test backend
docker restart jev-awq-test

# Rebuild and replace the server
docker compose up -d --build jev-engine

# Stop the test backend (test container only; production is untouched)
docker stop jev-awq-test
```

The test vLLM container uses `--rm`, so `docker stop jev-awq-test` removes it. The script
refuses to reuse an existing container name; restart the existing one or remove it before
relaunching.

Startup validation loads the tokenizer and decision head and fails fast if the first 16
choice labels are not the trained A-P verbalizers. After changing the artifact or tokenizer,
run:

```bash
make check-config
```

## 6. Troubleshooting

| Symptom | Check |
|---|---|
| `--check-config` fails on `decision_head.json` | `JEV_CHECKPOINT` points at `awq-w4a16-v2`; the head is at `decision_head/decision_head.json`. |
| `--check-config` label parity error | Tokenizer does not belong to this model, or the head changed. |
| `/v1/decide` returns 502 | vLLM is down, the model name differs, or the backend rejected the prompt; inspect its logs. |
| `/v1/decide` returns 504 | Backend exceeded `JEV_REQUEST_TIMEOUT_SECS`; check GPU load and `JEV_MAX_CONCURRENCY` vs `--max-num-seqs`. |
| Image decisions fail | Backend must allow one image: `--limit-mm-per-prompt {"image":1}` and `--trust-request-chat-template`. |
| Container cannot reach vLLM | Use `host.docker.internal:8002` plus `host-gateway`, or the host's routable address. |
| Signing consumers reject responses | Verify `x-jev-signature: sha256=<hex>` over the exact body bytes with the same `JEV_SIGNING_KEY`. |

Never expose the Python shim on `8500` or the production vLLM on `8001` as part of this
deployment. This stack's public surface is the Rust server on `8501`.

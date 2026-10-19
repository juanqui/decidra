#!/usr/bin/env bash
# Launch the test vLLM backend for jev-engine on GPU 1 only.
#
# This starts a disposable container named jev-awq-test. It does not touch the
# production stack, its containers, ports 8001/8500, or GPU 0.
set -euo pipefail

IMAGE="${JEV_VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}"
CONTAINER="${JEV_VLLM_CONTAINER:-jev-awq-test}"
GPU="${JEV_VLLM_GPU:-1}"
BIND_HOST="${JEV_VLLM_BIND_HOST:-127.0.0.1}"
HOST_PORT="${JEV_VLLM_HOST_PORT:-8002}"
MODELS_DIR="${JEV_MODELS_DIR:-/srv/models/jev-awq}"
HEALTH_HOST="${JEV_VLLM_HEALTH_HOST:-127.0.0.1}"
HEALTH_TIMEOUT="${JEV_VLLM_HEALTH_TIMEOUT:-900}"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker is required" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required for the health wait" >&2
  exit 1
fi
if [[ ! -d "$MODELS_DIR" ]]; then
  echo "error: model directory not found: $MODELS_DIR" >&2
  exit 1
fi
if [[ ! -d "$MODELS_DIR/awq-w4a16-v2" ]]; then
  echo "error: expected $MODELS_DIR/awq-w4a16-v2" >&2
  exit 1
fi
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "error: JEV_VLLM_GPU must be a numeric device id (default 1), got: $GPU" >&2
  exit 1
fi
if ! [[ "$HOST_PORT" =~ ^[0-9]+$ ]] || (( HOST_PORT < 1 || HOST_PORT > 65535 )); then
  echo "error: JEV_VLLM_HOST_PORT must be a TCP port, got: $HOST_PORT" >&2
  exit 1
fi
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "error: container '$CONTAINER' already exists." >&2
  echo "       Stop/remove it, or set JEV_VLLM_CONTAINER to a different name." >&2
  exit 1
fi

echo "starting $CONTAINER: image=$IMAGE gpu=$GPU bind=${BIND_HOST}:${HOST_PORT} models=$MODELS_DIR"

container_id="$(
  docker run -d --rm \
    --name "$CONTAINER" \
    --ipc=host \
    --gpus "device=${GPU}" \
    -p "${BIND_HOST}:${HOST_PORT}:8000" \
    -v "${MODELS_DIR}:/models:ro" \
    -e VLLM_NO_USAGE_STATS=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$IMAGE" \
    --model /models/awq-w4a16-v2 \
    --served-model-name jev-awq \
    --logprobs-mode processed_logprobs \
    --max-model-len 16384 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --limit-mm-per-prompt '{"image":1}' \
    --trust-request-chat-template \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 \
    --port 8000
)"

echo "container id: $container_id"
echo "waiting for http://${HEALTH_HOST}:${HOST_PORT}/health (timeout ${HEALTH_TIMEOUT}s)"

deadline=$((SECONDS + HEALTH_TIMEOUT))
while (( SECONDS < deadline )); do
  if curl -fsS --max-time 5 "http://${HEALTH_HOST}:${HOST_PORT}/health" >/dev/null 2>&1; then
    echo "vLLM ready: http://${HEALTH_HOST}:${HOST_PORT}"
    echo "logs: docker logs -f $CONTAINER"
    exit 0
  fi
  if ! docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "error: container '$CONTAINER' exited during startup" >&2
    docker logs --tail 80 "$CONTAINER" 2>&1 || true
    exit 1
  fi
  sleep 5
done

echo "error: vLLM did not become healthy within ${HEALTH_TIMEOUT}s" >&2
docker logs --tail 120 "$CONTAINER" 2>&1 || true
docker stop "$CONTAINER" >/dev/null 2>&1 || true
exit 1

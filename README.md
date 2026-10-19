# jev-engine

Production Rust API/decision server for **`JEV-27B-VL` AWQ W4A16** on one RTX 4090.
It exposes the JEV System 1 decision protocol, the OpenAI decisions shape, and transparent
OpenAI chat/completions passthrough, while vLLM owns token generation and the quantized
kernels.

The server deliberately does **not** start, stop, supervise, or benchmark vLLM. Point it at
an already-running backend with `JEV_VLLM_URL` (default `http://127.0.0.1:8002`).

## Why vLLM kernels

The AWQ artifact needs the best available kernel stack for this exact model on Ada:

- **Marlin W4A16 GEMM** for the 4-bit compressed-tensors matmuls.
- **FlashInfer / FlashAttention** for attention.
- **FLA / FlashInfer GDN** for the hybrid linear-attention layers.

Reimplementing any of that in Rust would be slower and would change the numerics. The Rust
process owns the parts where exactness and operational control matter:

- HTTP/API surface, validation, error semantics.
- Tokenizer-derived wide-choice label table and startup parity checks.
- Prompt construction byte-for-byte compatible with the Python reference.
- Decision-head bias, per-kind temperature, optional fitted calibration, softmax.
- Concurrency limits, upstream timeouts, connection reuse.
- HMAC-SHA256 response signing and Prometheus metrics.

## Architecture

```text
                    ┌──────────────────────────── jev-engine ───────────────────────────┐
 client ──HTTP────▶ │ axum router                                                       │
                    │  ├─ GET  /health, /v1/models, /v1/decide/info, /metrics            │
                    │  ├─ POST /v1/decide ──▶ DecisionEngine ──▶ Upstream ──┐            │
                    │  │       prompt + label ids + head math              │            │
                    │  ├─ POST /v1/decisions ─▶ DecisionEngine ────────────┤            │
                    │  ├─ POST /v1/chat/completions ─▶ passthrough ────────┤            │
                    │  ├─ POST /v1/completions ─────▶ passthrough ────────┤            │
                    │  └─ middleware: HMAC signing, HTTP/upstream metrics  │            │
                    └──────────────────────────────────────────────────────┼────────────┘
                                                                           │ semaphore + keep-alive
                                                                           ▼
                                                    vLLM 0.31.0  (AWQ W4A16, GPU 1)
                                                    /v1/completions, /v1/chat/completions
```

One shared `reqwest::Client` (rustls, keep-alive) is used for every upstream request. A
`tokio::sync::Semaphore` bounds concurrent upstream calls, including the parallel passes of
`tournament` and `permute`. There is no unbounded `tokio::spawn`.

### Exactness details

These are not cosmetic; each one changes the prompt or the distribution:

- `serde_json` is built with `preserve_order`, and state objects are rendered by a custom
  writer that matches CPython `json.dumps(value, ensure_ascii=False)`: insertion order,
  `", "`/`": "` separators, raw non-ASCII, and Python float notation.
- The `permute` strategy reproduces CPython's `random.Random(crc32(question))` MT19937
  seeding and `random.sample(range(n), n)` selection algorithm, so the four option orders
  are identical to the Python reference.
- Label ids are derived with the real tokenizer using the reference's single-token rule.
  Startup fails if the first 16 ids are not the decision head's trained A-P verbalizers.
- Prompt bytes, option lines, image part placement, and the raw chat template are ported
  directly from `decide_shim.py`.

## Repository layout

```text
src/
  main.rs       CLI, tracing, startup, graceful shutdown
  api.rs        axum routes, request/response types, decisions mapping, proxy, middleware
  engine.rs     single / tournament / permute strategies and usage accounting
  head.rs       decision_head.json + calibration.json loading, bias/temperature/softmax
  prompt.rs     label table, _parts, build_content, RAW_TEMPLATE, tokenizer loading
  pyjson.rs     CPython-compatible JSON/str/truthiness rendering
  rng.rs        CPython-compatible MT19937 and `random.sample` permutations
  upstream.rs   vLLM client, logprob parsing, streaming proxy, concurrency limiter
  config.rs     environment configuration and defaults
  error.rs      upstream errors and structured API errors
  telemetry.rs  Prometheus recorder installation
tests/
  integration.rs          mock-vLLM end-to-end tests (noul/choice/score/image/signing/proxy)
  fixtures/               tiny tokenizer + real decision-head metadata (no weights)
docs/SPEC.md              API contract, math, and edge cases
```

## Build and run

Prerequisites: stable Rust with edition 2024 support (see `rust-toolchain.toml`), and a
reachable vLLM backend.

```bash
cargo build --release
cargo test
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

Validate tokenizer/head/calibration without serving:

```bash
cargo run --release -- --check-config
```

Run the server:

```bash
JEV_VLLM_URL=http://127.0.0.1:8002 \
JEV_CHECKPOINT=/srv/models/jev-awq/awq-w4a16-v2 \
JEV_BIND_ADDR=0.0.0.0:8080 \
JEV_MAX_CONCURRENCY=8 \
JEV_REQUEST_TIMEOUT_SECS=600 \
JEV_SIGNING_KEY="$(cat /run/secrets/jev-signing-key)" \
./target/release/jev-engine
```

`--check-config` prints the resolved paths, label count, temperatures, and the derived
256 choice label ids, then exits. Startup always performs the same validation and refuses
to serve on a tokenizer/head mismatch.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `JEV_BIND_ADDR` | `0.0.0.0:8080` | Listen address. |
| `JEV_VLLM_URL` | `http://127.0.0.1:8002` | vLLM base URL; trailing `/` is removed. |
| `JEV_VLLM_MODEL` | `jev-awq` | Model name sent upstream. |
| `JEV_MODEL` | — | Alias for `JEV_VLLM_MODEL` when that is unset. |
| `JEV_CHECKPOINT` | `/srv/models/jev-awq/awq-w4a16-v2` | AWQ artifact directory. |
| `JEV_TOKENIZER` | checkpoint `tokenizer.json` | Tokenizer file or directory override. |
| `JEV_CALIBRATION` | discovered | `calibration.json` override. |
| `JEV_CALIBRATION_FITTED` | unset | Optional `fit_bias.py` output. Unset/absent disables it. |
| `JEV_HF_SNAPSHOT` | `.../snapshots/ba3f0d58...` | Fallback for artifacts missing from the AWQ dir. |
| `JEV_MAX_CONCURRENCY` | `8` | Max concurrent upstream HTTP requests (semaphore). |
| `JEV_REQUEST_TIMEOUT_SECS` | `600` | Per-upstream decision request timeout. Timeouts become HTTP 504. |
| `JEV_CONNECT_TIMEOUT_SECS` | `10` | TCP connect timeout. |
| `JEV_SIGNING_KEY` | unset | Enables `x-jev-signature`; empty string disables. |
| `JEV_METRICS_ENABLED` | `true` | Install the Prometheus recorder. |
| `JEV_MAX_BODY_BYTES` | `67108864` | Axum request body limit (images as data URLs). |
| `RUST_LOG` | `jev_engine=info` | `tracing-subscriber` filter. |

Calibration discovery order, relative to the checkpoint then the HF snapshot:

1. `<checkpoint>/decision_head/decision_head.json`
2. `<checkpoint>/adapter_vllm/decision_head.json`
3. `<checkpoint>/decision_head.json`
4. the same names under `JEV_HF_SNAPSHOT`

Calibration follows the same pattern with `decision_head/calibration.json` then
`calibration.json`. The production artifact at `/srv/models/jev-awq/awq-w4a16-v2` resolves
to `decision_head/decision_head.json` and `decision_head/calibration.json`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness and public model name. |
| `GET` | `/v1/models` | `autotrust/JEV-27B-VL`, `autotrust/JEV-27B`, and the upstream id. |
| `GET` | `/v1/decide/info` | Protocol, max options, native 16, temperatures, default strategy. |
| `GET` | `/metrics` | Prometheus text exposition. |
| `POST` | `/v1/decide` | JEV native decision (`noul`, `score`, `choice`). |
| `POST` | `/v1/decisions` | OpenAI Decisions API shape (`predicate`, `choice`, `score`). |
| `POST` | `/v1/chat/completions` | Transparent vLLM passthrough, model rewritten. |
| `POST` | `/v1/completions` | Transparent vLLM passthrough, model rewritten. |

Examples:

```bash
curl -s http://127.0.0.1:8080/v1/decide \
  -H 'content-type: application/json' \
  -d '{"kind":"noul","state":"The user asked for a refund twice.","question":"Is this abusive?"}'

curl -s http://127.0.0.1:8080/v1/decide \
  -H 'content-type: application/json' \
  -d '{"kind":"choice","state":"screenshot text","question":"Pick the best category",
       "options":["billing","abuse","spam"],"strategy":"auto"}'

curl -s http://127.0.0.1:8080/v1/decide \
  -H 'content-type: application/json' \
  -d '{"kind":"choice","state":["look at this",{"image":"data:image/png;base64,..."}],
       "question":"Is the receipt visible?","options":["no","yes"]}'
```

Image states mix text and at most one image; `{"image": "..."}` and OpenAI
`{"type":"image_url","image_url":{"url":"..."}}` parts are both accepted. Image decisions go
to vLLM `/v1/chat/completions` with the reference raw template and
`add_generation_prompt: false`.

## Strategies

`strategy` is `auto | single | tournament | permute`.

- `auto` is `tournament`; for choices of 16 or fewer, every strategy degrades to a single
  pass (the reference behavior). `adaptation` reports `native` for those cases.
- `single` sends one 1-token read-out over all options (up to 256 ids, chunked by 128).
- `tournament` splits options into `ceil(n/16)` near-equal groups, reads each group, keeps the
  winner of each group plus the best remaining options up to 16, runs one final 16-way pass,
  and combines group and final distributions using the reference formula.
- `permute` runs the identity order plus three CPython-compatible shuffles seeded by
  `crc32(question)` and averages the mapped-back probabilities.

See `docs/SPEC.md` for the exact equations.

## Signing

When `JEV_SIGNING_KEY` is non-empty, every response whose path starts with `/v1/decide` or
`/v1/decisions` gets:

```text
x-jev-signature: sha256=<hex HMAC-SHA256 over the exact response body bytes>
```

This includes `/v1/decide/info` and error responses on those paths. `/health`, `/metrics`,
`/v1/models`, and the passthrough routes are not signed. The signature is calculated after
the body is produced and before it is sent; `content-type` is preserved.

Verify with Python:

```python
import hashlib, hmac
expected = "sha256=" + hmac.new(key, body, hashlib.sha256).hexdigest()
assert expected == headers["x-jev-signature"]
```

## Correctness and evaluation

```bash
cargo test
cargo run --release -- --check-config
```

`--check-config` is the important production check:

- Loads the real tokenizer and decision head.
- Derives the wide-choice labels with the reference single-token rule.
- Asserts that the first 16 ids equal `decision_head.verbalizer_ids[8:24]`.
- Prints `max_options` (256 with the production Qwen tokenizer).

Unit tests cover the exact text prompt, image part placement, Python-compatible JSON
rendering, label derivation, bias/temperature/softmax, fitted calibration, CRC32 + MT19937
permutation vectors generated from CPython, tournament grouping/combination, and logprob
parsing.

Integration tests spin up a mock vLLM (axum) and exercise noul/choice/score, tournament and
permute request counts, image chat-completions requests, OpenAI decisions mapping, model
rewriting, streaming passthrough, signing, 400/502/504 semantics, and malformed JSON.

To evaluate against the Python reference on the same backend:

1. Point both front ends at the same vLLM URL (for example `http://127.0.0.1:8002`).
2. Send the same JSON bodies to `/v1/decide`, `/v1/decisions`, and image requests.
3. Compare `probabilities`, `choice`, `adaptation`, and usage. Floating-point values should
   agree to serialization precision; prompt bytes and label ids should agree exactly.
4. For `permute`, the four option orders are intentionally identical to Python's, so
   differences indicate a real regression, not RNG drift.

A minimal comparison loop:

```bash
for body in corpus/*.json; do
  curl -s localhost:8500/v1/decide -H 'content-type: application/json' -d @"$body" > /tmp/py.json
  curl -s localhost:8080/v1/decide -H 'content-type: application/json' -d @"$body" > /tmp/rs.json
  python3 - "$body" <<'PY'
import json, sys
a, b = json.load(open('/tmp/py.json')), json.load(open('/tmp/rs.json'))
assert a['choice'] == b['choice'], (sys.argv[1], a['choice'], b['choice'])
assert max(abs(x-y) for x, y in zip(a['probabilities'], b['probabilities'])) < 1e-12
PY
done
```

(`localhost:8500` is the existing Python shim; do not expose it to the internet in place of
the signed Rust front.)

## One-GPU deployment example (GPU 1 only)

The Rust process needs no GPU. GPU work belongs exclusively to the separately managed vLLM
test instance on **GPU 1**, listening on the test port **8002**. Do not touch the production
stack's ports (`8001`, `8500`) or its model directory.

```bash
# Terminal 1: the test backend, GPU 1 only, port 8002.
# This command is illustrative; vLLM lifecycle is owned by the deployment tooling.
CUDA_VISIBLE_DEVICES=1 vllm serve /srv/models/jev-awq/awq-w4a16-v2 \
  --served-model-name jev-awq \
  --host 127.0.0.1 --port 8002 \
  --logprobs-mode processed_logprobs \
  --max-model-len 16384 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image": 1}' \
  --max-num-seqs 8 \
  --trust-request-chat-template \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8

# Terminal 2: the Rust front.
export JEV_VLLM_URL=http://127.0.0.1:8002
export JEV_CHECKPOINT=/srv/models/jev-awq/awq-w4a16-v2
export JEV_BIND_ADDR=0.0.0.0:8080
export JEV_MAX_CONCURRENCY=8
export JEV_SIGNING_KEY="$(cat /run/secrets/jev-signing-key)"
exec ./target/release/jev-engine
```

Example systemd unit:

```ini
[Unit]
Description=JEV decision server (Rust front for vLLM)
After=network-online.target

[Service]
Type=simple
User=jev
WorkingDirectory=/opt/jev-engine
Environment=JEV_VLLM_URL=http://127.0.0.1:8002
Environment=JEV_CHECKPOINT=/srv/models/jev-awq/awq-w4a16-v2
Environment=JEV_BIND_ADDR=0.0.0.0:8080
Environment=JEV_MAX_CONCURRENCY=8
# Provide JEV_SIGNING_KEY through this mode-0600 file (or a systemd credential),
# never on the command line. Remove the line to run unsigned.
EnvironmentFile=-/etc/jev-engine/env
ExecStart=/opt/jev-engine/jev-engine
Restart=on-failure
RestartSec=2
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

`JEV_SIGNING_KEY_FILE` is not read by the binary; load the secret with your unit's
`EnvironmentFile` or an `ExecStart` wrapper. Never commit the key.

## Performance notes

- Every decision read-out asks for exactly **one token** (`max_tokens: 1`). Latency is
  dominated by prompt prefill, not decode.
- Wide choices multiply prefill work: a 256-option tournament makes 16 group passes plus one
  final pass; a permute makes 4 passes. Passes run concurrently but are bounded by
  `JEV_MAX_CONCURRENCY` and share the same semaphore as normal traffic.
- Use `JEV_MAX_CONCURRENCY` at or below vLLM's `--max-num-seqs` so the front does not queue
  work the backend cannot batch. Keep-alive avoids a TCP/TLS setup per pass.
- vLLM prefix caching helps because every pass repeats the same state/question prefix with a
  different option block.
- Rust overhead is small relative to prefill; the measurable levers are option count,
  strategy, prompt length, and backend batch capacity.

Prometheus metrics exposed:

- `jev_http_requests_total{method,path,status}`
- `jev_http_request_duration_seconds{path}`
- `jev_http_requests_in_flight`
- `jev_upstream_requests_total{endpoint,status}`
- `jev_upstream_request_duration_seconds{endpoint}`
- `jev_upstream_tokens_total{direction}`
- `jev_decide_total{kind,strategy}`
- `jev_decide_options`, `jev_decide_pass_options`
- `jev_decisions_total{kind}`

## Behavioral notes and intentional differences

- Upstream vLLM 4xx/5xx responses are surfaced as structured **502**; upstream timeouts as
  **504**; request validation stays **400**. The Python shim propagated the upstream status
  directly.
- `/v1/decide` additionally reports `confidence` (the top probability), as requested for this
  service. The Python shim omits it.
- `/v1/systemone` is not implemented; it is outside this server's contract.
- Image states are rejected client-side with 400 when more than one image is present.
- Optional fitted calibration is applied to raw head logits exactly like the shim. If a fit
  file supplies fewer biases than the served option count (the shipped fit covers 9 choice
  slots, while serving allows 16-256), missing entries are treated as zero rather than
  panicking.
- Response JSON key order and error envelopes are stable but not byte-identical to FastAPI's
  `{"detail": ...}`; signed consumers should verify the signature over whatever body is
  returned.

See `docs/SPEC.md` for the full contract and edge cases.

## License

MIT OR Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.

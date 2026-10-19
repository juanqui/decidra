# jev-engine

Production Rust API/decision server for **`JEV-27B-VL` AWQ W4A16** on one RTX 4090.
It exposes the JEV System 1 decision protocol and the OpenAI Decisions shape, and it
runs inference **in-process** through the custom Rust + CUDA engine in
[`native/`](native/README.md). There is no vLLM, Python, PyTorch, or HTTP upstream on
the runtime path.

The server owns the parts where exactness and operational control matter:

- HTTP/API surface, validation, error semantics, HMAC signing, and metrics.
- Tokenizer-derived wide-choice label table and startup parity checks.
- Prompt construction byte-for-byte compatible with the Python reference.
- Decision-head bias, per-kind temperature, optional fitted calibration, softmax.
- Tournament/permute scheduling, serialized native forwards, and usage accounting.

## Architecture

```text
                    ┌────────────────────── jev-engine (one process) ────────────────────┐
 client ──HTTP────▶ │ axum router                                                        │
                    │  ├─ GET  /health, /v1/models, /v1/decide/info, /metrics             │
                    │  ├─ POST /v1/decide ──▶ DecisionEngine ──▶ Backend trait            │
                    │  │       prompt + label ids + head math        │                   │
                    │  ├─ POST /v1/decisions ─▶ DecisionEngine ──────┤                   │
                    │  ├─ POST /v1/systemone ─▶ DecisionEngine ──────┤                   │
                    │  ├─ POST /v1/chat/completions, /v1/completions ── 501              │
                    │  └─ middleware: HMAC signing, HTTP metrics                         │
                    │                                          │                         │
                    │                                          ▼                         │
                    │                      NativeBackend (tokenizers + jev-native)       │
                    │                      spawn_blocking + mutex (one forward at a time)│
                    └──────────────────────────────────────────┼─────────────────────────┘
                                                               ▼
                                          GPU 1: AWQ W4A16 packed weights resident,
                                          custom CUDA kernels + cuBLAS BF16 GEMM
```

Everything decoded or generated is produced locally. The HTTP server is CPU-side; the
native engine selects GPU 1 through `CUDA_VISIBLE_DEVICES=1` and opens ordinal 0 of the
visible set.

### Backends

- `native` (feature `native`, shipped): in-process `jev_native::Engine`. Loads
  `/srv/models/jev-awq/awq-w4a16-v2`, uploads the packed W4A16 decoder weights once
  (`preload_all`), and keeps them resident.
- `MockBackend` (always compiled): deterministic test backend used by unit and
  integration tests. No GPU or native crate needed.

`cargo test` builds against the mock only. `cargo build --release --features native`
builds the shipped binary.

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
- Prompt bytes and option lines are ported directly from `decide_shim.py`.
- The native engine returns raw logits for the requested head token ids. Raw logits and
  log-probabilities differ by a constant `log Z`; the head softmax normalizer cancels it,
  so the existing bias/temperature/fitted-calibration math is unchanged.

## Repository layout

```text
src/
  main.rs           CLI, tracing, startup, graceful shutdown, --native-check
  api.rs            axum routes, request/response types, decisions mapping, middleware
  backend.rs        Backend trait + MockBackend
  native_backend.rs jev-native + tokenizers adapter (spawn_blocking, serialized forwards)
  engine.rs         single / tournament / permute strategies and usage accounting
  head.rs           decision_head.json + calibration.json loading, bias/temperature/softmax
  prompt.rs         label table, _parts, build_content, tokenizer loading
  pyjson.rs         CPython-compatible JSON/str/truthiness rendering
  rng.rs            CPython-compatible MT19937 and `random.sample` permutations
  config.rs         environment configuration and defaults
  error.rs          backend errors and structured API errors
  telemetry.rs      Prometheus recorder installation
native/             custom CUDA/cuBLAS engine (workspace member)
tests/
  integration.rs    mock-backend end-to-end tests
  native_smoke.rs   one real GPU forward, ignored by default
  fixtures/         tiny tokenizer + real decision-head metadata (no weights)
eval/               independent Python harness, saved baselines, offline mock
scripts/            native smoke, server run, smoke, evaluation
docs/SPEC.md        API contract, math, and edge cases
docs/RESULTS.md     correctness evidence; native benchmarks pending
docs/KERNELS.md     native SM89 kernel stack
docs/DEPLOY.md      one-GPU deployment, ports, health, restart
```

## Build and run

Build prerequisites:

- Stable Rust with edition 2024 support (see `rust-toolchain.toml`).
- CUDA toolkit `nvcc` at build time for the native kernels, or Docker so `native/build.rs`
  can compile them in a container (`JEV_NVCC`, `JEV_CUDA_IMAGE`).

Runtime prerequisites (no toolkit needed):

- NVIDIA driver exposing `libcuda.so.1`.
- `libcublas.so.13` and `libcublasLt.so.13` (CUDA 13). Point `JEV_CUDA_LIBS` at their
  directory if they are not on the loader path.
- RTX 4090 / sm_89.

```bash
cargo test                                    # mock backend; no GPU
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo build --release --features native
```

Validate tokenizer/head/calibration without touching the GPU:

```bash
cargo run --release -- --check-config
```

`--check-config` loads the real tokenizer and decision head, derives the wide-choice
labels with the reference single-token rule, asserts the first 16 ids equal
`decision_head.verbalizer_ids[8:24]`, prints the resolved paths and temperatures, and
exits. It does **not** load the native engine.

Run one real forward on GPU 1, then exit:

```bash
CUDA_VISIBLE_DEVICES=1 \
JEV_CUDA_LIBS=/path/to/cublas \
cargo run --release --features native -- --native-check
```

or use the ignored integration smoke test:

```bash
JEV_CUDA_LIBS=/path/to/cublas ./scripts/native-smoke.sh
```

Run the server:

```bash
CUDA_VISIBLE_DEVICES=1 \
JEV_NATIVE_MODEL=/srv/models/jev-awq/awq-w4a16-v2 \
JEV_BIND_ADDR=0.0.0.0:8080 \
JEV_SIGNING_KEY="$(cat /run/secrets/jev-signing-key)" \
./target/release/jev-engine
```

or `./scripts/run-native.sh` (sets `CUDA_VISIBLE_DEVICES=1`, builds with
`--features native`, default bind `0.0.0.0:8501`).

## Deployment assets

- `scripts/run-native.sh` — build and run the native-backend server (GPU 1).
- `scripts/native-smoke.sh` — one real forward against the AWQ checkpoint.
- `scripts/run-server.sh` — compatibility wrapper around `run-native.sh`.
- `scripts/smoke.sh` — health/models/info plus one real decision.
- `scripts/run-eval.sh` — saved-baseline suite, benchmark, functional and signing checks.
- `Dockerfile` — CUDA devel builder + CUDA runtime image; no weights.
- `docker-compose.yml` — one process, model mounted read-only, GPU 1 reserved.
- `eval/` — independent Python evaluation harness and saved baselines (dev tooling).
- `docs/` — contract, kernel stack, results, deployment.

GPU 0 is not used by this stack. Inside a container, reserve GPU 1
(`--gpus '"device=1"'` or the Compose reservation) and do not set
`CUDA_VISIBLE_DEVICES` yourself: the runtime exposes the selected device as ordinal 0.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `JEV_BIND_ADDR` | `0.0.0.0:8080` | Listen address. |
| `JEV_MODEL` | `jev-awq` | Model id returned by `/v1/models`. |
| `JEV_CHECKPOINT` | `/srv/models/jev-awq/awq-w4a16-v2` | Artifact with tokenizer/head/calibration. |
| `JEV_NATIVE_MODEL` | `$JEV_CHECKPOINT` | Checkpoint opened by the native engine. |
| `JEV_NATIVE_CACHE` | `true` | Keep packed decoder weights resident on the GPU. |
| `JEV_NATIVE_FUSED_W4A16` | `true` | Hybrid fused W4A16 policy: fused when `N >= 5120 && M <= 256`, otherwise dequant + cuBLAS BF16 GEMM. |
| `JEV_TOKENIZER` | checkpoint `tokenizer.json` | Tokenizer file or directory override. |
| `JEV_CALIBRATION` | discovered | `calibration.json` override. |
| `JEV_CALIBRATION_FITTED` | unset | Optional `fit_bias.py` output. Unset/absent disables it. |
| `JEV_HF_SNAPSHOT` | `.../snapshots/ba3f0d58...` | Fallback for artifacts missing from the AWQ dir. |
| `JEV_MAX_CONCURRENCY` | `8` | Max concurrent decision passes (forwards are serialized by the engine). |
| `JEV_SIGNING_KEY` | unset | Enables `x-jev-signature`; empty string disables. |
| `JEV_METRICS_ENABLED` | `true` | Install the Prometheus recorder. |
| `JEV_MAX_BODY_BYTES` | `67108864` | Axum request body limit. |
| `JEV_SERVED_IDENTITY` | `autotrust/JEV-27B-VL@ba3f0d58+awq-w4a16` | `model` returned by `/v1/systemone`. |
| `JEV_CUDA_LIBS` | unset | Directory prepended to `LD_LIBRARY_PATH` for `libcublas.so.13`/`libcublasLt.so.13`. |
| `CUDA_VISIBLE_DEVICES` | `1` in launch scripts | Selects GPU 1 before the native engine opens ordinal 0. |
| `RUST_LOG` | `jev_engine=info` | `tracing-subscriber` filter. |

`JEV_NATIVE_DEVICE` is not used: the native API opens ordinal 0 of the visible set, so
device selection is done before process start with `CUDA_VISIBLE_DEVICES`.

Head/calibration discovery order, relative to the checkpoint then the HF snapshot:

1. `<checkpoint>/decision_head/decision_head.json`
2. `<checkpoint>/adapter_vllm/decision_head.json`
3. `<checkpoint>/decision_head.json`
4. the same names under `JEV_HF_SNAPSHOT`

Calibration follows the same pattern with `decision_head/calibration.json` then
`calibration.json`. `adapter_vllm` is an existing directory name in the artifact, not a
runtime dependency.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness and public model name. |
| `GET` | `/v1/models` | `autotrust/JEV-27B-VL`, `autotrust/JEV-27B`, and the configured id. |
| `GET` | `/v1/decide/info` | Protocol, max options, native 16, temperatures, default strategy. |
| `GET` | `/metrics` | Prometheus text exposition. |
| `POST` | `/v1/decide` | JEV native decision (`noul`, `score`, `choice`). |
| `POST` | `/v1/decisions` | OpenAI Decisions API shape (`predicate`, `choice`, `score`). |
| `POST` | `/v1/systemone` | xPNP hosted System One (`noul`/`choice`, text only). |
| `POST` | `/v1/chat/completions` | **501** — native generation not implemented. |
| `POST` | `/v1/completions` | **501** — native generation not implemented. |

Examples:

```bash
curl -s http://127.0.0.1:8080/v1/decide \
  -H 'content-type: application/json' \
  -d '{"kind":"noul","state":"The user asked for a refund twice.","question":"Is this abusive?"}'

curl -s http://127.0.0.1:8080/v1/decide \
  -H 'content-type: application/json' \
  -d '{"kind":"choice","state":"screenshot text","question":"Pick the best category",
       "options":["billing","abuse","spam"],"strategy":"auto"}'

curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"state":"Customer: my card was charged twice.",
       "questions":{"safe":{"type":"noul","instructions":"Is this abusive?"},
                    "team":{"type":"choice","instructions":"Which team?",
                            "criteria":{"billing":"charges","support":"bugs","sales":"pricing"}}}}'
```

Text decisions are the supported surface. Any state containing an image part is rejected
with **400** (`native vision not implemented`); no image is proxied anywhere. Generation
endpoints return **501** with an OpenAI-shaped `not_implemented_error` envelope.

## Strategies

`strategy` is `auto | single | tournament | permute`.

- `auto` is `tournament`; for choices of 16 or fewer, every strategy degrades to a single
  pass (the reference behavior). `adaptation` reports `native` for those cases.
- `single` runs one read-out over all options (up to 256 ids).
- `tournament` splits options into `ceil(n/16)` near-equal groups, reads each group, keeps
  the winner of each group plus the best remaining options up to 16, runs one final 16-way
  pass, and combines group and final distributions using the reference formula.
- `permute` runs the identity order plus three CPython-compatible shuffles seeded by
  `crc32(question)` and averages the mapped-back probabilities.

Passes are issued concurrently by the async layer but the native engine serializes
forwards through a mutex. A wide choice therefore costs several sequential forwards; the
scheduler still bounds outstanding work with `JEV_MAX_CONCURRENCY`.

See `docs/SPEC.md` for the exact equations.

## Signing

When `JEV_SIGNING_KEY` is non-empty, every response whose path starts with `/v1/decide` or
`/v1/decisions` gets:

```text
x-jev-signature: sha256=<hex HMAC-SHA256 over the exact response body bytes>
```

This includes `/v1/decide/info` and error responses on those paths. `/health`, `/metrics`,
`/v1/models`, and the 501 generation routes are not signed. The signature is calculated
after the body is produced and before it is sent; `content-type` is preserved.

Verify with Python (tooling only, not runtime):

```python
import hashlib, hmac
expected = "sha256=" + hmac.new(key, body, hashlib.sha256).hexdigest()
assert expected == headers["x-jev-signature"]
```

## Tests and evaluation

```bash
cargo test                                    # 31 unit + 7 integration, mock backend
cargo build --release --features native
JEV_CUDA_LIBS=/path/to/cublas ./scripts/native-smoke.sh
```

- Unit tests cover the exact text prompt, image-part parsing/rejection,
  Python-compatible JSON rendering, label derivation, bias/temperature/softmax, fitted
  calibration, CRC32 + MT19937 permutations, and tournament math.
- Integration tests use `MockBackend` and exercise noul/choice/score, tournament/permute
  pass counts, `/v1/decisions`, xPNP System One validation, signing, 400/501/502/504
  semantics, and malformed JSON.
- `native_smoke.rs` is `#[ignore]` and runs one real forward when the checkpoint and GPU
  are available.
- `--check-config` validates tokenizer/head/calibration without a GPU.

To evaluate a running server against the saved production-v2 baseline:

```bash
./scripts/run-eval.sh
```

It runs the saved-baseline comparison (strict and documented-tolerance variants), the
combined benchmark, functional checks (256 choices, `/v1/decisions`), optional signing
(set `JEV_EVAL_SIGNING_KEY`), and writes reports to `eval/out/`. It never starts or stops
GPU services.

### Measured results

Correctness (native engine, from `native/README.md`, validated against CPU golden data):

- AWQ 27B layer outputs: cosine ≥ **0.99995**; final norm **0.9999221**; full logits
  **0.9999969**.
- Top-1 token and last logit identical to golden (`867`, `3.578125`); two-way softmax
  **0.8568** vs golden **0.8559**.
- Merged-BF16 tiny model: every stage at the bf16 rounding floor.

Single-forward native measurements on GPU 1 with resident packed weights
(`native/README.md`):

| prompt | reference path (dequant + cuBLAS) | hybrid fused (`JEV_NATIVE_FUSED_W4A16=true`, default) |
|---|---|---|
| 36 tokens | 174.7 ms (first request 259 ms) | **84.8 ms** (2.06x) |
| 76 tokens | 202.2 ms | **126.4 ms** (1.60x) |
| 1024 tokens | 1261.2 ms | **1261.2 ms** (falls back, 1.00x) |

Preload is **5.8 s** (12.4 GiB packed weights resident); init is 0.2 s. During the
36-token hybrid run the GPU was at 100% utilization, ~184 W mean / 213 W peak. Outputs are
bit-identical to the streaming path, and the golden checks are unchanged.

The hybrid policy uses fused only for `N >= 5120 && M <= 256` and falls back to
dequant + cuBLAS otherwise; that is why 1K prefill sees no fused benefit. The server adds
tokenization, head math, and JSON on top of these forward times. Short live benchmark
(16 unique requests/level, hybrid default): c=1 **8.44 req/s**, p50 **123.7 ms**; c=4
**8.48 req/s**, p50 **459.3 ms**, ~**587 tok/s**. Throughput is flat because forwards are
serialized; c=4 is queueing, not batching.

Live API comparison against the **CPU AWQ golden** (`eval/data/awq-cpu-suite28.json`,
28 rows): top-1 **28/28**, mean KL **3.06e-5**, mean MAE **8.45e-4**, max delta
**1.0345e-2** (`r20`). Acceptance uses `top1 1.0`, `max_kl 0.001`, `mean_mae 0.001`,
`max_delta 0.011` and passes. The historical vLLM baseline gives top-1 28/28, mean KL
8.07e-5, mean MAE 1.64e-3, max delta 1.45e-2; the vLLM result itself differs from CPU
golden by max 0.01377 / mean MAE 0.00156, so it is context only. See `docs/RESULTS.md`
for per-kind tables. Run `scripts/run-eval.sh` to reproduce.

## One-GPU deployment example (GPU 1 only)

The HTTP server is CPU-side; the native engine uses **GPU 1** inside the same process.
Do not touch the production stack's ports (`8001`, `8500`) or its model directory.

```bash
export CUDA_VISIBLE_DEVICES=1
export JEV_NATIVE_MODEL=/srv/models/jev-awq/awq-w4a16-v2
export JEV_CHECKPOINT=/srv/models/jev-awq/awq-w4a16-v2
export JEV_BIND_ADDR=0.0.0.0:8080
export JEV_CUDA_LIBS=/path/to/cublas
export JEV_SIGNING_KEY="$(cat /run/secrets/jev-signing-key)"
exec ./target/release/jev-engine
```

Example systemd unit:

```ini
[Unit]
Description=JEV decision server (native CUDA backend)
After=network-online.target

[Service]
Type=simple
User=jev
WorkingDirectory=/opt/jev-engine
Environment=CUDA_VISIBLE_DEVICES=1
Environment=JEV_NATIVE_MODEL=/srv/models/jev-awq/awq-w4a16-v2
Environment=JEV_CHECKPOINT=/srv/models/jev-awq/awq-w4a16-v2
Environment=JEV_BIND_ADDR=0.0.0.0:8080
# Optional: directory with libcublas.so.13 / libcublasLt.so.13.
#Environment=JEV_CUDA_LIBS=/opt/cuda-libs
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

- The native engine loads all packed decoder weights once at startup (~12.4 GiB, ~5.8 s
  preload) and keeps them resident. First request latency does not include a checkpoint
  upload.
- One forward produces one decision read-out position. Latency is dominated by prefill;
  nothing is decoded. Single-forward measurements: 176.6 ms (36 tokens), 202.9 ms
  (76 tokens), 1277.6 ms (1024 tokens).
- Wide choices multiply forwards: a 256-option tournament makes 16 group passes plus one
  final pass; a permute makes 4 passes. The engine serializes them, so `permute` and
  `tournament` scale roughly linearly with pass count.
- The `single` strategy is the lowest-latency option for ≤16 choices.
- `JEV_NATIVE_FUSED_W4A16=true` is the default hybrid policy: fused for
  `N >= 5120 && M <= 256`, dequant + cuBLAS otherwise. It gives 84.8 ms vs 174.7 ms at
  36 tokens (2.06x) and 126.4 ms at 76 tokens, but **no benefit at 1K tokens** (falls
  back). Small-N projections and large prefills remain on the reference path.
- The fallback path dequantizes each module into a BF16 workspace and uses cuBLAS BF16
  GEMM.

Prometheus metrics exposed:

- `jev_http_requests_total{method,path,status}`
- `jev_http_request_duration_seconds{path}`
- `jev_http_requests_in_flight`
- `jev_backend_requests_total{status}`
- `jev_backend_forward_seconds`
- `jev_backend_prompt_tokens_total`
- `jev_decide_total{kind,strategy}`
- `jev_decide_options`, `jev_decide_pass_options`
- `jev_decisions_total{kind}`
- `jev_systemone_requests_total`, `jev_systemone_questions_total{kind}`

## Behavioral notes and intentional differences

- Native engine failures are surfaced as structured **502**; backend timeouts as **504**;
  request validation stays **400**.
- Image states return **400** (`native vision not implemented`).
- `/v1/chat/completions` and `/v1/completions` return **501**
  (`native generation not implemented`). They never proxy to another process.
- `/v1/decide` additionally reports `confidence` (the top probability), as requested for
  this service. The Python shim omits it.
- `/v1/systemone` is ported from the shim's xPNP handler: one pass per question, no
  tournament/permute, at most 8 questions, `noul`/`choice` only, text only.
- Optional fitted calibration is applied to raw head logits exactly like the shim. If a fit
  file supplies fewer biases than the served option count, missing entries are treated as
  zero rather than panicking.
- Response JSON key order and error envelopes are stable but not byte-identical to
  FastAPI's `{"detail": ...}`; signed consumers should verify the signature over whatever
  body is returned.

See `docs/SPEC.md` for the full contract and edge cases.

## License

MIT OR Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.

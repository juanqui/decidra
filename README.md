# jev-engine

Production Rust API/decision server for **`JEV-27B-VL` AWQ W4A16** on one RTX 4090.
It exposes the JEV System 1 decision protocol and the OpenAI Decisions shape, and it
runs inference **in-process** through the custom Rust + CUDA engine in
[`native/`](native/README.md). There is no external model server, Python, PyTorch, or HTTP upstream on
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
  `/srv/models/jev-awq/awq-w4a16-v2`, repacks/upload the W4A16 decoder weights once
  (`preload_all`), and keeps them resident. Default weight kernel is the repacked Marlin
  path; set `JEV_NATIVE_MARLIN=false` for the dequant/hybrid fallback.
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
- `Dockerfile` — CUDA devel builder + CUDA runtime image; copies cuBLAS/cuBLASLt into the
  runtime, reserves no GPU itself, bakes no weights.
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
| `JEV_NATIVE_MARLIN` | `true` | Repack weights into the Marlin W4A16 layout during preload and use the Marlin kernel. Set `false` for the dequant/fused fallback. |
| `JEV_NATIVE_FUSED_W4A16` | `true` | Fallback-only hybrid fused policy (ignored when Marlin is on): fused when `N >= 5120 && M <= 256`, otherwise dequant + cuBLAS. |
| `JEV_NATIVE_BATCH` | `true` | Enable the native request batcher. Disabled sets one request per forward and window 0. |
| `JEV_NATIVE_MAX_BATCH` | `8` | Maximum requests per batch (`>= 1`). Effective value is `1` when batching is disabled. |
| `JEV_NATIVE_BATCH_WINDOW_US` | `2000` | Microseconds the worker waits after a batch's first request for more work (max `50000`). Effective value is `0` when batching is disabled. |
| `JEV_NATIVE_MAX_BATCH_TOKENS` | `16384` | Maximum total prompt tokens per batch (`>= 512`). A single larger request runs alone. |
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
2. `<checkpoint>/decision_head.json`
3. the same names under `JEV_HF_SNAPSHOT`

Calibration follows the same pattern with `decision_head/calibration.json` then
`calibration.json`.

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
- Long-context CPU AWQ gate (Marlin + tiled FA2 + gated split GDN): final rel_rms
  **0.808% / 0.795% / 0.692% / 0.727%** at 4K/8K/16K/32K, two-way probability delta
  **3.5e-4 / 3.0e-4 / 6.7e-4 / 6.0e-4**, top-1 **true** at every length; **32K is the
  verified maximum** on 24 GB. 10/10 long-context suite cases correct.

Single-forward native measurements on GPU 1 with resident packed weights
(`native/README.md`):

| prompt | reference (dequant + cuBLAS) | hybrid fused fallback | Marlin (default) |
|---|---|---|---|
| 36 tokens | 174.7 ms | 84.8 ms | **45.1 ms** (3.87x) |
| 76 tokens | 202.2 ms | 126.4 ms | **73.0 ms** (2.77x) |
| 1024 tokens | 1261.2 ms | 1261.2 ms | **488 ms** (2.58x) |

Marlin preload is **10.1 s** (repack) and uses **13.1 GiB** resident; the fallback paths
use 5.8 s / 12.7 GiB. Init is 0.2 s. At 36 tokens Marlin runs at 100% GPU utilization.
Long-context latency with Marlin + tiled FA2 + gated split GDN: **4K 2.12-2.18 s**,
**8K 4.23-4.32 s**, **16K 8.62-8.97 s**, **32K (32,763 tokens) 18.60-19.03 s**. Peak
memory is **18.86 GiB at 16K** and **19.09 GiB at 32K** (24 GB card, ~4.4 GiB headroom);
the GPU is at 100% utilization, ~455 W. The split-dk GDN recurrence is 3.9x faster on its
hot kernel (1K per-layer 13.35 -> 3.41 ms) and cuts the 1K forward from 958 to ~488 ms.
It is used only for sequences **>= 512 tokens**: shorter prompts use the accurate shared
GDN because the four-way f32 reduction reorders the state dot products enough to move
rare short-prompt logits (suite `r22`, 54 tokens: split two-way delta 1.43e-2 vs shared
6.43e-3). For **seq >= 24,576** the engine enables a low-memory buffer-pool mode: it
releases dropped large tensors instead of retaining them and drops large MLP/attention/GDN
intermediates as soon as they are consumed. 1K/16K timings are unchanged. Full long
context is **GEMM-bound** at ~0.49-0.55 ms/token. Tiled FA2 remains 7-12x faster than the
warp kernel on identical buffers (kernel-only 1K 0.276 ms, 4K 2.85 ms, 8K 9.68 ms).

With `JEV_NATIVE_MARLIN=false`, the fallback hybrid policy uses fused only for
`N >= 5120 && M <= 256` and falls back to dequant + cuBLAS otherwise. Marlin is the
default and ignores both fallback flags.

Live API comparison against the **CPU AWQ golden** (`eval/data/awq-cpu-suite28.json`,
28 rows, final gated build): top-1 **28/28**, mean KL **3.281e-5**, mean MAE
**7.365e-4**, max delta **1.010e-2** (`r17`, score). The acceptance gates are
`top1 1.0`, `max_kl 0.001`, `mean_mae 0.001`, `max_delta 0.011` — **PASS**. Per-kind
MAE: choice 2.35e-4 / noul 1.237e-3 / score 1.441e-3. The historical production-v2
baseline is context only: it itself differs from CPU golden by max 0.01377 / mean MAE
0.00156. See `docs/RESULTS.md` for per-kind tables.

Short live benchmark, batched build (16 unique ~69-token requests/level; `scripts/bench-batch.sh`):
c=1 **15.83 req/s**, p50 **61.9 ms**; c=2 **23.41 req/s**, p50 **83.1 ms**; c=4
**30.12 req/s**, p50 **133.8 ms**, p95 **139.1 ms**, ~**2,133 tok/s**; c=8 **29.69 req/s**,
p50 **218.7 ms**; c=16 **30.64 req/s**, p50 **378.9 ms**, ~**2,204 tok/s**. Versus the
serialized build, c=4 throughput is **1.79x** and p50 drops from **225.7 ms to 133.8 ms**;
c=1 is unchanged within noise. Above c=4 throughput plateaus near 30 req/s because the
worker runs one batch at a time.

A 16,444-token `/v1/decide` completed in **8.60 s**, and the 32,763-token CPU-AWQ probe
completed in **18.53 s**, choice `true` (0.98378), with no timeout/OOM.

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

- The native engine repacks and loads all decoder weights once at startup (~13.1 GiB,
  ~10.1 s preload with Marlin) and keeps them resident. First request latency does not
  include a checkpoint upload.
- One forward produces one decision read-out position. Latency is dominated by prefill;
  nothing is decoded. Marlin single-forward timings: **45.1 ms** (36 tokens),
  **73.0 ms** (76 tokens), **488 ms** (1024 tokens). Long context is GEMM-bound:
  4K **2.12-2.18 s**, 8K **4.23-4.32 s**, 16K **8.62-8.97 s**, 32K **18.60-19.03 s**
  (~0.49-0.58 ms/token; peak **18.86 GiB at 16K / 19.09 GiB at 32K**). For
  seq >= 24,576 the low-memory buffer-pool mode drops intermediates early; 1K/16K
  timings are unchanged.
- The split-dk GDN recurrence is 3.9x faster than the previous shared-state kernel
  (1K per-layer 13.35 -> 3.41 ms) but is gated to sequences >= 512 tokens; shorter
  prompts keep the accurate shared recurrence. Tiled FA2 attention is 7-12x faster than
  warp on identical buffers (kernel-only 1K 0.276 ms, 4K 2.85 ms, 8K 9.68 ms).
- Requests are batched. A single engine thread owns the CUDA context; the HTTP
  reactor tokenizes and enqueues, then awaits a oneshot reply. The worker blocks for
  the first request and drains more until the **2 ms** window expires, `MAX_BATCH`
  (8) is reached, or the next request would exceed `MAX_BATCH_TOKENS` (16,384).
  Order is strict FIFO and a request is never split. A request larger than the token
  cap (e.g. 32K) runs alone, so long context is unaffected by the 2 ms window.
- Dropped clients are skipped when the batch is formed; executor errors are returned
  per request. A panicking executor fails only its batch.
- Wide choices multiply passes: a 256-option tournament makes 16 group passes plus one
  final pass; a permute makes 4 passes. Passes are independent requests and can share
  a batch when they arrive together.
- The `single` strategy is the lowest-latency option for ≤16 choices.
- Batching only helps when requests arrive together. A lone request still pays the
  2 ms window; set `JEV_NATIVE_BATCH_WINDOW_US=0` or `JEV_NATIVE_BATCH=false` for
  latency-critical single-client deployments.
- `JEV_NATIVE_MARLIN=true` is the default and uses the repacked Marlin W4A16 kernel. At 1K
  the gain is 2.58x (1261 -> 488 ms) after the gated split-GDN work; short prompts keep
  the accurate shared GDN.
- With Marlin off, `JEV_NATIVE_FUSED_W4A16=true` selects the hybrid fused policy (fused for
  `N >= 5120 && M <= 256`), and the remaining path dequantizes into a BF16 workspace for
  cuBLAS BF16 GEMM.
- `JEV_MAX_CONCURRENCY` bounds outstanding decision passes; the batch worker still
  executes one batch at a time.

Prometheus metrics exposed:

- `jev_http_requests_total{method,path,status}`
- `jev_http_request_duration_seconds{path}`
- `jev_http_requests_in_flight`
- `jev_backend_requests_total{status}`
- `jev_backend_forward_seconds` (enqueue to reply, includes batch wait)
- `jev_backend_prompt_tokens_total`
- `jev_native_batch_size` (requests per executed batch)
- `jev_native_batch_queue_depth` (queued, not executing)
- `jev_native_batch_wait_seconds` (enqueue to batch start)
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

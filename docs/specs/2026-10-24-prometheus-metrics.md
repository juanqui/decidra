# Prometheus metrics for the native d3-turbo engine

Status: implemented 2026-10-25; four review passes complete. Date: 2026-10-24.
Scope:
`src/telemetry.rs`, `src/api.rs`, `src/native_backend.rs`, `src/batch.rs`,
`src/main.rs`, `native/src/cuda.rs`, `native/src/lib.rs`, and the operator docs.
The native CUDA kernels are out of scope, except for the small, metrics-neutral
`PhaseTimer` described in §8.6.

The deliverable is a Prometheus-compatible `GET /metrics` endpoint that
describes the server, the single-CUDA-worker scheduler, the exact-state cache,
and the request SLOs well enough to run d3-turbo in production. It is modeled on
vLLM, NVIDIA Flare, SGLang, TGI, llama.cpp, and Triton, but follows Prometheus
naming and aggregation rules where those systems do not.

## 1. Decisions at a glance

| id | decision | why |
|---|---|---|
| D1 | Prometheus text exposition format 0.0.4 over `GET /metrics` | Every target engine is Prometheus-scrape based; 0.0.4 is the lowest common denominator understood by Prometheus 2.x and 3.x. |
| D2 | Namespace `decidra_`, no `:` | Prometheus reserves `:` for recording rules; colons require escaping in many query/OTel paths. vLLM/SGLang/llama.cpp use colons for historical reasons, not as a best practice. |
| D3 | Keep `metrics` + `metrics-exporter-prometheus`; do not hand-render | The dependency and macros already exist. The exporter implements escaping, `_bucket`/`_sum`/`_count`, `le`, `HELP`/`TYPE`, and global labels. A custom renderer (Flare-style) is more code and more format risk. |
| D4 | Histograms, not summaries, with explicit per-family buckets | Summaries cannot be aggregated across instances and are not stable across scrapes. The exporter silently defaults to summaries when no buckets are configured; that is a bug in the current setup. |
| D5 | Global label `model_name`; no per-request identifiers | Matches vLLM/SGLang dashboards and lets one query aggregate across instances. Cardinality stays bounded. |
| D6 | Instrument the server/worker boundary, not the CUDA kernels | The native crate stays metrics-free and reusable. Capture, pass, group, cache, queue, and GPU state are all visible at the worker. Kernel-level profiling remains `DECIDRA_PROFILE`/Nsight. |
| D7 | Install the recorder **before** loading the engine; run upkeep every 5 s | Startup work is observable, and raw histogram samples are drained into bounded distributions between scrapes. `install_recorder()` does not spawn upkeep. |
| D8 | Rename `semel_*` to `decidra_*`; no aliases | `semel` is an internal codename. Aliases double cardinality and make the contract permanent. Dashboards can be updated; a recording rule can bridge an external consumer if one appears. |
| D9 | Metrics disabled => `404`, no recorder, all macros no-op | Operators can remove the endpoint without changing the build; serving is unaffected. A disabled target scrapes as `up=0`; remove the scrape target or silence `DecidraDown` in the same change. |
| D10 | Same listener, unauthenticated, network-restricted | Matches vLLM/Flare/SGLang. The endpoint exposes only aggregate data. A separate listener/auth is a future option, not a v1 requirement. |
| D11 | Metric names, types, labels, units, and buckets are a public contract | Dashboards, alerts, and autoscaling depend on them. Changes need a CHANGELOG note and, after 1.0, a deprecation window. |
| D12 | No self-computed rates (q/s, tokens/s) | Counters plus `rate()` are correct under restarts, scrape gaps, and multiple replicas. Only current-state ratios (cache usage) are gauges. |

## 2. What the current code does, and where it falls short

The current endpoint is a placeholder, not a metric contract:

| location | current behavior | problem |
|---|---|---|
| `src/telemetry.rs:7` | `PrometheusBuilder::new().install_recorder()` | Histograms default to **summaries**; no buckets, no descriptions, no upkeep, no global label. |
| `src/api.rs:61` | `/metrics` returns `handle.render()` as a bare string | Axum emits `text/plain; charset=utf-8`, which is valid but leaves the exposition version implicit; we want an explicit, tested `text/plain; version=0.0.4`. |
| `src/api.rs:148` | `decidra_systemone_requests_total` increments before the backend call | Counts attempts, not outcomes; failures are invisible in the counter. |
| `src/batch.rs:405-617` | `semel_*` counters, summaries, gauges | Mixed prefix, no help, no cache tier/eviction/offload detail, no token/question throughput, no queue depth, no cancellation. |
| `src/batch.rs:504-514` | `semel_queue_wait_seconds` measured after cache restore/capture | It is queue wait **plus** capture, not queue wait. |
| `src/batch.rs:549-552` | Cache gauges only at the end of a successful group | Early failures leave stale gauges; host/GPU tier detail is missing. |
| `src/native_backend.rs:487` | `semel_requests_total` increments only after rows arrive | Failed and overloaded requests are missing from the request counter. |
| `src/main.rs:149-152` | Recorder installed after `load_native` | Any startup/self-test metrics are lost. |
| `docs/ENGINE.md:137` | Documents `semel_state_cache_*` | The docs would become the only record of a stale prefix. |

Additional correctness gaps found by reading the exporter source
(`metrics-exporter-prometheus` 0.18.3):

- `DistributionBuilder::get_distribution()` returns a **summary** unless
  `set_buckets` or `set_buckets_for_metric` matched. Every current
  `semel_capture_seconds`/`semel_pass_seconds`/`semel_group_*` is therefore a
  summary with quantiles, not a histogram.
- `install_recorder()` does not spawn the upkeep task. Between scrapes,
  histograms accumulate raw samples. A 5 s `run_upkeep()` moves them into
  aggregated distributions and bounds memory.
- Counters are rendered as `u64`; `metrics` counters are `f64`. All our
  increments are integral, so this is safe, but fractional increments must not
  be introduced.
- `with_recommended_naming(true)` appends `_total` to counters and unit
  suffixes to metrics that lack them. Our names already carry `_total`,
  `_seconds`, and `_bytes`; enabling it is a guardrail, not a rename.
- `render()` itself drains histograms and commits descriptions, so a scrape
  alone is correct. Upkeep is still needed to bound memory when scrapes are
  slow or absent.

## 3. Prior art

| engine | endpoint / format | naming | metric types | request/queue/cache coverage | labels | lesson |
|---|---|---|---|---|---|---|
| vLLM | `/metrics`, Prometheus text; `prometheus_client`; optional multi-process | `vllm:` (colon) | counters, gauges, histograms with bucket families | running/waiting, KV cache usage, prefix-cache queries/hits, prompt/generation tokens, request success, e2e/TTFT/ITL/TPOT, HTTP metrics | `model_name`, `finished_reason`, `engine` | The reference set for inference SLOs; label every metric with the model. Colon prefix is historical, not a model to copy. |
| NVIDIA Flare | `/metrics`, text 0.0.4 by default; optional JSON; custom renderer | `flare_` | counters, gauges, custom histograms | active/pending, prefill/generated tokens, KV pages, GPU+host prefix cache, speculative/graph counters, HTTP requests | bounded enum labels (`kind`, `reason`, `outcome`) | A Rust engine can render metrics directly with a central snapshot; it avoids recorder overhead but reimplements format/escaping. Docs are the contract: every metric has a help string. |
| SGLang | `/metrics` with `--enable-metrics` | `sglang:` | counters, gauges, histograms | running/queue requests, token usage, cache hit rate, TTFT/e2e/TPOT, GPU/host pools | `model_name`, `is_streaming` | Rich cache/tier metrics matter for hybrid attention. A `cache_hit_rate` gauge is convenient but less aggregatable than counters. |
| TGI | `/metrics` | `tgi_` | counters, gauges, histograms | queue size, batch size, request duration/input, generated tokens, validation/queue durations | `method` | Missing unit suffixes (`tgi_request_duration` in seconds) make dashboards ambiguous. Always encode units. |
| llama.cpp | `/metrics` with `--metrics` | `llamacpp:` | counters, gauges | prompt/predicted tokens, tokens/s, KV-cache usage/tokens, requests processing/deferred | router `model` query | A small set is enough to operate a single-model server; per-slot detail can be deferred. |
| Triton / TensorRT-LLM | `/metrics` on the metrics port; DCGM for GPU | `nv_`, `trtllm_` | counters, histograms, summaries | request success/failure, pending/queue count, request/queue/compute durations (µs), KV cache | per-model/per-version | Counts by outcome and queue duration are essential. Microseconds and summaries are compatibility traps; use seconds and histograms. |

The design borrows:
- vLLM/SGLang: `model_name` label, running/waiting gauges, request outcome
  counters, cache query/hit counters, histogram-based latency.
- Flare: bounded enum labels, explicit help for every series, cache-tier detail,
  and a clear separation between instantaneous gauges and lifetime counters.
- Triton: outcome reasons and queue time as first-class signals.
- TGI/llama.cpp: a compact set that fits one model and one worker; not every
  internal counter is exported.

It deliberately rejects:
- colon-prefixed names (reserved by Prometheus),
- summaries as the primary latency type (non-aggregatable),
- microseconds, percentages 0–100, and unit-less duration names,
- unbounded labels (request IDs, state digests, image hashes, error strings),
- self-computed throughput gauges.

Sources consulted:

- vLLM metrics design: <https://docs.vllm.ai/en/latest/design/metrics/>
- vLLM production metric catalog: <https://docs.vllm.ai/en/stable/usage/metrics/>
- vLLM bucket families: <https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/buckets.py>
- vLLM Prometheus logger: <https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py>
- NVIDIA Flare metrics contract: <https://github.com/NVIDIA/flare-local-inference/blob/main/docs/api.md>
- Flare's Rust metrics renderer: `helios/src/server/metrics.rs` in that repository
- SGLang production metrics: <https://docs.sglang.io/docs/references/production_metrics>
- SGLang collector: <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/observability/metrics_collector.py>
- TGI metrics: <https://huggingface.co/docs/text-generation-inference/en/reference/metrics>
- llama.cpp server metrics: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Triton metrics: <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html>
- Prometheus exposition formats: <https://prometheus.io/docs/instrumenting/exposition_formats/>
- Prometheus naming practices: <https://prometheus.io/docs/practices/naming/>
- `metrics-exporter-prometheus` builder/handle docs: <https://docs.rs/metrics-exporter-prometheus/0.18.1/metrics_exporter_prometheus/struct.PrometheusBuilder.html>

## 4. Requirements

| id | requirement | acceptance |
|---|---|---|
| R1 | Prometheus 2.x/3.x can scrape `/metrics` | `Content-Type: text/plain; version=0.0.4; charset=utf-8`; `# HELP`/`# TYPE`; final LF; `promtool check metrics` clean. |
| R2 | Queue and scheduler state are visible | `decidra_engine_requests_{waiting,running}`, queue capacity, group size, group duration. |
| R3 | Request SLOs are measurable | e2e, queue, tokenize, engine, capture, question-pass histograms; outcomes by class. |
| R4 | Exact-state cache behavior is visible | per-request hit result (GPU/host/miss), entries/bytes by tier, capacity, usage, evictions, offload/restore, reused tokens. |
| R5 | Engine work is countable | state/question tokens, questions answered/failed/cancelled, captures, OOM retries, failures by stage. |
| R6 | Cardinality is bounded | no user-controlled label values; route from `MatchedPath`; status/method/outcome/reason enums; documented series budget. |
| R7 | Metrics do not stall or serialize GPU work | no new stream synchronize on the success path; `PhaseTimer` uses CUDA events read after the pass's existing sync. |
| R8 | Lifecycle is correct | recorder installed before engine load; `run_upkeep` every 5 s; disabled mode installs no recorder and returns 404. |
| R9 | No request content leaks | no state/question text, image bytes, token IDs, digests, or exception strings in names or labels. |
| R10 | The contract is testable and documented | catalog tests, API tests, `docs/METRICS.md`, scrape/alert examples, migration note. |

## 5. Architecture

### 5.1 Components

```
                      one process
┌──────────────────────────────────────────────────────────────────┐
│ axum HTTP reactor                                                 │
│  /health  /v1/models  /v1/systemone  /metrics                    │
│      │          │             │              │                   │
│      │          │   tokenize/prepare         │ render()          │
│      │          │   (blocking pool)           │                   │
│      │          ▼             ▼              ▼                   │
│      │      NativeBackend → BatchHandle → PrometheusHandle       │
│      │                         │                                 │
│      │                         ▼                                 │
│      │                  one CUDA worker thread                   │
│      │                  coalesce → group by state → capture →    │
│      │                  question pass → replies                  │
│      └────────────────── metrics macros ─────────────────────────┘
│                                                                   │
│  global `metrics` recorder (metrics-exporter-prometheus)          │
│  atomic counters/gauges + histogram sample buffers                │
│  upkeep task every 5 s: run_upkeep()                              │
└──────────────────────────────────────────────────────────────────┘
```

There is exactly one recorder, one CUDA worker, and one engine. The HTTP
reactor and the CUDA worker both call `metrics` macros; the recorder is
thread-safe. The `/metrics` handler calls `PrometheusHandle::render()` on the
reactor. The render cost is bounded by the small series set and does not touch
the GPU.

### 5.2 Why not a custom renderer

Flare's custom renderer is appropriate for a codebase that wants zero recorder
dependencies and a JSON snapshot. Here, `metrics` and the Prometheus exporter
are already in `Cargo.toml`, and the existing worker already uses the macros.
Hand-rendering would require reimplementing:

- metric/label escaping (`\\`, `\"`, `\n`),
- `_bucket`, `_sum`, `_count`, `le="+Inf"`, cumulative bucket order,
- `# HELP`/`# TYPE` grouping and uniqueness,
- label-set merging with global labels,
- concurrent scrape/record safety.

That is more risk than value. The exporter's two real gaps—default summaries
and missing upkeep—are fixed with configuration and one task.

### 5.3 Recorder lifecycle

`main::serve` must do, in this order:

1. `Config::from_env()`.
2. If `config.metrics_enabled`: `telemetry::install(&config, &build_info)`.
   This builds the recorder with the `model_name` global label, bucket
   overrides, recommended naming, installs it globally, registers all
   descriptions, initializes known zero series, and sets process/build/config
   gauges.
3. `load_native(config)` (engine open, preload, self-test). The backend records
   `decidra_engine_info`, `decidra_gpu_info`, and the initial GPU memory gauges
   after the engine exists.
4. Build `AppState` with `metrics: Some(handle)`; spawn the 5 s upkeep task.
5. Bind and serve. Abort the upkeep task after `axum::serve` returns.

If installation fails, startup fails. A server that was asked for metrics but
silently serves without them is harder to diagnose than a failed deploy.

`telemetry::install_recorder()` currently calls `install_recorder()` and does
not spawn upkeep. `PrometheusHandle::run_upkeep()` is synchronous, locks the
recorder's distribution map, and drains raw histogram samples. The task is:

```rust
let handle = handle.clone();
tokio::spawn(async move {
    let mut tick = tokio::time::interval(Duration::from_secs(5));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tick.tick().await;
        handle.run_upkeep();
    }
});
```

The handle is cheap to clone. The task is aborted on graceful shutdown. With
18 histograms and the expected traffic, the drain is microseconds. If profiling
shows reactor jitter, move the call to `spawn_blocking`; do not skip it. Wrap
`run_upkeep` in `catch_unwind` so an unexpected panic logs and ends the task
instead of silently killing the last line of memory defense; `render()` still
drains histograms at scrape time.

### 5.4 Concurrency and scrape semantics

- Counters/gauges are atomics in the recorder; a scrape is a snapshot, not a
  transaction. A request can be visible in a counter before its duration
  histogram is rendered if the scrape interleaves; that is normal Prometheus.
- Histograms are drained into cumulative bucket distributions under a lock.
  `run_upkeep` and `render` both drain; each raw sample is consumed once.
- Gauges that are event-driven (queue depth, cache entries) are set at the
  transition that changes them. GPU memory is sampled when `cudaMemGetInfo` is
  already needed for admission, plus once at the end of each group.
- The HTTP in-flight gauge uses an RAII guard so a dropped client future or a
  panic cannot leak it. Queue/running/channel are **not** raw
  increment/decrement gauges: they are backed by shared `AtomicUsize` counters
  and published with `set()` on every transition (§7.4, §8.5). The running
  decrement is driven by a guard that calls `QueueCounters::dec`, which uses a
  saturating `fetch_update`.
- Scrapes do not acquire the CUDA context and do not synchronize the stream.

## 6. Endpoint contract

### 6.1 Request/response

| property | value |
|---|---|
| method | `GET` and `HEAD` (axum's `get` also matches `HEAD`) |
| path | `/metrics`; query parameters are ignored |
| success status | `200 OK` |
| success `Content-Type` | `text/plain; version=0.0.4; charset=utf-8` |
| success `Content-Encoding` | identity; gzip is not implemented |
| success `Cache-Control` | `no-store` |
| body | Prometheus text exposition, UTF-8, LF line endings, final LF |
| disabled status | `404 Not Found`, `text/plain`, `metrics disabled\n` |
| auth | none; same trust boundary as `/v1/systemone` |
| exposure | internal only; do not publish the metrics port to the public internet without a network policy or authenticating proxy |
| timeouts | none beyond the server's normal HTTP behavior |
| OpenMetrics/protobuf | not implemented |

Prometheus 3 requires a valid, supported `Content-Type`. A bare `String`
response yields `text/plain; charset=utf-8`, which is valid but leaves the
format version implicit; set it explicitly so scrapers do not have to guess.
The handler should be:

```rust
async fn metrics_endpoint(State(state): State<AppState>) -> Response {
    let Some(handle) = &state.metrics else {
        return (StatusCode::NOT_FOUND, "metrics disabled\n").into_response();
    };
    let mut body = handle.render();
    if !body.ends_with('\n') {
        body.push('\n');
    }
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        body,
    )
        .into_response()
}
```

Using `HeaderValue::from_static` avoids per-scrape parsing. `render()` returns
a `String`; the expected size is under 200 KiB at the ~600–800 series budget.
The handler caches the rendered body for one second (§8.3); the code above
shows the response shape, not the cache. If a future series set grows past the
budget, switch to `render_to_write` into a reusable buffer.

### 6.2 What is deliberately not exposed

- No raw request path: a scanner must not create a new label value per URL.
  Use axum's `MatchedPath`; unmatched routes are `route="unmatched"`.
- No `/metrics` series in the HTTP metrics: scraping must not feed back into
  request latency/volume. Prometheus's own `up`/`scrape_duration_seconds` cover
  the scrape.
- No request/response bodies, state, questions, image bytes, token IDs, state
  digests, or image hashes.
- No exception messages, model paths, or hostnames as label values. Detailed
  failures stay in `tracing`.
- No `format=json` mode. The Prometheus text endpoint is the contract; a debug
  JSON view would be a second format with its own compatibility surface.

## 7. Metric contract

### 7.1 Naming and type rules

- All names start with `decidra_`.
- Counters end in `_total` and only increase (until process restart).
- Durations are seconds and end in `_seconds`; bytes end in `_bytes`;
  current-state fractions end in `_ratio` and are 0–1.
- Counts of things (requests, questions, tokens, entries) do not get a
  `_count` suffix; `_count` is reserved for the histogram sub-series.
- `_info` metrics are gauges with value `1` and static labels.
- Histograms are used for distributions; summaries are forbidden.
- A metric name refers to one quantity and one unit. Do not mix capacity and
  current usage in one series; export capacity separately.
- Counters are initialized to `0` for every known label combination where an
  error-path series must be visible to alerts. Histograms are not initialized.

### 7.2 Global label

Every metric carries `model_name="<DECIDRA_MODEL>"`. Prometheus adds `job` and
`instance`; do not add them at the exporter. If the server ever serves more
than one model, this becomes a per-metric label; v1 is single-model.

### 7.3 Cardinality budget

| label | values | bounded by |
|---|---|---|
| `model_name` | 1 per process | config |
| `method` | `GET`, `POST`, `HEAD`, `OPTIONS`, `other` | HTTP method mapping |
| `route` | `/v1/systemone`, `/health`, `/v1/models`, `unmatched` | router |
| `status` | the server's status codes (`200`, `400`, `404`, `405`, `500`, `502`, `503`, `504`, `413`) | handler/middleware |
| `outcome` (requests) | `success`, `invalid_request`, `unsupported`, `overloaded`, `timeout`, `backend_error`, `internal_error` | enum |
| `outcome` (groups) | `success`, `failed` | enum |
| `outcome` (questions) | `answered`, `failed`, `cancelled` | enum |
| `result` (cache) | `gpu_hit`, `host_hit`, `miss` | enum |
| `cache` (group) | `gpu_hit`, `host_hit`, `miss`, `unresolved` | enum |
| `tier` | `gpu`, `host` | enum |
| `reason` (eviction) | `entry_limit`, `byte_limit`, `host_budget`, `offload_failed` | enum |
| `reason` (admission) | `queue_full`, `image_busy`, `worker_stopped` | enum |
| `stage` (failure) | `vision_encode`, `state_capture`, `state_restore`, `state_offload`, `question_pass`, `rows`, `positions`, `scheduler`, `headroom` | enum |
| `stage` (cancel) | `queued`, `reply` | enum |
| `phase` | `state`, `question` | enum |
| `device` | `0` | one device |

Expected steady-state series count is roughly **600–800**. The 18 histogram
families produce ~471 non-HTTP series at their steady label sets (including
four `cache` values for `group_seconds` and three for
`question_pass_seconds`), plus ~60 for the common three HTTP route/method
pairs and ~70 counters/gauges/info series. The enumerated HTTP domains admit
5 methods × 4 routes = 20 combinations, so the worst case adds 400 HTTP
histogram series and 180 status counter series to the ~471 non-HTTP
histograms. Plan for a ceiling of about **1,120 series** (the `unresolved`
value is already included in the 471). A scrape should stay under **200 KiB**
and render in under **10 ms** at that size; the alert thresholds and scrape
timeout are sized accordingly.

No label value is derived from a request body, token id, state digest, image
hash, question id, client address, or error message.

### 7.4 Catalog

The following tables are the normative contract. "Zero-init" means the metric
is registered at startup for every listed label combination with value 0.
Where a label combination is not meaningful (`tier="host"` with
`reason="entry_limit"`/`"byte_limit"`/`"offload_failed"`, or `tier="gpu"` with
`reason="host_budget"`), do not create the series; initialize only valid
combinations.

#### Build, process, GPU info

| metric | type | labels | unit | zero-init | help |
|---|---|---|---|---|---|
| `decidra_build_info` | gauge | `version`, `profile`, `compiled_arch`, `native` | — | yes (1) | Build metadata; value is always 1. |
| `decidra_engine_info` | gauge | `attention_mode`, `quantization`, `vision`, `gpu_embed`, `gdn_chunked`, `weights_cached` | — | after engine load (1) | Engine configuration metadata; value is always 1. |
| `decidra_gpu_info` | gauge | `device`, `name`, `arch`, `sms`, `compute_capability` | — | after engine load (1) | GPU identity metadata; value is always 1. |
| `decidra_process_start_time_seconds` | gauge | — | seconds | yes | Unix time at which the process started. |
| `decidra_gpu_memory_free_bytes` | gauge | `device` | bytes | after engine load | Free device memory from `cudaMemGetInfo`, including other processes. |
| `decidra_gpu_memory_total_bytes` | gauge | `device` | bytes | after engine load | Total device memory from `cudaMemGetInfo`. |

`quantization` describes the **configured execution path** derived from the
serving flags, not a provenance claim about the checkpoint artifact. Use this
exact predicate (evaluated in order):

| condition | `quantization` |
|---|---|
| `native_marlin && native_int8 && native_int8_group_scale` | `w4a8_g64` |
| `native_marlin && native_int8 && !native_int8_group_scale` | `w4a8` |
| `native_marlin && !native_int8` | `w4a16` |
| `!native_marlin && native_fused_w4a16` | `w4a16_fused` |
| `!native_marlin && !native_fused_w4a16` | `bf16` (configured fallback path) |

`native_int8_exclude`, `native_int8_group_exclude`, and
`native_int8_all_layers` do not change the label; they select per-module
scope within the same execution path.
`compiled_arch` is the native build's `DECIDRA_COMPILED_ARCH` normalized to
`sm_<major><minor>` (`sm_89`, `sm_90`, `sm_120`; an architecture suffix such as
`sm_120a` is stripped) or `none`. `gpu_info{arch}` uses the same normalized
vocabulary from the runtime compute capability, so the two labels join.
`profile` is `release` or `debug`.
`weights_cached` reflects `DECIDRA_NATIVE_CACHE`; `gdn_chunked` reflects
`DECIDRA_NATIVE_GDN_CHUNKED`; `gpu_embed` reflects `DECIDRA_NATIVE_GPU_EMBED`;
`vision` reflects `engine.supports_vision()`. `name`, `sms`, and
`compute_capability` come from `engine.gpu`; `arch` is derived from the runtime
compute capability as `sm_<major><minor>` so it uses the same vocabulary as
`compiled_arch` (do not use `GpuArch::label()`, which returns a human-readable
`"SM89 (Ada)"`). Failures are logged and the label is `unknown` rather than
failing startup.

#### HTTP

| metric | type | labels | unit | zero-init | help |
|---|---|---|---|---|---|
| `decidra_http_requests_total` | counter | `method`, `route`, `status` | — | no | Completed HTTP responses by method, matched route, and status. |
| `decidra_http_request_duration_seconds` | histogram | `method`, `route` | seconds | — | HTTP request duration from middleware entry to response. |
| `decidra_http_requests_in_flight` | gauge | `route` | — | `/v1/systemone`, `/health`, `/v1/models` = 0 | HTTP requests currently being handled by matched route. |

The metrics middleware is outermost so it sees body-limit 413s and handler
errors. It skips `route == "/metrics"`. `route` comes from
`axum::extract::MatchedPath`; if absent, `route="unmatched"`. `method` maps
known methods and otherwise `"other"`. `status` is the numeric code as a
string. The in-flight gauge uses an RAII guard so a dropped client future
cannot leak it.

#### d3-turbo request

| metric | type | labels | unit | zero-init | help |
|---|---|---|---|---|---|
| `decidra_systemone_requests_total` | counter | `outcome` | — | yes | Completed `POST /v1/systemone` responses by final outcome, including admission and body-limit rejections. |
| `decidra_systemone_request_duration_seconds` | histogram | `outcome` | seconds | — | HTTP middleware-entry to response duration for `POST /v1/systemone`, including admission and body extraction. |
| `decidra_request_queue_seconds` | histogram | — | seconds | — | Time from worker submission to the start of the request's state group. |
| `decidra_request_engine_seconds` | histogram | — | seconds | — | Time from worker submission to reply, including queue wait and GPU work. |
| `decidra_request_tokenize_seconds` | histogram | — | seconds | — | Tokenization, image decode, and request preparation time. |
| `decidra_image_decode_seconds` | histogram | — | seconds | — | Decode time for one input image. |
| `decidra_request_state_tokens` | histogram | — | tokens | — | Tokens in the shared state block of a request. |
| `decidra_request_question_tokens` | histogram | — | tokens | — | Total question-block tokens in a request. |
| `decidra_request_questions` | histogram | — | — | — | Questions in a request. |
| `decidra_request_images` | histogram | — | — | — | Input images in a request. |

`outcome` values: `success`, `invalid_request`, `unsupported`, `overloaded`,
`timeout`, `backend_error`, `internal_error`. A client cancellation during
queue/execution is counted by `decidra_engine_requests_cancelled_total`;
`systemone_requests_total` describes completed handler responses. The error
mapping is in §8.3.

`request_tokenize_seconds` includes image decode (the decode histogram
overlaps with it) and blocking-pool wait. `request_engine_seconds` starts when
the job is submitted to the worker, so it includes queue wait and group
execution but excludes HTTP parsing, tokenization, and response serialization.
`request_queue_seconds` is recorded by the worker for every live job at group
start; it includes the coalescing window.

#### Engine scheduler and groups

| metric | type | labels | unit | zero-init | help |
|---|---|---|---|---|---|
| `decidra_engine_worker_alive` | gauge | — | — | yes (0) | 1 while the CUDA worker thread is running, 0 after it exits or panics. |
| `decidra_engine_requests_waiting` | gauge | — | — | yes (0) | Requests submitted to the CUDA worker but not yet executing in a state group; includes the channel, the coalescing window, and jobs queued behind an earlier group. |
| `decidra_engine_requests_running` | gauge | — | — | yes (0) | Requests whose state group is currently executing in the CUDA worker. |
| `decidra_engine_channel_occupancy` | gauge | — | — | yes (0) | Jobs currently in the bounded worker channel. |
| `decidra_engine_queue_capacity` | gauge | — | — | yes | Channel capacity; the threshold at which `try_send` returns 503. |
| `decidra_engine_groups_total` | counter | `outcome` | — | yes | State groups executed by the CUDA worker, by outcome. |
| `decidra_engine_group_requests` | histogram | — | — | — | Requests per executed state group. |
| `decidra_engine_group_questions` | histogram | — | — | — | Questions per executed state group. |
| `decidra_engine_group_seconds` | histogram | `cache` | seconds | — | Wall-clock duration of one state group from execution start to replies or failure. |
| `decidra_engine_question_pass_seconds` | histogram | `cache` | seconds | — | Batched question-pass duration for one state group. |
| `decidra_engine_state_capture_seconds` | histogram | — | seconds | — | Cold state-capture span, including any OOM trim/retry; measured with CUDA events. |
| `decidra_engine_vision_encode_seconds` | histogram | — | seconds | — | GPU vision-tower encoding time for one state group. |
| `decidra_engine_oom_retries_total` | counter | `stage` | — | yes | OOM trim-and-retry attempts by stage (`state_capture`, `question_pass`). |
| `decidra_engine_failures_total` | counter | `stage` | — | yes | Failed engine operations by stage. |
| `decidra_engine_admission_rejections_total` | counter | `reason` | — | yes | Requests rejected before execution by reason. |
| `decidra_engine_requests_cancelled_total` | counter | `stage` | — | yes | Requests abandoned by the client, by cancellation stage (`queued`, `reply`). |
| `decidra_engine_tokens_total` | counter | `phase` | tokens | yes | Tokens processed by the engine; `state` counts successful cold captures, `question` counts successful question passes. |
| `decidra_engine_questions_total` | counter | `outcome` | — | yes | Questions processed by the engine, by outcome. |
| `decidra_engine_state_captures_total` | counter | — | — | yes | Cold state captures completed successfully. |

Semantics:

- A **request** is one `POST /v1/systemone`; a **question** is one entry in its
  `questions` object; a **group** is one exact-state group the worker executes;
  a **pass** is the question pass for a group, which may split internally at
  the engine's row/key/score caps.
- `engine_worker_alive` is set to 1 **before** the worker thread is spawned and
  to 0 by a `WorkerGuard` on every exit path, including panic. Setting it in the
  spawn path closes the startup window where a scrape could see 0 and fire
  `DecidraWorkerDown`. The guard also resets the queue/running/channel atomics
  so a dead worker does not leave impossible gauges. If the worker dies,
  submissions fail through `Disconnected` and are counted as
  `admission_rejections{reason="worker_stopped"}`.
- Queue/running/channel gauges are backed by `AtomicUsize` counters shared by
  the HTTP threads and the worker. Decrements use `fetch_update(... saturating_sub(1))`
  and every transition writes the metric from the atomic. This prevents
  negative gauges if a worker-reset races a submit rollback. A plain
  `gauge.decrement()` is not sufficient for these three series.
- `engine_requests_waiting` includes jobs in the channel, jobs in the
  coalescing window, and jobs already pulled into `run_jobs` but queued behind
  an earlier state group. It is incremented **before** `try_send` (rolled back
  on failure) and decremented per job when that job's `run_group` starts (or
  when a client-closed job is filtered). Incrementing before the send prevents
  a fast worker from decrementing a job that the HTTP thread has not yet
  counted.
- `engine_channel_occupancy` is the subset currently inside the `sync_channel`.
  It is incremented before `try_send` (rolled back on failure) and decremented
  immediately after `recv`/`recv_timeout` returns the job. This is the correct
  denominator for channel saturation; `engine_queue_capacity` is the channel
  capacity, not a bound on `engine_requests_waiting`.
- `engine_requests_running` is incremented for the live jobs of one group at
  `run_group` start and decremented by an RAII guard when the group finishes.
  Filtered (client-closed) jobs never enter `run_group`; they are cancelled.
- `engine_groups_total{outcome="success"}` counts groups that produced replies.
  A group where every job was filtered before execution is not a group; its
  requests are counted as cancelled.
- `engine_questions_total{outcome="answered"}` counts questions in successful
  replies; `failed` counts all questions in a group that failed; `cancelled`
  counts questions filtered before execution. Questions for requests rejected
  at admission (queue full / worker stopped) never reach the worker and are
  not counted here.
- `engine_tokens_total{phase="state"}` increments by the state length once per
  successful cold capture, not once per request sharing that state.
  `phase="question"` increments by the total question tokens of a successful
  pass. Failed attempts are not counted; OOM retries are counted separately.
- `engine_failures_total` stages: `vision_encode`, `state_capture`,
  `state_restore`, `state_offload`, `question_pass`, `rows`, `positions`,
  `scheduler`, `headroom`. `rows` means the engine returned a row count
  different from the question count; `positions` means the image state's
  mRoPE positions did not match its token count; `scheduler` means the reply
  channel was dropped; `headroom` means the VRAM preflight refused the pass.
- `engine_admission_rejections_total` reasons: `queue_full` (sync channel full),
  `image_busy` (image-decode semaphore wait exceeded its 30 s bound), and
  `worker_stopped` (channel disconnected). A capacity preflight failure is a
  `headroom` engine failure, not an admission rejection. The server's separate
  128-slot HTTP admission gate returns 503 before the engine; that is counted
  by `systemone_requests_total{outcome="overloaded"}`, not here.
- `engine_requests_cancelled_total{stage="queued"}` is for jobs filtered before
  work; `{stage="reply"}` is for successful-pass replies whose receiver was
  closed after the work completed. A send failure while returning a pre-work
  error is not counted here. A cancelled reply still counts its questions as
  answered if the pass succeeded, because the GPU work was done.
- `engine_state_capture_seconds` and `engine_vision_encode_seconds` are GPU
  event timings read after the question pass's existing stream sync (§8.6).
  They are not host submission times. If a group fails before that sync, skip
  the phase observation rather than publishing a partial number.
- `engine_group_seconds{cache="..."}` uses the **post-headroom** cache result:
  `gpu_hit`, `host_hit`, or `miss`. If the first `ensure_headroom` refuses
  before the post-headroom re-lookup, no cache action was attempted; use
  `cache="unresolved"` and do not increment `state_cache_requests_total` for
  that group. `question_pass_seconds` only occurs after resolution, so its
  label domain is the three resolved values. Do not use the pre-headroom `hit`
  boolean; cache pressure can turn a nominal host hit into a cold capture.

#### Exact-state cache

| metric | type | labels | unit | zero-init | help |
|---|---|---|---|---|---|
| `decidra_state_cache_requests_total` | counter | `result` | — | yes | Inference requests by exact-state cache result (`gpu_hit`, `host_hit`, `miss`). |
| `decidra_state_cache_reused_tokens_total` | counter | — | tokens | yes | State tokens reused from cache, counted once per hit request. |
| `decidra_state_cache_entries` | gauge | `tier` | — | yes (0) | Cached exact states by tier. |
| `decidra_state_cache_bytes` | gauge | `tier` | bytes | yes (0) | Estimated bytes held by the exact-state cache by tier. |
| `decidra_state_cache_capacity_entries` | gauge | — | — | yes | Configured maximum GPU state-cache entries. |
| `decidra_state_cache_capacity_bytes` | gauge | `tier` | bytes | yes | Configured state-cache byte budget by tier. |
| `decidra_state_cache_usage_ratio` | gauge | `tier` | ratio | yes (0) | Fraction of configured state-cache capacity in use by tier. |
| `decidra_state_cache_evictions_total` | counter | `tier`, `reason` | — | yes | State-cache evictions by tier and reason. |
| `decidra_state_cache_offloads_total` | counter | — | — | yes | States copied from GPU to host L2. |
| `decidra_state_cache_offload_bytes_total` | counter | — | bytes | yes | State bytes copied from GPU to host L2. |
| `decidra_state_cache_offload_seconds` | histogram | — | seconds | — | GPU-to-host state offload duration. |
| `decidra_state_cache_restores_total` | counter | — | — | yes | States restored from host L2 to GPU. |
| `decidra_state_cache_restore_bytes_total` | counter | — | bytes | yes | State bytes restored from host L2 to GPU. |
| `decidra_state_cache_restore_seconds` | histogram | — | seconds | — | Host-to-GPU state restore duration. |

Semantics:

- `state_cache_requests_total` is per inference request, not per group. If one
  GPU-hit group has four requests, each result label gets four increments.
  `state_cache_reused_tokens_total` adds the state length for each hit request.
  Requests whose group failed before cache resolution increment no cache
  result; they are outside the hit-ratio denominator.
- The cache result is computed **after** the post-`ensure_headroom` re-lookup,
  not from the initial `hit` boolean. `ensure_headroom` can offload other GPU
  states and trim the host L2; it can evict the group's own host entry, in
  which case the group is a cold `miss`, not a `host_hit`. The same value is
  used for the per-request counter, `reused_tokens`, `group_seconds{cache}`,
  and `question_pass_seconds{cache}`.
- `miss` means no cache entry was available, so a cold capture was needed;
  `host_hit` means a host entry was available, so a restore was needed;
  `gpu_hit` means the entry was already resident. A resolved result is
  recorded even if the subsequent restore/capture fails or headroom refuses
  the work; failures are counted in `engine_failures_total` and the
  successful-action counters stay flat.
- `entries`/`bytes` are instantaneous after the last cache mutation. `bytes`
  uses the server's estimate (`state_len * 65,536 + 160 MiB`), the same value
  used for the cache budget; it is not an exact allocator measurement.
- `capacity_entries` is GPU-only. Host capacity is bytes-only.
- `usage_ratio{tier="gpu"}` is
  `max(entries / capacity_entries, bytes / capacity_bytes)` with a zero
  denominator treated as `0`; `tier="host"` is `bytes / capacity_bytes`.
  A single oversized state admitted after evicting the cache can make the GPU
  ratio exceed 1; that is intentional and diagnosable.
- Eviction reasons: `entry_limit` and `byte_limit` for the unpinned GPU cache
  dropping an entry to make room; `host_budget` when the host L2 drops an entry
  to stay within its budget; `offload_failed` when a GPU entry is dropped
  because `StateCache::offload` failed.
- `offloads_total`/`offload_bytes_total` count successful `StateCache::offload`
  calls, whether triggered by admission headroom or by end-of-group overflow.
  `restores_total`/`restore_bytes_total` count successful host restores.
- Failed offloads/restores increment `engine_failures_total` with
  `stage="state_offload"`/`"state_restore"`; they do not increment the success
  counters.
- `offload_seconds` is wall time around `offload`; `restore_seconds` is a GPU
  event timing read after the pass sync, for the same reason as capture.

### 7.5 Histogram buckets

All buckets are upper bounds in the metric's unit, strictly increasing. The
exporter adds `+Inf` automatically. Families:

| family key | metrics | boundaries |
|---|---|---|
| `http` | `decidra_http_request_duration_seconds` | 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300 |
| `request` | `decidra_systemone_request_duration_seconds`, `decidra_request_engine_seconds` | 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300 |
| `queue` | `decidra_request_queue_seconds` | 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30 |
| `tokenize` | `decidra_request_tokenize_seconds` | 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10 |
| `image` | `decidra_image_decode_seconds` | 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10 |
| `phase` | `decidra_engine_state_capture_seconds`, `decidra_engine_question_pass_seconds` | 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120 |
| `group` | `decidra_engine_group_seconds` | 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300 |
| `vision` | `decidra_engine_vision_encode_seconds` | 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30 |
| `offload` | `decidra_state_cache_offload_seconds`, `decidra_state_cache_restore_seconds` | 0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5 |
| `tokens` | `decidra_request_state_tokens`, `decidra_request_question_tokens` | 1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152 |
| `questions` | `decidra_request_questions` | 1, 2, 4, 8, 16, 32, 64 |
| `images` | `decidra_request_images` | 0, 1, 2, 4, 8, 16 |
| `group_requests` | `decidra_engine_group_requests` | 1, 2, 4, 8, 16, 32 |
| `group_questions` | `decidra_engine_group_questions` | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 |

Buckets are configured with `set_buckets_for_metric(Matcher::Full(name), ...)`
for every histogram. Do **not** rely on a global `set_buckets` default: a
future histogram with a count unit would get second-sized buckets. A unit test
must render one sample of every histogram in the table and assert that no
`{quantile=` series appears and that every family has `_bucket`, `_sum`, and
`_count`.

## 8. Instrumentation points

### 8.1 `src/telemetry.rs`

New public surface:

```rust
pub struct BuildInfo {
    pub version: &'static str,
    pub profile: &'static str,       // "release" | "debug"
    pub compiled_arch: &'static str, // normalized: "sm_89" | "sm_90" | "sm_120" | "none"
    pub native: bool,
}

pub struct EngineInfo {
    pub attention_mode: &'static str, // "prefix_bidir"
    pub quantization: &'static str,   // w4a8_g64 | w4a8 | w4a16 | w4a16_fused | bf16
    pub vision: bool,
    pub gpu_embed: bool,
    pub gdn_chunked: bool,
    pub weights_cached: bool,
}

pub struct GpuInfo {
    pub device: &'static str, // "0"
    pub name: String,
    pub arch: String, // "sm_89" | "sm_90" | "sm_120", matching compiled_arch
    pub sms: usize,
    pub compute_capability: String, // "8.9"
}

pub(crate) fn builder(
    config: &Config,
    build: &BuildInfo,
) -> anyhow::Result<PrometheusBuilder>;
pub fn install(config: &Config, build: &BuildInfo) -> anyhow::Result<PrometheusHandle>;
pub fn spawn_upkeep(handle: PrometheusHandle) -> tokio::task::JoinHandle<()>;
pub fn record_engine_info(info: &EngineInfo, gpu: &GpuInfo);
pub fn set_gpu_memory(free: usize, total: usize);
```

`install` calls `builder`, then `install_recorder`. The contract test calls the
same `builder` with `build_recorder`, so the global-label and bucket
configuration cannot drift between production and tests.

`install` builds the recorder:

```rust
let handle = PrometheusBuilder::new()
    .add_global_label("model_name", config.model_id.clone())
    .with_recommended_naming(true)
    .set_buckets_for_metric(Matcher::Full("decidra_http_request_duration_seconds".into()), HTTP)?
    // ... every histogram family ...
    .install_recorder()?;

describe_all();
init_known_series();
set_build_and_process_metrics(build, config);
Ok(handle)
```

Keep the bucket table in one `const` so the builder and the contract test share
it. `describe_all()` calls `describe_counter!`, `describe_gauge!`, and
`describe_histogram!` for every metric in §7.4, with `Unit::Seconds`,
`Unit::Bytes`, and `Unit::Count` where applicable. `init_known_series()`
increments every zero-init counter with `increment(0)` and sets the build,
process-start, capacity, worker-alive, queue/running, channel, and cache
gauges. Engine/GPU info and memory gauges are set later by
`NativeBackend::load`, once the engine exists.

`record_engine_info`/`set_gpu_memory` are called by `NativeBackend::load` after
the engine exists. They are best-effort: `device_name()`/`mem_info()` failures
are logged and the affected gauge is skipped; metrics must never prevent
serving.

### 8.2 `src/main.rs`

Reorder `serve()`:

```rust
let config = Config::from_env()?;
let build = BuildInfo {
    version: env!("CARGO_PKG_VERSION"),
    profile: if cfg!(debug_assertions) { "debug" } else { "release" },
    compiled_arch: compiled_arch(),
    native: cfg!(feature = "native"),
};
let metrics = if config.metrics_enabled {
    Some(telemetry::install(&config, &build)?)
} else {
    None
};
let load_config = config.clone();
let backend = tokio::task::spawn_blocking(move || load_native(load_config))
    .await
    .context("native backend loader task failed")??;
let mut state = AppState::new(config.clone(), backend);
if let Some(handle) = metrics {
    state = state.with_metrics(handle.clone());
    let upkeep = telemetry::spawn_upkeep(handle);
    // serve; then upkeep.abort()
}
```

`compiled_arch()` is `decidra_native::COMPILED_ARCH` normalized to
`sm_<major><minor>` with the `native` feature, otherwise `"none"`. Add
`pub const COMPILED_ARCH: &str = env!("DECIDRA_COMPILED_ARCH");` to
`native/src/lib.rs` and use it from `native/src/cuda.rs` to avoid duplicating
the `env!`. A small `GpuArch::arch_label()` (distinct from the human-readable
`label()`) can provide the normalized string for both build and runtime
labels.

### 8.3 `src/api.rs`

Add `track_http` as the **outermost** middleware, outside `admit_request` and
`DefaultBodyLimit`, so it sees the 503 from admission and the 413 from the body
limit. Add one shared `track_http` for HTTP metrics and for the
`systemone_requests_total` outcome counter:

```rust
#[derive(Clone, Copy)]
enum Outcome { Success, InvalidRequest, Unsupported, Overloaded, Timeout, BackendError, InternalError }

impl Outcome {
    fn as_str(self) -> &'static str { /* ... */ }
    fn from_status(status: StatusCode) -> Self {
        match status {
            StatusCode::OK => Self::Success,
            StatusCode::BAD_REQUEST | StatusCode::PAYLOAD_TOO_LARGE => Self::InvalidRequest,
            StatusCode::SERVICE_UNAVAILABLE => Self::Overloaded,
            StatusCode::GATEWAY_TIMEOUT => Self::Timeout,
            StatusCode::BAD_GATEWAY => Self::BackendError,
            other if other.is_server_error() => Self::InternalError,
            _ => Self::InternalError,
        }
    }
}

async fn track_http(request: Request, next: Next) -> Response {
    let method = method_label(request.method());
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_owned())
        .unwrap_or_else(|| "unmatched".to_owned());
    if route == "/metrics" {
        return next.run(request).await;
    }
    let _in_flight = InFlightGuard::new(route.clone());
    let started = Instant::now();
    let response = next.run(request).await;
    let status = response.status().as_u16().to_string();
    metrics::counter!(
        "decidra_http_requests_total",
        "method" => method.clone(),
        "route" => route.clone(),
        "status" => status
    )
    .increment(1);
    metrics::histogram!(
        "decidra_http_request_duration_seconds",
        "method" => method.clone(),
        "route" => route.clone()
    )
    .record(started.elapsed().as_secs_f64());
    if route == "/v1/systemone" && method == "POST" {
        let outcome = response
            .extensions()
            .get::<Outcome>()
            .copied()
            .unwrap_or_else(|| Outcome::from_status(response.status()));
        metrics::counter!(
            "decidra_systemone_requests_total",
            "outcome" => outcome.as_str()
        )
        .increment(1);
        metrics::histogram!(
            "decidra_systemone_request_duration_seconds",
            "outcome" => outcome.as_str()
        )
        .record(started.elapsed().as_secs_f64());
    }
    response
}
```

This is the only writer of `systemone_requests_total` and
`systemone_request_duration_seconds`. It counts handler responses and
pre-handler 503/413 rejections exactly once. The handler attaches the semantic
`Outcome` to the response.

Layer `track_http` last so it is outermost:

```rust
Router::new()
    .route("/health", get(health))
    .route("/v1/models", get(models))
    .route("/v1/systemone", post(systemone))
    .route("/metrics", get(metrics_endpoint))
    .layer(DefaultBodyLimit::max(body_limit))
    .layer(middleware::from_fn_with_state(state.clone(), admit_request))
    .layer(middleware::from_fn(track_http))
    .with_state(state)
```

`admit_request` must bypass the 128-permit request-slot semaphore for
`/metrics`; otherwise a saturated API makes the scrape itself return 503 and
`up=0` exactly when the metrics are needed. `MatchedPath` is available in
`Router::layer` middleware:

```rust
if request
    .extensions()
    .get::<MatchedPath>()
    .is_some_and(|path| path.as_str() == "/metrics")
{
    return next.run(request).await;
}
```

The metrics handler also caches its rendered body for one second so a scraper
loop or several Prometheus replicas cannot multiply registry work:

```rust
struct CachedMetrics { at: Instant, body: String }

async fn metrics_endpoint(State(state): State<AppState>) -> Response {
    let Some(handle) = &state.metrics else { /* 404 */ };
    let mut cache = state
        .metrics_cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(cached) = cache.as_ref() {
        if cached.at.elapsed() < Duration::from_secs(1) {
            return metrics_response(cached.body.clone());
        }
    }
    let mut body = handle.render();
    if !body.ends_with('\n') { body.push('\n'); }
    *cache = Some(CachedMetrics { at: Instant::now(), body: body.clone() });
    metrics_response(body)
}
```

`AppState` gains `metrics_cache: Arc<std::sync::Mutex<Option<CachedMetrics>>>`,
initialized to `None`. `render()` still drains histograms; returning a cached
body for one second does not lose samples because the next render sees the
cumulative distributions. A one-second staleness is below any normal scrape
interval.

The handler then attaches the semantic outcome:

```rust
async fn systemone(State(state): State<AppState>, body: Bytes) -> Response {
    let (mut response, outcome) = match parse_systemone(&body) {
        Ok((state_value, questions, images)) => {
            match state.backend.d3_answers(&state_value, questions, images).await {
                Ok(value) => (success_response(&state.config, value), Outcome::Success),
                Err(error) => error_response(error),
            }
        }
        Err(message) => (error(StatusCode::BAD_REQUEST, message), Outcome::InvalidRequest),
    };
    response.extensions_mut().insert(outcome);
    response
}
```

`parse_systemone` returns `Result<(Value, Map<String, Value>, Vec<String>), String>`
and preserves the current validation messages. `error_response` maps
`EngineError` to `(Response, Outcome)`:

| error | status | outcome |
|---|---|---|
| parse/validation failure | 400 | `invalid_request` |
| `Config` | 400 | `invalid_request` |
| `Unsupported` | 400 | `unsupported` |
| `Overloaded` | 503 | `overloaded` |
| `Timeout` | 504 | `timeout` |
| `Backend` with memory/out-of-capacity text | 503 | `overloaded` |
| `Backend` other | 502 | `backend_error` |
| `BadResponse` | 502 | `backend_error` |
| `Internal` | 500 | `internal_error` |

Keep the detailed messages in the response/logs; only the enum goes into
labels. The `Backend` memory/out-of-capacity split currently matches strings;
prefer a typed capacity error if `EngineError` is extended, and map it to
`overloaded`.

### 8.4 `src/native_backend.rs`

Instrument `answers` without duplicating the handler outcome counter:

- Record `decidra_request_images` after image-count validation (even if a
  later decode fails).
- Inside the blocking closure, time each `decode_data_url` call and record
  `decidra_image_decode_seconds` on success or failure.
- Record `decidra_request_tokenize_seconds` around the whole
  `spawn_blocking(...).await` result, before propagating an error.
- After `prepare_request` succeeds, record `decidra_request_state_tokens`,
  `decidra_request_question_tokens`, and `decidra_request_questions`.
- Record `decidra_request_engine_seconds` around the submit-to-reply span
  (`Instant` before `submit_state_once`, record after `receiver.await` on both
  success and error). This is user-visible engine latency and includes queue
  wait.
- If `receiver.await` returns `Err` (the worker dropped the reply), increment
  `decidra_engine_failures_total{stage="scheduler"}` before mapping the error
  to `EngineError::Internal`.
- Do not record `decidra_systemone_requests_total` here; the HTTP layer owns
  final outcomes.

`prepare_request` already computes `state_token_count`, `question_token_count`,
and `block_count`; use those values rather than re-walking blocks.

### 8.5 `src/batch.rs`

Bundle worker limits to keep the instrumentation readable:

```rust
#[derive(Clone, Copy)]
struct CacheLimits { entries: usize, bytes: usize, host_bytes: usize }

#[derive(Clone, Copy)]
struct WorkerConfig {
    max_queue: usize,
    batch_requests: usize,
    batch_rows: usize,
    batch_window_us: u64,
    cache: CacheLimits,
}
```

Capacity gauges (`decidra_engine_queue_capacity`,
`decidra_state_cache_capacity_entries`, `decidra_state_cache_capacity_bytes`)
are set once by `telemetry::install` from `Config`; `BatchHandle` only updates
dynamic gauges. This avoids two writers for the same series and lets the
capacity exist before the engine loads.

`BatchHandle::spawn` sets `decidra_engine_worker_alive=1` before spawning the
thread and starts the worker with a `WorkerGuard` constructed before any
fallible setup; on `Drop`, the guard sets it to 0 and resets the shared
`QueueCounters` atomics for waiting/running/channel.
This makes a worker panic or unexpected exit visible instead of leaving
impossible gauges. `QueueCounters::dec` uses a saturating `fetch_update` and
writes the gauge after every transition, so a reset racing a submit rollback
cannot produce a negative value.

Use `refresh_cache_gauges` after every cache mutation:

```rust
fn refresh_cache_gauges(cache: &[CachedState], host: &[HostCachedState], limits: CacheLimits) {
    let gpu_bytes: usize = cache.iter().map(|e| e.bytes).sum();
    let host_bytes: usize = host.iter().map(|e| e.bytes).sum();
    metrics::gauge!("decidra_state_cache_entries", "tier" => "gpu").set(cache.len() as f64);
    metrics::gauge!("decidra_state_cache_entries", "tier" => "host").set(host.len() as f64);
    metrics::gauge!("decidra_state_cache_bytes", "tier" => "gpu").set(gpu_bytes as f64);
    metrics::gauge!("decidra_state_cache_bytes", "tier" => "host").set(host_bytes as f64);
    let gpu_ratio = ratio(cache.len(), limits.entries).max(ratio(gpu_bytes, limits.bytes));
    metrics::gauge!("decidra_state_cache_usage_ratio", "tier" => "gpu").set(gpu_ratio);
    metrics::gauge!("decidra_state_cache_usage_ratio", "tier" => "host").set(ratio(host_bytes, limits.host_bytes));
}
```

Queue/running accounting:

- `submit_state_once`: increment `engine_requests_waiting` and
  `engine_channel_occupancy` **before** `try_send`. If it returns `Full`,
  decrement both and increment
  `engine_admission_rejections_total{reason="queue_full"}`. If it returns
  `Disconnected`, decrement both and increment
  `{reason="worker_stopped"}`. On `Ok`, leave both set; the worker owns the
  decrements. Incrementing first prevents a worker dequeue from racing ahead
  of the HTTP thread and making the gauge temporarily negative.
- Worker `recv`/`recv_timeout`: decrement `engine_channel_occupancy`
  immediately for each returned job. Do **not** decrement
  `engine_requests_waiting` here; a pulled job may still wait behind an earlier
  state group.
- `run_jobs` entry: filter closed replies. For each filtered job decrement
  `engine_requests_waiting` and increment
  `engine_requests_cancelled_total{stage="queued"}` plus
  `engine_questions_total{outcome="cancelled"}` by its question count. Live
  jobs stay `waiting` while they are grouped and while earlier groups run.
- `run_group` start: for the group's live jobs, decrement
  `engine_requests_waiting` and increment `engine_requests_running`, then hold
  a `RunningGuard` that decrements running when the group finishes (all return
  paths, including panics). Record `decidra_request_queue_seconds` for each
  live job as `group_started - job.submitted`; record `group_requests` and
  `group_questions` once for the group; determine `cache` label.
- Every `run_group` return path funnels through a `finish_group` helper or
  guard that records `engine_group_seconds{cache}`, increments
  `engine_groups_total{outcome}`, and calls `refresh_cache_gauges`. Early
  returns must not skip these series. On the success path, the end-of-group
  cache insert/offload must run before the final gauge refresh (or refresh
  once more after it), so the last admission is visible.
- Cache result: `cache_result` is `Option<CacheResult>`. Resolve it from
  `cached`/`host_cached` **after** `ensure_headroom` and its host re-lookup.
  If the first `ensure_headroom` returns an error before that point, leave it
  `None`; use `cache="unresolved"` for the group histogram and do not touch
  the per-request cache counters. Once resolved, use the value for the rest of
  the group: for each live job increment `state_cache_requests_total{result}`
  and, for hits, add `state.len()` to `state_cache_reused_tokens_total`.
  Record `state_cache_restores_total`/bytes once on a successful host restore,
  or `engine_state_captures_total`/`engine_tokens_total{phase="state"}` once on
  a successful cold capture. Never count a host restore for a group that
  fell back to a cold capture after headroom pressure.
- Timers around vision encode, restore, and capture use `PhaseTimer`; `stop()`
  after the operation; read `seconds()` after the question pass returns (or
  skip on failure). Record `state_cache_restore_seconds`,
  `engine_state_capture_seconds`, `engine_vision_encode_seconds`.
- Question pass: record `engine_question_pass_seconds{cache}` on success.
  Record `engine_tokens_total{phase="question"}` by total question tokens and
  `engine_questions_total{outcome="answered"}` by question count. On failure,
  record `engine_questions_total{outcome="failed"}`, the appropriate
  `engine_failures_total{stage}`, and still record `engine_group_seconds` /
  `engine_groups_total{outcome="failed"}`.
- OOM retry branches increment `engine_oom_retries_total{stage}`.
- Reply send: if `job.reply.send(...)` fails on the success-path reply (rows
  produced and length-checked), increment
  `engine_requests_cancelled_total{stage="reply"}` for that request. A send
  failure while returning a pre-work error is not a reply-stage cancellation;
  the engine failure counter already classifies the underlying error.
- At group end: record `engine_group_seconds{cache}` and
  `engine_groups_total{outcome}` at reply time. Perform the end-of-group cache
  insert/offload **before** the `finish_group` refresh, or call
  `refresh_cache_gauges` again after the insert; otherwise the last admitted
  state is missing from `entries`/`bytes`/`usage_ratio` until the next group.
  Then sample `engine.gpu.mem_info()` once and set
  `decidra_gpu_memory_free_bytes` (total is set at startup); ignore/log
  failures.
- `ensure_headroom`: set `decidra_gpu_memory_free_bytes` after every
  `mem_info` call; increment `engine_failures_total{stage="headroom"}` on
  refusal; count evictions/offloads where they happen.
- `host_insert`: time `offload`; on success increment `state_cache_offloads_total`
  and bytes, record `state_cache_offload_seconds`; on failure increment
  `engine_failures_total{stage="state_offload"}` and
  `state_cache_evictions_total{tier="gpu",reason="offload_failed"}`. When
  trimming host entries for budget, increment
  `state_cache_evictions_total{tier="host",reason="host_budget"}` per entry.
- The unpinned GPU eviction loop increments
  `state_cache_evictions_total{tier="gpu",reason="entry_limit"|"byte_limit"}`.
  If the cache is full and `pin_cache` keeps the existing entries, the new
  entry is simply not cached; that is not an eviction.

The current `capture_ms` and `wait_max_ms` variables are removed or redefined:
`capture_ms` was host submission time, and `wait_max_ms` included capture. The
new metrics separate queue wait, capture (GPU event time), and pass.

### 8.6 `PhaseTimer` in the native crate

`capture_state` and `vision_encode` return after enqueuing kernels; the stream
is not synchronized until `readout_state_once_batch` copies the readout rows to
the host. Measuring them with `Instant` would publish host enqueue time, not
GPU time. `StateCache::restore` is in the same position. Do not add a stream
synchronize on the success path: it would remove host enqueue overlap.

Add a small timing utility to the native crate:

```rust
// native/src/cuda.rs
pub struct PhaseTimer { start: CudaEvent, end: CudaEvent }

impl Gpu {
    /// Create timing-enabled events and record the start event on the stream.
    /// `new_event(None)` defaults to CU_EVENT_DISABLE_TIMING, so pass
    /// `Some(CUevent_flags::CU_EVENT_DEFAULT)`.
    pub fn phase_timer(&self) -> Result<PhaseTimer> { /* ... */ }
}

impl PhaseTimer {
    /// Record the end event after the operation's kernels/copies are enqueued.
    pub fn stop(&mut self, gpu: &Gpu) -> Result<()> { /* end.record(stream) */ }

    /// Valid only after the stream has synchronized past the end event.
    /// `elapsed_ms` synchronizes the two events; call this only after the
    /// question-pass sync, and log an error instead of silently dropping it.
    pub fn seconds(&self) -> Result<f64> { /* start.elapsed_ms(&end) / 1000 */ }
}
```

The worker creates a timer before `vision_encode`/`restore`/`capture`, calls
`stop()` after the call returns, and reads `seconds()` only after
`readout_state_once_batch` returns successfully (its final pageable
`clone_dtoh` is the point where the stream is known to be drained; verify that
assumption once on the target driver). If the group fails before that point,
skip the phase observation. A failed timer create/stop/read is logged and the
phase metric is skipped; it must never fail the inference request. Event
creation is two calls per cold/restore/vision, not per request; the warm path
is unaffected.

`StateCache::offload` uses `clone_dtoh` to pageable `Vec`s. On current cudarc
this is synchronous in practice for pageable memory, so `offload_seconds` may
be measured with `Instant`. If the driver is observed to complete the copy
asynchronously, switch that one metric to a `PhaseTimer` and record it after
the next stream sync; do not add a per-offload synchronize to the hot path
without measuring.

This is the only native change required by the metrics work. It has no
`metrics` dependency and is useful for any future profiling.

## 9. Semantics cheat sheet

| question | answer |
|---|---|
| What is a request? | One `POST /v1/systemone`, 1–64 questions. |
| What is a group? | One exact-state set executed by the single CUDA worker; one or more coalesced requests sharing state. |
| What is a pass? | The question forward for a group; may split into chunks internally at row/key/score caps. |
| Queue time vs engine time vs e2e | Queue: submit → group start. Engine: submit → reply. E2E: handler entry → response. |
| Why can `request_engine_seconds` exceed `engine_group_seconds`? | Request engine time includes the wait before the request's group started; group time starts when that group starts. Use request metrics for user latency and group metrics for worker cost. |
| How is q/s computed? | `sum(rate(decidra_engine_questions_total{outcome="answered"}[1m]))`. |
| How is cache hit ratio computed? | `sum(rate(decidra_state_cache_requests_total{result!="miss"}[5m])) / sum(rate(decidra_state_cache_requests_total[5m]))`. |
| Why are state tokens lower than the sum of request state tokens? | A cold state is captured once but may serve many coalesced requests; cache hits do not reprocess state tokens. |
| Why no generated tokens? | d3-turbo is a readout classifier; it produces one answer per question, no generation loop. |
| What does `state_cache_usage_ratio` mean? | Fraction of the configured budget, not of physical GPU memory. Use `gpu_memory_free_bytes` for physical headroom. |
| Why are counters zero-init? | Error-path series must exist before the first error so alerts and dashboards do not treat absence as zero. |
| Why do counters reset on restart? | Prometheus counters are process-lifetime. Use `rate()`/`increase()`; `process_start_time_seconds` detects a restart. |

## 10. Performance and overhead

| item | budget | method |
|---|---|---|
| Per-request metrics overhead | < 1% of `POST /v1/systemone` p50 | A/B two processes, metrics on/off, same checkpoint, 64q c1 and 16q c8; 20 runs each. |
| Per-group worker overhead | < 0.5 ms | tens of atomic metric calls per group plus one `mem_info`; measure the group duration distribution with/without. |
| Scrape cost | < 10 ms, < 200 KiB | `curl -w`/`promtool` at the ~600–800 expected series and the ~1,120 worst case, idle and under load. |
| Raw histogram memory | bounded by 5 s of samples | `run_upkeep` every 5 s; no unbounded queue. |
| GPU interference | no new stream synchronize on success | `PhaseTimer` events read after the pass's existing sync. |
| Card | no metrics-driven GPU allocation | only CUDA events for cold/restore/vision. |

If the A/B shows a regression above budget, the first fallback is to reduce
histogram observations (for example, drop per-image decode timing) or move
`render()`/`run_upkeep()` to `spawn_blocking`. Do not remove the queue/cache
metrics; they are the operational core.

## 11. Testing

### 11.1 Unit tests

`telemetry` tests use a local recorder so they do not contend for the global
recorder:

```rust
let recorder = builder(config, &build).build_recorder();
let handle = recorder.handle();
metrics::with_local_recorder(&recorder, || {
    describe_all();
    init_known_series();
    // emit one sample per catalog metric
});
let text = handle.render();
```

Assertions:

1. `Content-Type` is set by the endpoint test, not by `render()`.
2. Every catalog metric has a `# HELP` line and a `# TYPE` line.
3. No rendered line contains `{quantile=`.
4. Every histogram family has `_bucket`, `_sum`, `_count`, and a `le="+Inf"`
   bucket equal to `_count`.
5. Counter names end in `_total`; no metric name contains `:`.
6. Labels escape `"`/`\`/newline in `model_name` and GPU name.
7. Bucket lists are strictly increasing and non-negative (`decidra_request_images`
   legitimately starts at 0).
8. Zero-init creates the enumerated error series at value 0.
9. `usage_ratio` is 0 with zero capacity and `max(...)` otherwise.
10. Disabled mode returns 404 and does not panic.

### 11.2 API integration tests

Add `tower = { version = "0.5", features = ["util"] }` under
`[dev-dependencies]` (it is currently only transitive through axum) and use
`tower::ServiceExt::oneshot` with `AppState` and `MockBackend`. `MockBackend`
does not run `NativeBackend`, so it cannot produce backend-only series:

- POST a valid body; assert 200, then scrape and assert
  `decidra_systemone_requests_total{outcome="success"}`,
  `decidra_http_requests_total{route="/v1/systemone",status="200"}`, and
  `decidra_systemone_request_duration_seconds_count`.
- Do **not** assert `decidra_request_questions`,
  `decidra_request_state_tokens`, `decidra_request_engine_seconds`,
  `decidra_image_decode_seconds`, or `decidra_request_tokenize_seconds` in
  this test; those are recorded by `NativeBackend::answers`. Leave them to the
  GPU smoke tests in §11.3, or add a dedicated instrumentation-helper test
  that calls the recording function directly.
- POST invalid JSON and a non-object; assert 400 and
  `outcome="invalid_request"`.
- Use a test backend that returns `Overloaded`, `Timeout`, `Backend`, and
  `Internal`; assert status/outcome mappings.
- Exhaust the 128 request slots (or call `admit_request` directly) and assert a
  `/v1/systemone` POST is 503 while `GET /metrics` still returns 200. The 503
  must increment `systemone_requests_total{outcome="overloaded"}` even though
  the handler never ran.
- Send a body over `DECIDRA_MAX_BODY_BYTES`; assert 413 and
  `systemone_requests_total{outcome="invalid_request"}`.
- Scrape twice within one second; assert the bodies are identical (cache) and
  that counters are monotonic and `_count` is cumulative.
- `/metrics` is absent from `decidra_http_requests_total`.
- Disabled state: `/metrics` 404, no recorder needed.
- Parallel scrapes while requests run; no panic, no malformed output.

Because the global recorder can be installed only once, use a single
`#[tokio::test]` for the end-to-end metrics contract or a `OnceLock` plus
delta assertions. Do not install a second global recorder in another test.

### 11.3 Worker tests

`run_jobs` needs an `Engine`, so full queue/cache tests run on the GPU smoke
path. Factor pure helpers (`ratio`, label mapping, `refresh_cache_gauges`,
outcome mapping) into testable functions and test them with a local recorder.
The GPU tests assert:

- one request: waiting/running/channel occupancy return to 0 and
  `engine_worker_alive` is 1;
- dropping the worker (shutdown or forced panic) leaves `engine_worker_alive=0`
  and waiting/running/channel occupancy at 0;
- request-level `decidra_request_state_tokens`, `decidra_request_question_tokens`,
  `decidra_request_questions`, `decidra_request_tokenize_seconds`, and
  `decidra_request_engine_seconds` are present after a real request;
- two coalesced requests with the same state: one capture, `group_requests=2`,
  `state_cache_requests_total{result="miss"}=2`, next group
  `{result="gpu_hit"}`;
- cold vs host restore vs GPU hit labels, including a host-hit group that is
  forced into a cold capture by headroom pressure (assert `result="miss"` and
  `state_cache_restores_total` unchanged);
- queue full returns 503 and increments `admission_rejections{queue_full}`;
- a dropped reply after a successful pass increments
  `cancelled{stage="reply"}`, not `cancelled{stage="queued"}`;
- OOM retry paths increment the right stage (fault injection is hard; at
  minimum review the branches and exercise the ordinary path).

### 11.4 Smoke and acceptance

1. `cargo test --locked -p decidra`.
2. `cargo check --features native`.
3. GPU: start with a checkpoint, run one 2-question request, `curl /metrics`;
   verify `promtool check metrics`.
4. Run the existing 64q benchmark with metrics on/off; record p50/p95.
5. Scrape while a 64q request is running; verify the scrape returns in <10 ms.
6. Restart and verify counters reset, gauges reinitialize, no duplicate
   `HELP`/`TYPE`.
7. `curl -I`/`HEAD /metrics`; verify headers and empty body.
8. `DECIDRA_METRICS_ENABLED=false`; verify 404 and no metrics overhead.

Acceptance is §4 R1–R10 plus the performance budget in §10.

## 12. Operations

### 12.1 Scrape configuration

```yaml
scrape_configs:
  - job_name: decidra
    metrics_path: /metrics
    scrape_interval: 15s
    scrape_timeout: 5s
    static_configs:
      - targets: ["decidra:8080"]
```

Kubernetes pod annotations if not using a ServiceMonitor:

```yaml
prometheus.io/scrape: "true"
prometheus.io/port: "8080"
prometheus.io/path: "/metrics"
```

### 12.2 PromQL

```promql
# Questions per second answered
sum(rate(decidra_engine_questions_total{outcome="answered"}[1m]))

# Successful requests per second
sum(rate(decidra_systemone_requests_total{outcome="success"}[1m]))

# Engine tokens per second by phase
sum by (phase) (rate(decidra_engine_tokens_total[1m]))

# p95 end-to-end request duration on the success path (the SLO)
histogram_quantile(
  0.95,
  sum by (le) (rate(decidra_systemone_request_duration_seconds_bucket{outcome="success"}[5m]))
)

# p95 queue wait and pass time
histogram_quantile(0.95, sum by (le) (rate(decidra_request_queue_seconds_bucket[5m])))
histogram_quantile(0.95, sum by (le, cache) (rate(decidra_engine_question_pass_seconds_bucket[5m])))

# Error ratio
sum(rate(decidra_systemone_requests_total{outcome!="success"}[5m]))
  / sum(rate(decidra_systemone_requests_total[5m]))

# State-cache hit ratio (requests)
sum(rate(decidra_state_cache_requests_total{result=~"gpu_hit|host_hit"}[5m]))
  / sum(rate(decidra_state_cache_requests_total[5m]))

# GPU state-cache pressure
decidra_state_cache_usage_ratio{tier="gpu"}

# Channel saturation (the only ratio that predicts a 503)
decidra_engine_channel_occupancy / decidra_engine_queue_capacity

# Worker busy fraction (state groups plus post-reply cache offload)
sum(rate(decidra_engine_group_seconds_sum[1m]))
  + (sum(rate(decidra_state_cache_offload_seconds_sum[1m])) or vector(0))

# GPU memory headroom
decidra_gpu_memory_free_bytes / decidra_gpu_memory_total_bytes
```

### 12.3 Alerts

| alert | expression | for | why |
|---|---|---|---|
| `DecidraDown` | `up{job="decidra"} == 0` | 1m | scrape failure/process down |
| `DecidraWorkerDown` | `decidra_engine_worker_alive == 0` | 1m | CUDA worker exited or panicked; all inference will fail |
| `DecidraErrorRate` | error ratio > 0.01 | 5m | user-visible failures |
| `DecidraChannelSaturated` | `decidra_engine_channel_occupancy > 0.8 * decidra_engine_queue_capacity` | 2m | admission 503s imminent |
| `DecidraBacklog` | p95 `decidra_request_queue_seconds` > 1s | 5m | worker backlog, even if the channel is not full |
| `DecidraEngineFailure` | `increase(decidra_engine_failures_total[5m]) > 0` | 0m | any engine stage failure |
| `DecidraOomRetry` | `increase(decidra_engine_oom_retries_total[10m]) > 0` | 0m | memory headroom too tight |
| `DecidraCacheIneffective` | `((sum by (job, instance, model_name) (rate(decidra_state_cache_requests_total{result="miss"}[10m])) / sum by (job, instance, model_name) (rate(decidra_state_cache_requests_total[10m]))) > 0.9) and on(job, instance, model_name) (decidra_state_cache_usage_ratio{tier="gpu"} >= 0.99)` | 30m | cache is full but most requests still cold-capture; capacity warning, not a page |
| `DecidraGpuMemoryLow` | `(decidra_gpu_memory_total_bytes > 0) and (decidra_gpu_memory_free_bytes / decidra_gpu_memory_total_bytes < 0.1)` | 5m | physical headroom |
| `DecidraPassSlow` | p95 pass > 5s | 5m | engine regression |

Do not alert on `decidra_engine_requests_waiting > 0` alone; that is normal
under load. Do not alert on cache entries without the capacity ratio. The
cache-ineffective alert is a capacity warning: a workload with genuinely
unique states will legitimately miss, so tune the threshold and cache size
rather than paging on it. A pinned cache that is full does not evict; the
signal that matters is the miss ratio, not the usage ratio alone.

### 12.4 Dashboard panels

1. Throughput: q/s, successful requests/s, tokens/s by phase.
2. Latency: success-path e2e p50/p95/p99, queue, tokenize, capture, pass by cache; error latency is visible in outcome counters, not the SLO panel.
3. Queue: channel occupancy/capacity, waiting, running, queue-wait p95.
4. Errors: outcomes stacked, failures by stage, OOM retries.
5. Cache: hit ratio, entries/bytes by tier, usage ratio, evictions, offload
   and restore rates.
6. GPU: free/total, worker busy fraction, engine group p95 by cache.
7. Build/engine info table (version, quantization, arch, vision) and worker
   liveness.

## 13. Migration and rollout

| phase | work | gate |
|---|---|---|
| P0 | Add this spec; update README/DEPLOY/ENGINE docs to the new names; add a CHANGELOG entry for the breaking `semel_*` → `decidra_*` rename. | docs review |
| P1 | Telemetry foundation: builder, descriptions, buckets, zero-init, upkeep, endpoint content type. | local recorder tests |
| P2 | HTTP middleware + `systemone` outcome refactor + backend timings/tokens. | API tests |
| P3 | Worker queue/group/cache/GPU metrics + `PhaseTimer`. | GPU smoke + A/B |
| P4 | Dashboards, alerts, load test, contract docs (`docs/METRICS.md`). | acceptance §4 |

Rollout:

- Treat `/metrics` as an internal endpoint. The server has no authentication;
  use a Kubernetes NetworkPolicy, security group, or an authenticating reverse
  proxy. The endpoint exposes aggregate state only, but it is still a scrape
  target and a cardinality surface.
- Deploy to one instance with `DECIDRA_METRICS_ENABLED=true`; scrape it and
  compare counters to the existing logs for one hour.
- Update dashboards to `decidra_*`; keep a temporary Prometheus recording rule
  for any external `semel_*` consumer if one is discovered:
  `semel_requests_total` etc. must not be emitted by the server.
- Roll to the fleet. Metrics are observability-only; the rollback is
  `DECIDRA_METRICS_ENABLED=false` or the previous binary. Disabling makes
  `/metrics` return 404, so Prometheus marks the target `up=0` and
  `DecidraDown` fires; remove the target or silence that alert as part of the
  same change. `DecidraWorkerDown` disappears with the series, so `up` is the
  fallback signal.
- After 1.0, metric renames/removals require a two-release deprecation note;
  additions are non-breaking.

Legacy name mapping (dashboards only; the server does **not** emit these
aliases):

| old `semel_*` | new metric | semantic change |
|---|---|---|
| `semel_requests_total` | `decidra_systemone_requests_total{outcome="success"}` | old counted only successful rows; new counts completed responses by outcome |
| `semel_tokenize_seconds` | `decidra_request_tokenize_seconds` | same meaning; explicit request prefix |
| `semel_state_cache_hits_total` | `decidra_state_cache_requests_total{result="gpu_hit"}` + `...{result="host_hit"}` | old was per group; new is per request |
| `semel_state_cache_misses_total` | `decidra_state_cache_requests_total{result="miss"}` | old was per group; new is per request |
| `semel_group_jobs` | `decidra_engine_group_requests` | same meaning |
| `semel_group_rows` | `decidra_engine_group_questions` | same meaning |
| `semel_capture_seconds` | `decidra_engine_state_capture_seconds` | now true GPU event time, not host enqueue time |
| `semel_queue_wait_seconds` | `decidra_request_queue_seconds` | old included capture; new is queue/coalescing only |
| `semel_pass_seconds` | `decidra_engine_question_pass_seconds` | adds `cache` label |
| `semel_state_cache_states` | `decidra_state_cache_entries{tier="gpu"}` | same meaning |
| `semel_state_cache_bytes` | `decidra_state_cache_bytes{tier="gpu"}` | same estimate |
| `semel_host_cache_states` | `decidra_state_cache_entries{tier="host"}` | same meaning |
| `semel_gpu_free_bytes` | `decidra_gpu_memory_free_bytes` | adds `device="0"` |
| `semel_state_capture_oom_retries_total` | `decidra_engine_oom_retries_total{stage="state_capture"}` | same meaning |
| `semel_question_pass_oom_retries_total` | `decidra_engine_oom_retries_total{stage="question_pass"}` | same meaning |

Because the unit changed for the cache hit/miss counters (group → request), a
naive recording rule is not equivalent. Update the dashboard query rather than
aliasing it.

## 14. Risks

| risk | severity | mitigation |
|---|---|---|
| Histograms silently become summaries | high | explicit bucket table + test for `{quantile=` |
| Scrape content type rejected by Prometheus 3 | high | exact `text/plain; version=0.0.4` + header test |
| Raw histogram memory grows between scrapes | medium | 5 s `run_upkeep`; `render()` also drains, so a dead upkeep task does not lose data, only bounds memory less tightly. Log unexpected upkeep-task exit. |
| Recorder installed after engine load | medium | reorder startup; test build info present |
| Gauge drift on cancelled requests/panics | medium | RAII guards for in-flight/running; increment-before-send with rollback for waiting/channel; reconciliation on group end |
| Misleading capture timing | medium | `PhaseTimer` CUDA events; skip on failure |
| Renaming `semel_*` breaks a dashboard | medium | docs/migration note; recording rule for external consumers |
| High cardinality via route/status/error | high | `MatchedPath`, enums, no raw paths/messages |
| Metrics endpoint exposed without auth | medium | same trust boundary as the API; network policy; aggregate data only |
| Scrape/render contend with worker | low | bounded series; measure; `spawn_blocking` fallback |
| GPU memory gauge is stale when idle | low | document; refresh at group end; DCGM for physical monitoring |
| Global `model_name` prevents multi-model per process | low | v1 is single-model; future per-metric label |
| Counter resets confuse raw queries | low | document `rate`/`increase`; start-time gauge |

## 15. Alternatives considered

| alternative | pros | cons | decision |
|---|---|---|---|
| `metrics` + `metrics-exporter-prometheus` (chosen) | already present; standard text/buckets/labels/help; thread-safe | global recorder; summaries by default; upkeep responsibility | chosen |
| Flare-style custom renderer | full control; zero recorder overhead; JSON snapshot | reimplements exposition/escaping/buckets; more tests | rejected for v1 |
| OpenTelemetry SDK + Prometheus exporter | vendor-neutral, tracing correlation | heavier dependency; translation layer; still needs a scrape endpoint | deferred |
| `prometheus` crate | mature encoder, native histograms possible | manual registration/labels; global singleton; more code | rejected |
| `axum-prometheus` | turnkey HTTP metrics | own registry/exporter; less control over labels/exclusions | rejected |
| Logs only | no new endpoint | no time series, no SLOs/autoscaling | rejected |
| Push gateway | works behind NAT | not the standard scrape model; extra service | rejected |
| In-process NVML for GPU utilization | fresh physical metrics | new dependency, device handle, duplicate of DCGM | deferred |
| Separate metrics listener/auth | isolation | more lifecycle/config surface; not needed for v1 | deferred |

## 16. Open questions and future work

1. Separate metrics listener (`DECIDRA_METRICS_BIND_ADDR`) or bearer auth for
   environments where the API port is exposed. Deferred; network policy is the
   v1 control.
2. OpenMetrics text/protobuf/native histograms. The current exporter supports
   protobuf behind a feature; text 0.0.4 is sufficient and universally
   scrapable. Revisit if Exemplars or native histograms are needed.
3. Per-question latency. The pass is one batched GPU execution; per-question
   timing would require per-row CUDA events or timestamps. Use request and pass
   metrics until a concrete need appears.
4. Kernel/stage breakdown (attention, GDN, GEMM, norms). `DECIDRA_PROFILE` and
   Nsight already measure this; always-on CUDA events would add overhead.
   Consider an opt-in `EngineStats` snapshot if operators need it.
5. GPU utilization/power/temperature. Use DCGM exporter / `nvidia-smi`; the
   server should not duplicate DCGM. The engine-busy fraction from
   `group_seconds_sum` is a useful proxy.
6. Process RSS/CPU/FD metrics. cAdvisor/node_exporter cover the container; add
   only if a deployment cannot run them.
7. Custom histogram buckets via environment. vLLM supports it; for v1 the
   static table is a contract. Add only if a deployment proves the defaults
   wrong.
8. Multi-model serving. The global `model_name` label and one-engine worker
   assume a single model; a future multi-model design needs per-metric labels
   and per-model queue/cache state.

## 17. Review log

This section is filled by the required review passes. Each pass reads this
document and the referenced code, then records severity-ranked findings and
their disposition. The document is not final until pass 3 signs off.

### Pass 1 — 2026-10-24, repo fit, semantics, feasibility

Reviewer: independent reviewer agent. Verdict: no blocker; 1 high, 4 medium,
6 low findings. All findings were accepted and applied.

| id | finding | disposition |
|---|---|---|
| H1 | Cache result was taken at group start, but `ensure_headroom` can evict the group's own host entry and force a cold capture. | Compute one `cache_result` after the post-headroom host re-lookup and use it for `state_cache_requests_total`, reused tokens, `group_seconds{cache}`, `question_pass_seconds{cache}`, and restore/capture accounting. §7.4, §8.5. |
| M1 | `engine_queue_capacity` is the channel capacity, but `engine_requests_waiting` also includes pulled/backlogged jobs; the saturation alert could fire with no 503 possible. | Added `decidra_engine_channel_occupancy`; saturation alert now uses occupancy/capacity, plus a separate backlog alert. §7.4, §12.2–12.3. |
| M2 | The <500-series budget ignored `_sum`, `_count`, `+Inf`, and label multiplicity. | Recalculated to ~600–800 steady state and ~1,120 worst case after later label additions; scrape budget raised to <200 KiB/<10 ms. §7.3, §10. |
| M3 | The mock-backend test asserted backend-only metrics that `MockBackend` never records. | Mock test now asserts only HTTP/systemone series; request/engine histograms moved to GPU smoke tests. §11.2–11.3. |
| M4 | `tower::ServiceExt` is not a direct dependency. | Add `tower = { version = "0.5", features = ["util"] }` to `[dev-dependencies]`. §11.2. |
| L1 | `PhaseTimer` must use timing-enabled CUDA events; `new_event(None)` disables timing. | Use `Some(CUevent_flags::CU_EVENT_DEFAULT)` and log `seconds()` failures. §8.6. |
| L2 | `engine_failures_total{stage="scheduler"}` had no producer. | Increment it when `receiver.await` fails in `NativeBackend::answers`. §8.4. |
| L3 | Reply-stage cancellation was applied to pre-work error sends. | Only successful-pass reply sends count as `stage="reply"`. §7.4, §8.5. |
| L4 | The bucket test required positive bounds, but the images family starts at 0. | Assertion changed to strictly increasing and non-negative. §11.1. |
| L5 | `gpu_info{arch}` from `GpuArch::label()` would not join with `compiled_arch`. | Derive `sm_<major><minor>` from compute capability. §8.1. |
| L6 | Waiting/channel increments after `try_send` could race a fast worker dequeue. | Increment before `try_send`, roll back on `Full`/`Disconnected`. §7.4, §8.5. |

Residual note from the review: capture/vision/restore are genuinely
unsynchronized until the question pass's final `clone_dtoh`; §8.6 now states
that assumption explicitly and requires verifying it once on the target
driver.

### Pass 2 — 2026-10-24, Prometheus correctness, cardinality, SLOs, tests

Reviewer: independent reviewer agent. Verdict: incorrect before fixes; 1 P1,
2 P2, 3 P3 findings. The reviewer independently verified the exporter
internals, bucket coverage, naming/unit behavior, and the code caps. All
findings were accepted and applied.

| id | finding | disposition |
|---|---|---|
| P1 | `DecidraCachePressure` fired permanently once the pinned 16-entry cache filled, because the ratio is pinned at 1.0 even when the cache is healthy. | First fix used capacity + cold-capture churn, but pass 3 showed that also fires permanently under churn. Final alert is a miss-ratio + full-cache **warning** (`DecidraCacheIneffective`, 30m); the ratio is part of the warning gate but not sufficient alone, and remains a dashboard level signal. §12.3. |
| P2 | `image_busy` admission reason had no producer: image decode uses `acquire_owned().await`, not `try_acquire`. | The initial code had no timeout, so the label was removed. The final shipped code has a 30 s semaphore timeout that returns `Overloaded`; the implementation therefore retains `image_busy` and instruments it at that timeout. §7.3, §7.4. |
| P2 | The e2e p95 query aggregated all `outcome` values, so fast 400s masked slow failures. | Filtered the SLO query to `outcome="success"`; error latency is visible in outcome counters. §12.2, §12.4. |
| P3 | Worker-busy query returned one series per `cache` value. | Wrapped in `sum(...)` before treating it as a wall-time fraction. §12.2. |
| P3 | `DecidraBacklog` and `DecidraQueueWaitSlow` were the same alert. | Kept `DecidraBacklog`; removed `DecidraQueueWaitSlow`. §12.3. |
| P3 | Worst-case series ceiling was understated at 1,000. | Recomputed the enumerated domains: 20 HTTP method/route combinations plus the `unresolved` group label → ~1,120 series worst case. §7.3, §10. |

The reviewer confirmed the following as correct: text 0.0.4 content type,
final LF, `+Inf`/`_count` behavior, naming/unit idempotence, bucket coverage
against `MAX_QUESTIONS=64`/`MAX_IMAGES=16`/`MAX_STATE_TOKENS=32768`,
counter/gauge separation, and zero-init semantics for existing label domains.

### Pass 3 — 2026-10-24, security, performance, failure modes, rollout

Reviewer: independent reviewer agent. Verdict: incorrect before fixes; 1 P1,
5 P2, 3 P3 findings. All findings were accepted and applied.

| id | finding | disposition |
|---|---|---|
| F1 | The pass-2 cache-pressure alert still fired permanently on a pinned full cache because captures continue for every new state. | Replaced with a miss-ratio + full-cache warning (`DecidraCacheIneffective`, 30m), explicitly a capacity warning, not a page. §12.3. |
| F2 | `/metrics` was behind the 128-permit admission semaphore and could 503 under saturation; scrape floods multiplied render cost. | `admit_request` bypasses the semaphore for `MatchedPath == "/metrics"`; the endpoint caches its rendered body for 1 s to single-flight concurrent scrapes. §8.3, §11.2. |
| F3 | `systemone_requests_total` missed the admission 503 and body-limit 413 because they bypass the handler. | `track_http` is now the sole writer, reads an `Outcome` response extension set by the handler, and falls back to status mapping for pre-handler rejections. §8.3, §11.2. |
| F4 | Cache-result accounting was undefined on `run_group` exits before the post-headroom re-lookup. | `cache_result: Option<CacheResult>`; unresolved groups use `cache="unresolved"`, do not touch per-request cache counters, and still record group failure. §7.3–7.4, §8.5. |
| F5 | The flag-to-`quantization` mapping was unspecified. | Added the exact ordered predicate table for the five values; per-module exclusion lists do not affect the label. §7.4. |
| F6 | Disabling metrics makes `/metrics` 404, so `up=0` and `DecidraDown` fires; the rollback text ignored this. | Documented the required scrape-target removal/silence in D9 and §13; `up` is the fallback if worker series disappear. §1, §13. |
| F7 | The end-of-group cache insert happened after the finish guard's gauge refresh, leaving gauges one entry low. | Insert/offload before the final refresh, or refresh again after it. §8.5. |
| F8 | A stale “queue full/image busy” parenthetical remained after removing the phantom admission reason. | Replaced with “queue full / worker stopped”. §7.4. |
| F9 | The contract test used a `builder` helper that was not in the declared API. | Added `pub(crate) fn builder(...)`; `install` and the test share it so buckets/global labels cannot drift. §8.1. |

The reviewer verified the exporter suffix logic, zero-init behavior, bucket
coverage, and queue/gauge accounting. The remaining design choices are
intentional: text 0.0.4 only, no aliases for `semel_*`, one `model_name`
global label, and a 404 when metrics are disabled.

### Pass 4 — 2026-10-24, final sign-off

Reviewer: independent reviewer agent. Verdict: **ready for implementation**;
no blockers or high findings. One P2 remained: §5.4 still described raw
gauge increment/decrement for queue/running, contradicting the atomic-backed
`QueueCounters` rule in §7.4/§8.5. Fixed by making the mechanism explicit:
in-flight uses an RAII guard; queue/running/channel use saturating atomics and
`set()`. The reviewer also flagged two stale review-log numbers (the series
ceiling and the cache-alert role); both were corrected. All pass-1/2/3
dispositions were confirmed present in the normative sections, not only in
this log. The reviewer verified the exporter's suffix behavior, the
`Option<CacheResult>` insertion point, `track_http`'s visibility of the
admission 503 and body-limit 413, and the alert label sets.

### Final consistency check

A final pass re-read the document end-to-end after the four reviews and
checked:

- every catalog metric is unique, uses `decidra_` and a valid unit/type;
- every histogram has a bucket family and no summary fallback;
- every counter ends in `_total`; every gauge is a current-state value;
- zero-init label domains match the label tables (including invalid
  tier/reason combinations being excluded);
- the queue/running/channel accounting has one writer per transition and an
  RAII guard for panic paths;
- `/metrics` is outside admission and HTTP self-metrics, with a one-second
  scrape cache;
- the error ratio and SLO queries count pre-handler rejections;
- the legacy mapping does not claim semantic equivalence where the unit
  changed;
- the review findings above are reflected in the normative sections, not only
  in this log.

Implementation note (2026-10-25): the shipped API has a 30 s image-decode
semaphore timeout, so `image_busy` is a real admission reason after all; the
catalog and zero-init include it. The image-position validation failure also
adds a `positions` failure stage. Live verification on tinylife covered text,
image, GPU-hit, host-restore, and 8-concurrent-request paths; the Prometheus
target and the generated TinyLife Models dashboard row were checked against
the live scrape. See `docs/ops/tinylife-decidra-metrics.md`.

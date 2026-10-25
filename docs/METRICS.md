# Prometheus metrics

Decidra exposes aggregate engine, scheduler, cache, and request metrics at
`GET /metrics`. The endpoint uses Prometheus text exposition format **0.0.4**
with an explicit `Content-Type: text/plain; version=0.0.4; charset=utf-8`.
It is unauthenticated and intended for the internal network only.

The design and review log are in
[`docs/specs/2026-10-24-prometheus-metrics.md`](specs/2026-10-24-prometheus-metrics.md).

## Endpoint

| property | value |
|---|---|
| method | `GET`, `HEAD` |
| path | `/metrics` |
| success | `200`, text 0.0.4 |
| disabled | `404` (`DECIDRA_METRICS_ENABLED=false`) |
| cache | rendered body cached for 1 s; concurrent scrapes share it |
| admission | bypasses the 128-slot HTTP request gate so it remains scrapeable under saturation |
| auth | none; same trust boundary as `/v1/systemone` |

All metrics carry `model_name="<DECIDRA_MODEL>"`. Prometheus adds `job` and
`instance`. Counters reset on process restart; use `rate()`/`increase()`.

## Metric catalog

### Request and HTTP

| metric | type | labels | meaning |
|---|---|---|---|
| `decidra_systemone_requests_total` | counter | `outcome` | completed `POST /v1/systemone` responses; outcomes: `success`, `invalid_request`, `unsupported`, `overloaded`, `timeout`, `backend_error`, `internal_error` |
| `decidra_systemone_request_duration_seconds` | histogram | `outcome` | HTTP middleware-entry to response, including admission/body extraction |
| `decidra_http_requests_total` | counter | `method`, `route`, `status` | all HTTP responses except `/metrics` |
| `decidra_http_request_duration_seconds` | histogram | `method`, `route` | all HTTP requests except `/metrics` |
| `decidra_http_requests_in_flight` | gauge | `route` | currently handled HTTP requests |
| `decidra_request_queue_seconds` | histogram | — | worker submission to state-group start |
| `decidra_request_engine_seconds` | histogram | — | worker submission to reply, including queue wait and GPU work |
| `decidra_request_tokenize_seconds` | histogram | — | tokenization, image decode, blocking-pool wait, preparation |
| `decidra_image_decode_seconds` | histogram | — | one image decode |
| `decidra_request_state_tokens` | histogram | — | shared state-block tokens per request |
| `decidra_request_question_tokens` | histogram | — | question-block tokens per request |
| `decidra_request_questions` | histogram | — | questions per request |
| `decidra_request_images` | histogram | — | images per request |

### Scheduler and engine

| metric | type | labels | meaning |
|---|---|---|---|
| `decidra_engine_worker_alive` | gauge | — | 1 while the CUDA worker thread is running |
| `decidra_engine_requests_waiting` | gauge | — | submitted requests not yet executing in a group |
| `decidra_engine_requests_running` | gauge | — | requests in the current executing group |
| `decidra_engine_channel_occupancy` | gauge | — | jobs in the bounded worker channel |
| `decidra_engine_queue_capacity` | gauge | — | channel/request-slot capacity (`DECIDRA_MAX_QUEUE`) |
| `decidra_engine_groups_total` | counter | `outcome` | state groups, `success`/`failed` |
| `decidra_engine_group_requests` | histogram | — | requests per group |
| `decidra_engine_group_questions` | histogram | — | questions per group |
| `decidra_engine_group_seconds` | histogram | `cache` | group duration; `cache`: `gpu_hit`, `host_hit`, `miss`, `unresolved` |
| `decidra_engine_question_pass_seconds` | histogram | `cache` | batched question pass |
| `decidra_engine_state_capture_seconds` | histogram | — | cold capture GPU span, including OOM retry |
| `decidra_engine_vision_encode_seconds` | histogram | — | vision-tower encoding |
| `decidra_engine_oom_retries_total` | counter | `stage` | `state_capture`, `question_pass` |
| `decidra_engine_failures_total` | counter | `stage` | `vision_encode`, `state_capture`, `state_restore`, `state_offload`, `question_pass`, `rows`, `positions`, `scheduler`, `headroom` |
| `decidra_engine_admission_rejections_total` | counter | `reason` | `queue_full`, `image_busy`, `worker_stopped` |
| `decidra_engine_requests_cancelled_total` | counter | `stage` | `queued`, `reply` |
| `decidra_engine_tokens_total` | counter | `phase` | `state` tokens actually captured; `question` tokens actually passed |
| `decidra_engine_questions_total` | counter | `outcome` | `answered`, `failed`, `cancelled` |
| `decidra_engine_state_captures_total` | counter | — | successful cold captures |

### Exact-state cache

| metric | type | labels | meaning |
|---|---|---|---|
| `decidra_state_cache_requests_total` | counter | `result` | per request: `gpu_hit`, `host_hit`, `miss` |
| `decidra_state_cache_reused_tokens_total` | counter | — | state tokens reused, once per hit request |
| `decidra_state_cache_entries` | gauge | `tier` | cached states, `gpu`/`host` |
| `decidra_state_cache_bytes` | gauge | `tier` | estimated state bytes |
| `decidra_state_cache_capacity_entries` | gauge | — | GPU entry budget |
| `decidra_state_cache_capacity_bytes` | gauge | `tier` | byte budget |
| `decidra_state_cache_usage_ratio` | gauge | `tier` | fraction of budget in use; may exceed 1 for an oversized state |
| `decidra_state_cache_evictions_total` | counter | `tier`, `reason` | `entry_limit`, `byte_limit`, `host_budget`, `offload_failed` |
| `decidra_state_cache_offloads_total` | counter | — | GPU→host copies |
| `decidra_state_cache_offload_bytes_total` | counter | — | GPU→host bytes |
| `decidra_state_cache_offload_seconds` | histogram | — | GPU→host copy duration |
| `decidra_state_cache_restores_total` | counter | — | host→GPU restores |
| `decidra_state_cache_restore_bytes_total` | counter | — | host→GPU bytes |
| `decidra_state_cache_restore_seconds` | histogram | — | host→GPU restore duration |

### Build, engine, and GPU identity

| metric | type | labels |
|---|---|---|
| `decidra_build_info` | gauge (1) | `version`, `profile`, `compiled_arch`, `native` |
| `decidra_engine_info` | gauge (1) | `attention_mode`, `quantization`, `vision`, `gpu_embed`, `gdn_chunked`, `weights_cached` |
| `decidra_gpu_info` | gauge (1) | `device`, `name`, `arch`, `sms`, `compute_capability` |
| `decidra_process_start_time_seconds` | gauge | — |
| `decidra_gpu_memory_free_bytes` | gauge | `device` |
| `decidra_gpu_memory_total_bytes` | gauge | `device` |

`quantization` is the configured execution path (`w4a8_g64`, `w4a8`, `w4a16`,
`w4a16_fused`, `bf16`), not checkpoint provenance. `arch`/`compiled_arch` use
normalized `sm_<major><minor>`.

## Semantics

- A **request** is one `POST /v1/systemone`; a **question** is one entry in
  `questions`; a **group** is one exact-state set executed by the worker.
- `request_queue_seconds` stops when the group starts. `request_engine_seconds`
  includes queue wait and GPU work. The HTTP request duration includes
  admission, tokenization, and serialization.
- Cache result is resolved after the VRAM preflight and its host-cache
  re-lookup. A host entry evicted by that preflight becomes a cold `miss`.
- `engine_tokens_total{phase="state"}` counts cold-capture work once per group;
  cache hits do not re-count state tokens. `phase="question"` counts question
  tokens actually passed.
- `systemone_requests_total` includes admission 503s and body-limit 413s.
  Client cancellation is reported by `engine_requests_cancelled_total`; a
  request cancelled before any handler response has no `systemone` outcome.
- `state_cache_usage_ratio` is the larger of entry and byte usage for the GPU
  tier, and byte usage for the host tier. It is a budget ratio, not physical
  GPU memory.
- `engine_worker_alive=0` means the CUDA worker exited or panicked; all
  inference will fail. A dead worker also resets the queue gauges.
- Metrics are best-effort: a GPU memory query or phase timer failure is logged
  and skipped, never returned to the client.

## PromQL examples

```promql
# questions/s answered
sum(rate(decidra_engine_questions_total{outcome="answered"}[1m]))

# successful requests/s
sum(rate(decidra_systemone_requests_total{outcome="success"}[1m]))

# tokens/s by phase
sum by (phase) (rate(decidra_engine_tokens_total[1m]))

# success-path p95 end-to-end latency
histogram_quantile(
  0.95,
  sum by (le) (rate(decidra_systemone_request_duration_seconds_bucket{outcome="success"}[5m]))
)

# p95 queue wait and question pass
histogram_quantile(0.95, sum by (le) (rate(decidra_request_queue_seconds_bucket[5m])))
histogram_quantile(0.95, sum by (le, cache) (rate(decidra_engine_question_pass_seconds_bucket[5m])))

# error ratio
sum(rate(decidra_systemone_requests_total{outcome!="success"}[5m]))
  / sum(rate(decidra_systemone_requests_total[5m]))

# cache hit ratio
sum(rate(decidra_state_cache_requests_total{result!="miss"}[5m]))
  / sum(rate(decidra_state_cache_requests_total[5m]))

# channel saturation
decidra_engine_channel_occupancy / decidra_engine_queue_capacity

# worker busy fraction (state groups + post-reply offload)
sum(rate(decidra_engine_group_seconds_sum[1m]))
  + (sum(rate(decidra_state_cache_offload_seconds_sum[1m])) or vector(0))
```

## Alerts

| alert | expression | for |
|---|---|---|
| `DecidraDown` | `up{job="decidra"} == 0` | 1m |
| `DecidraWorkerDown` | `decidra_engine_worker_alive == 0` | 1m |
| `DecidraErrorRate` | error ratio > 0.01 | 5m |
| `DecidraChannelSaturated` | `decidra_engine_channel_occupancy > 0.8 * decidra_engine_queue_capacity` | 2m |
| `DecidraBacklog` | p95 `decidra_request_queue_seconds` > 1s | 5m |
| `DecidraEngineFailure` | `increase(decidra_engine_failures_total[5m]) > 0` | 0m |
| `DecidraOomRetry` | `increase(decidra_engine_oom_retries_total[10m]) > 0` | 0m |
| `DecidraCacheIneffective` | miss ratio > 0.9 and GPU usage ratio >= 0.99 | 30m (warning) |
| `DecidraGpuMemoryLow` | free/total < 0.1 with total > 0 | 5m |
| `DecidraPassSlow` | p95 question pass > 5s | 5m |

Disabling metrics makes `/metrics` return 404, so Prometheus marks the target
`up=0` and `DecidraDown` fires. Remove the scrape target or silence that alert
at the same time.

## Migration from `semel_*`

The server no longer emits `semel_*`. Update dashboards to the names above.
Important semantic changes:

- cache hit/miss counters are per request, not per group;
- `request_queue_seconds` no longer includes state capture;
- `state_capture_seconds` is CUDA-event time, not host submission time;
- the cache result is resolved after the VRAM preflight;
- `systemone_requests_total` includes pre-handler 503/413 rejections.

A temporary Prometheus recording rule can bridge an external consumer, but
do not run an old dashboard unchanged because the units changed.

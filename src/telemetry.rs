//! Prometheus recorder, metric contract, and periodic upkeep.
//!
//! The metric names, types, labels, and bucket families here are the public
//! observability contract. Keep them in sync with `docs/METRICS.md` and the
//! contract test at the bottom of this module.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use metrics::{Unit, describe_counter, describe_gauge, describe_histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};

use crate::config::Config;

const HTTP_SECONDS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
    120.0, 300.0,
];
const REQUEST_SECONDS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0,
];
const QUEUE_SECONDS: &[f64] = &[
    0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
    5.0, 10.0, 30.0,
];
const TOKENIZE_SECONDS: &[f64] = &[
    0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
    5.0, 10.0,
];
const IMAGE_SECONDS: &[f64] = &[
    0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];
const PHASE_SECONDS: &[f64] = &[
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0,
];
const GROUP_SECONDS: &[f64] = &[
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0,
];
const VISION_SECONDS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];
const OFFLOAD_SECONDS: &[f64] = &[
    0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
];
const TOKENS: &[f64] = &[
    1.0, 16.0, 64.0, 256.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0,
    262144.0, 524288.0, 1048576.0, 2097152.0,
];
const QUESTIONS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
const IMAGES: &[f64] = &[0.0, 1.0, 2.0, 4.0, 8.0, 16.0];
const GROUP_REQUESTS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
const GROUP_QUESTIONS: &[f64] = &[
    1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0,
];

/// Every histogram and its bucket family. A missing entry would silently make
/// the exporter render a summary, so the contract test checks this table.
const HISTOGRAMS: &[(&str, &[f64])] = &[
    ("decidra_http_request_duration_seconds", HTTP_SECONDS),
    (
        "decidra_systemone_request_duration_seconds",
        REQUEST_SECONDS,
    ),
    ("decidra_request_engine_seconds", REQUEST_SECONDS),
    ("decidra_request_queue_seconds", QUEUE_SECONDS),
    ("decidra_request_tokenize_seconds", TOKENIZE_SECONDS),
    ("decidra_image_decode_seconds", IMAGE_SECONDS),
    ("decidra_request_state_tokens", TOKENS),
    ("decidra_request_question_tokens", TOKENS),
    ("decidra_request_questions", QUESTIONS),
    ("decidra_request_images", IMAGES),
    ("decidra_engine_group_requests", GROUP_REQUESTS),
    ("decidra_engine_group_questions", GROUP_QUESTIONS),
    ("decidra_engine_group_seconds", GROUP_SECONDS),
    ("decidra_engine_question_pass_seconds", PHASE_SECONDS),
    ("decidra_engine_state_capture_seconds", PHASE_SECONDS),
    ("decidra_engine_vision_encode_seconds", VISION_SECONDS),
    ("decidra_state_cache_offload_seconds", OFFLOAD_SECONDS),
    ("decidra_state_cache_restore_seconds", OFFLOAD_SECONDS),
];

#[derive(Clone, Debug)]
pub struct BuildInfo {
    pub version: &'static str,
    pub profile: &'static str,
    /// Normalized `sm_<major><minor>` (`sm_89`, `sm_90`, `sm_120`) or `none`.
    pub compiled_arch: String,
    pub native: bool,
}

#[derive(Clone, Debug)]
pub struct EngineInfo {
    pub attention_mode: &'static str,
    pub quantization: &'static str,
    pub vision: bool,
    pub gpu_embed: bool,
    pub gdn_chunked: bool,
    pub weights_cached: bool,
}

#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub device: &'static str,
    pub name: String,
    /// Normalized `sm_<major><minor>`, matching `BuildInfo::compiled_arch`.
    pub arch: String,
    pub sms: usize,
    pub compute_capability: String,
}

/// Build the recorder exactly as production uses it. Tests call this with
/// `build_recorder()` so bucket and global-label configuration cannot drift.
pub(crate) fn builder(config: &Config, _build: &BuildInfo) -> Result<PrometheusBuilder> {
    let mut builder = PrometheusBuilder::new()
        .add_global_label("model_name", config.model_id.clone())
        .with_recommended_naming(true);
    for (name, buckets) in HISTOGRAMS {
        builder = builder
            .set_buckets_for_metric(Matcher::Full((*name).to_owned()), buckets)
            .with_context(|| format!("configuring histogram buckets for {name}"))?;
    }
    Ok(builder)
}

/// Install the process-global recorder, register the contract, and initialize
/// known series. Must run before the engine loads.
pub fn install(config: &Config, build: &BuildInfo) -> Result<PrometheusHandle> {
    let handle = builder(config, build)?
        .install_recorder()
        .context("installing Prometheus metrics recorder")?;
    describe_all();
    init_known_series(config, build);
    Ok(handle)
}

/// Drain raw histogram samples into aggregated distributions between scrapes.
/// `render()` also drains, so this is a memory bound rather than a
/// correctness requirement.
pub fn spawn_upkeep(handle: PrometheusHandle) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(5));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            let handle = handle.clone();
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle.run_upkeep()))
                .is_err()
            {
                tracing::error!("metrics upkeep panicked; upkeep task stopped");
                break;
            }
        }
    })
}

pub fn record_engine_info(info: &EngineInfo, gpu: &GpuInfo) {
    let yes = |value: bool| if value { "true" } else { "false" };
    metrics::gauge!(
        "decidra_engine_info",
        "attention_mode" => info.attention_mode,
        "quantization" => info.quantization,
        "vision" => yes(info.vision),
        "gpu_embed" => yes(info.gpu_embed),
        "gdn_chunked" => yes(info.gdn_chunked),
        "weights_cached" => yes(info.weights_cached),
    )
    .set(1.0);
    metrics::gauge!(
        "decidra_gpu_info",
        "device" => gpu.device,
        "name" => gpu.name.clone(),
        "arch" => gpu.arch.clone(),
        "sms" => gpu.sms.to_string(),
        "compute_capability" => gpu.compute_capability.clone(),
    )
    .set(1.0);
}

pub fn set_gpu_memory(free: usize, total: usize) {
    metrics::gauge!("decidra_gpu_memory_free_bytes", "device" => "0").set(free as f64);
    metrics::gauge!("decidra_gpu_memory_total_bytes", "device" => "0").set(total as f64);
}

fn describe_all() {
    describe_counter!(
        "decidra_http_requests_total",
        Unit::Count,
        "Completed HTTP responses by method, matched route, and status."
    );
    describe_histogram!(
        "decidra_http_request_duration_seconds",
        Unit::Seconds,
        "HTTP request duration from middleware entry to response."
    );
    describe_gauge!(
        "decidra_http_requests_in_flight",
        Unit::Count,
        "HTTP requests currently being handled by matched route."
    );

    describe_counter!(
        "decidra_systemone_requests_total",
        Unit::Count,
        "Completed POST /v1/systemone responses by final outcome, including admission and body-limit rejections."
    );
    describe_histogram!(
        "decidra_systemone_request_duration_seconds",
        Unit::Seconds,
        "HTTP middleware-entry to response duration for POST /v1/systemone, including admission and body extraction."
    );
    describe_histogram!(
        "decidra_request_queue_seconds",
        Unit::Seconds,
        "Time from worker submission to the start of the request's state group."
    );
    describe_histogram!(
        "decidra_request_engine_seconds",
        Unit::Seconds,
        "Time from worker submission to reply, including queue wait and GPU work."
    );
    describe_histogram!(
        "decidra_request_tokenize_seconds",
        Unit::Seconds,
        "Tokenization, image decode, blocking-pool wait, and request preparation time."
    );
    describe_histogram!(
        "decidra_image_decode_seconds",
        Unit::Seconds,
        "Decode time for one input image."
    );
    describe_histogram!(
        "decidra_request_state_tokens",
        Unit::Count,
        "Tokens in the shared state block of a request."
    );
    describe_histogram!(
        "decidra_request_question_tokens",
        Unit::Count,
        "Total question-block tokens in a request."
    );
    describe_histogram!(
        "decidra_request_questions",
        Unit::Count,
        "Questions in a request."
    );
    describe_histogram!(
        "decidra_request_images",
        Unit::Count,
        "Input images in a request."
    );

    describe_gauge!(
        "decidra_engine_worker_alive",
        "1 while the CUDA worker thread is running, 0 after it exits or panics."
    );
    describe_gauge!(
        "decidra_engine_requests_waiting",
        Unit::Count,
        "Requests submitted to the CUDA worker but not yet executing in a state group; includes the channel, the coalescing window, and jobs queued behind an earlier group."
    );
    describe_gauge!(
        "decidra_engine_requests_running",
        Unit::Count,
        "Requests whose state group is currently executing in the CUDA worker."
    );
    describe_gauge!(
        "decidra_engine_channel_occupancy",
        Unit::Count,
        "Jobs currently in the bounded worker channel."
    );
    describe_gauge!(
        "decidra_engine_queue_capacity",
        Unit::Count,
        "Channel capacity; the threshold at which try_send returns 503."
    );
    describe_counter!(
        "decidra_engine_groups_total",
        Unit::Count,
        "State groups executed by the CUDA worker, by outcome."
    );
    describe_histogram!(
        "decidra_engine_group_requests",
        Unit::Count,
        "Requests per executed state group."
    );
    describe_histogram!(
        "decidra_engine_group_questions",
        Unit::Count,
        "Questions per executed state group."
    );
    describe_histogram!(
        "decidra_engine_group_seconds",
        Unit::Seconds,
        "Wall-clock duration of one state group from execution start to replies or failure."
    );
    describe_histogram!(
        "decidra_engine_question_pass_seconds",
        Unit::Seconds,
        "Batched question-pass duration for one state group."
    );
    describe_histogram!(
        "decidra_engine_state_capture_seconds",
        Unit::Seconds,
        "Cold state-capture span, including any OOM trim/retry; measured with CUDA events."
    );
    describe_histogram!(
        "decidra_engine_vision_encode_seconds",
        Unit::Seconds,
        "GPU vision-tower encoding time for one state group."
    );
    describe_counter!(
        "decidra_engine_oom_retries_total",
        Unit::Count,
        "OOM trim-and-retry attempts by stage."
    );
    describe_counter!(
        "decidra_engine_failures_total",
        Unit::Count,
        "Failed engine operations by stage."
    );
    describe_counter!(
        "decidra_engine_admission_rejections_total",
        Unit::Count,
        "Requests rejected before execution by reason."
    );
    describe_counter!(
        "decidra_engine_requests_cancelled_total",
        Unit::Count,
        "Requests abandoned by the client, by cancellation stage."
    );
    describe_counter!(
        "decidra_engine_tokens_total",
        Unit::Count,
        "Tokens processed by the engine; state counts successful cold captures, question counts successful question passes."
    );
    describe_counter!(
        "decidra_engine_questions_total",
        Unit::Count,
        "Questions processed by the engine, by outcome."
    );
    describe_counter!(
        "decidra_engine_state_captures_total",
        Unit::Count,
        "Cold state captures completed successfully."
    );

    describe_counter!(
        "decidra_state_cache_requests_total",
        Unit::Count,
        "Inference requests by exact-state cache result."
    );
    describe_counter!(
        "decidra_state_cache_reused_tokens_total",
        Unit::Count,
        "State tokens reused from cache, counted once per hit request."
    );
    describe_gauge!(
        "decidra_state_cache_entries",
        Unit::Count,
        "Cached exact states by tier."
    );
    describe_gauge!(
        "decidra_state_cache_bytes",
        Unit::Bytes,
        "Estimated bytes held by the exact-state cache by tier."
    );
    describe_gauge!(
        "decidra_state_cache_capacity_entries",
        Unit::Count,
        "Configured maximum GPU state-cache entries."
    );
    describe_gauge!(
        "decidra_state_cache_capacity_bytes",
        Unit::Bytes,
        "Configured state-cache byte budget by tier."
    );
    describe_gauge!(
        "decidra_state_cache_usage_ratio",
        Unit::Percent,
        "Fraction of configured state-cache capacity in use by tier."
    );
    describe_counter!(
        "decidra_state_cache_evictions_total",
        Unit::Count,
        "State-cache evictions by tier and reason."
    );
    describe_counter!(
        "decidra_state_cache_offloads_total",
        Unit::Count,
        "States copied from GPU to host L2."
    );
    describe_counter!(
        "decidra_state_cache_offload_bytes_total",
        Unit::Bytes,
        "State bytes copied from GPU to host L2."
    );
    describe_histogram!(
        "decidra_state_cache_offload_seconds",
        Unit::Seconds,
        "GPU-to-host state offload duration."
    );
    describe_counter!(
        "decidra_state_cache_restores_total",
        Unit::Count,
        "States restored from host L2 to GPU."
    );
    describe_counter!(
        "decidra_state_cache_restore_bytes_total",
        Unit::Bytes,
        "State bytes restored from host L2 to GPU."
    );
    describe_histogram!(
        "decidra_state_cache_restore_seconds",
        Unit::Seconds,
        "Host-to-GPU state restore duration."
    );

    describe_gauge!("decidra_build_info", "Build metadata; value is always 1.");
    describe_gauge!(
        "decidra_engine_info",
        "Engine configuration metadata; value is always 1."
    );
    describe_gauge!(
        "decidra_gpu_info",
        "GPU identity metadata; value is always 1."
    );
    describe_gauge!(
        "decidra_process_start_time_seconds",
        Unit::Seconds,
        "Unix time at which the process started."
    );
    describe_gauge!(
        "decidra_gpu_memory_free_bytes",
        Unit::Bytes,
        "Free device memory from cudaMemGetInfo, including other processes."
    );
    describe_gauge!(
        "decidra_gpu_memory_total_bytes",
        Unit::Bytes,
        "Total device memory from cudaMemGetInfo."
    );
}

fn init_known_series(config: &Config, build: &BuildInfo) {
    // Build and process identity.
    metrics::gauge!(
        "decidra_build_info",
        "version" => build.version,
        "profile" => build.profile,
        "compiled_arch" => build.compiled_arch.clone(),
        "native" => if build.native { "true" } else { "false" },
    )
    .set(1.0);
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    metrics::gauge!("decidra_process_start_time_seconds").set(started);

    // Capacity gauges are written only here.
    metrics::gauge!("decidra_engine_queue_capacity").set(config.max_queue as f64);
    metrics::gauge!("decidra_state_cache_capacity_entries").set(config.state_cache_entries as f64);
    metrics::gauge!("decidra_state_cache_capacity_bytes", "tier" => "gpu")
        .set(config.state_cache_bytes as f64);
    metrics::gauge!("decidra_state_cache_capacity_bytes", "tier" => "host")
        .set(config.host_state_cache_bytes as f64);

    // Dynamic gauges start at zero so dashboards have a series immediately.
    metrics::gauge!("decidra_engine_worker_alive").set(0.0);
    metrics::gauge!("decidra_engine_requests_waiting").set(0.0);
    metrics::gauge!("decidra_engine_requests_running").set(0.0);
    metrics::gauge!("decidra_engine_channel_occupancy").set(0.0);
    for tier in ["gpu", "host"] {
        metrics::gauge!("decidra_state_cache_entries", "tier" => tier).set(0.0);
        metrics::gauge!("decidra_state_cache_bytes", "tier" => tier).set(0.0);
        metrics::gauge!("decidra_state_cache_usage_ratio", "tier" => tier).set(0.0);
    }
    for route in ["/v1/systemone", "/health", "/v1/models"] {
        metrics::gauge!("decidra_http_requests_in_flight", "route" => route).set(0.0);
    }

    // Error-path counters must exist before the first error.
    for outcome in [
        "success",
        "invalid_request",
        "unsupported",
        "overloaded",
        "timeout",
        "backend_error",
        "internal_error",
    ] {
        metrics::counter!("decidra_systemone_requests_total", "outcome" => outcome).increment(0);
    }
    for outcome in ["success", "failed"] {
        metrics::counter!("decidra_engine_groups_total", "outcome" => outcome).increment(0);
    }
    for outcome in ["answered", "failed", "cancelled"] {
        metrics::counter!("decidra_engine_questions_total", "outcome" => outcome).increment(0);
    }
    for stage in ["state_capture", "question_pass"] {
        metrics::counter!("decidra_engine_oom_retries_total", "stage" => stage).increment(0);
    }
    for stage in [
        "vision_encode",
        "state_capture",
        "state_restore",
        "state_offload",
        "question_pass",
        "rows",
        "positions",
        "scheduler",
        "headroom",
    ] {
        metrics::counter!("decidra_engine_failures_total", "stage" => stage).increment(0);
    }
    for reason in ["queue_full", "image_busy", "worker_stopped"] {
        metrics::counter!("decidra_engine_admission_rejections_total", "reason" => reason)
            .increment(0);
    }
    for stage in ["queued", "reply"] {
        metrics::counter!("decidra_engine_requests_cancelled_total", "stage" => stage).increment(0);
    }
    for phase in ["state", "question"] {
        metrics::counter!("decidra_engine_tokens_total", "phase" => phase).increment(0);
    }
    metrics::counter!("decidra_engine_state_captures_total").increment(0);

    for result in ["gpu_hit", "host_hit", "miss"] {
        metrics::counter!("decidra_state_cache_requests_total", "result" => result).increment(0);
    }
    metrics::counter!("decidra_state_cache_reused_tokens_total").increment(0);
    for (tier, reason) in [
        ("gpu", "entry_limit"),
        ("gpu", "byte_limit"),
        ("gpu", "offload_failed"),
        ("host", "host_budget"),
    ] {
        metrics::counter!(
            "decidra_state_cache_evictions_total",
            "tier" => tier,
            "reason" => reason
        )
        .increment(0);
    }
    metrics::counter!("decidra_state_cache_offloads_total").increment(0);
    metrics::counter!("decidra_state_cache_offload_bytes_total").increment(0);
    metrics::counter!("decidra_state_cache_restores_total").increment(0);
    metrics::counter!("decidra_state_cache_restore_bytes_total").increment(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config::for_test()
    }

    fn test_build() -> BuildInfo {
        BuildInfo {
            version: "0.0.0-test",
            profile: "debug",
            compiled_arch: "sm_89".to_owned(),
            native: true,
        }
    }

    #[test]
    fn every_catalog_histogram_is_configured() {
        let recorder = builder(&test_config(), &test_build())
            .unwrap()
            .build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            describe_all();
            init_known_series(&test_config(), &test_build());
            for (name, buckets) in HISTOGRAMS {
                for bucket in *buckets {
                    metrics::histogram!((*name).to_owned()).record(*bucket);
                }
            }
        });
        let text = handle.render();
        assert!(
            !text.contains("{quantile="),
            "histogram rendered as a summary"
        );
        for (name, _) in HISTOGRAMS {
            assert!(
                text.contains(&format!("# TYPE {name} histogram")),
                "missing type for {name}"
            );
            assert!(
                text.contains(&format!("{name}_bucket")),
                "missing buckets for {name}"
            );
            assert!(
                text.contains(&format!("{name}_sum")),
                "missing sum for {name}"
            );
            assert!(
                text.contains(&format!("{name}_count")),
                "missing count for {name}"
            );
        }
        assert!(text.contains("model_name=\"test/model\""));
        assert!(text.contains("decidra_engine_worker_alive{model_name=\"test/model\"} 0"));
        assert!(text.contains("decidra_engine_queue_capacity{model_name=\"test/model\"} 4"));
    }

    #[test]
    fn http_response_shape_is_explicit() {
        let recorder = builder(&test_config(), &test_build())
            .unwrap()
            .build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            describe_all();
            init_known_series(&test_config(), &test_build());
        });
        let text = handle.render();
        assert!(text.ends_with('\n'));
        assert!(text.contains("# HELP decidra_systemone_requests_total"));
        assert!(text.contains("# TYPE decidra_systemone_requests_total counter"));
        assert!(text.contains("decidra_systemone_requests_total{"));
    }
}

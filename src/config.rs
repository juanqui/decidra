//! d3-turbo-only server configuration.
//!
//! Decidra serves exactly one checkpoint family: `vllm-sr/d3-turbo` with
//! the `prefix_bidir` decision mask and a separate F32 readout. Every knob
//! below is an explicit, validated serving knob; model-family configuration
//! is part of the checkpoint contract, not an environment switch.

use std::{net::SocketAddr, path::PathBuf};

use anyhow::{Context, Result, bail};

pub const DEFAULT_BIND_ADDR: &str = "0.0.0.0:8080";
pub const DEFAULT_MODEL: &str = "vllm-sr/d3-turbo";
pub const DEFAULT_CHECKPOINT: &str = "/srv/models/d3-turbo-w4a8-vision-v1";
pub const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_MAX_QUEUE: usize = 32;
pub const DEFAULT_BATCH_REQUESTS: usize = 32;
pub const DEFAULT_BATCH_ROWS: usize = 32;
pub const DEFAULT_BATCH_WINDOW_US: u64 = 2_000;
pub const DEFAULT_STATE_CACHE_ENTRIES: usize = 16;
/// SM89 keeps large states from exhausting 24 GiB; one oversized state is
/// still admitted after evicting the rest.
pub const DEFAULT_STATE_CACHE_BYTES: usize = 5 * 1024 * 1024 * 1024;
/// Host-memory L2 for states evicted from the pinned GPU cache.
pub const DEFAULT_HOST_STATE_CACHE_BYTES: usize = 8 * 1024 * 1024 * 1024;
pub const MAX_STATE_TOKENS: usize = 32_768;
pub const MAX_QUESTIONS: usize = 64;

/// Production W4A8 configuration for the v4 d3-turbo artifact. The exclusions
/// are the measured gate configuration: `o/v/k` and `gate_proj` stay on the
/// W4A16 epilogue, the rest use INT8 per-token/per-64-group activations.
pub const DEFAULT_INT8_EXCLUDE: &str = "self_attn.v_proj,self_attn.k_proj";
/// Empty: every INT8 linear uses per-64-group activation scales. Together with
/// the SM89 default exclusions this keeps gate/up/down/o/q on W4A8, keeps k/v
/// on W4A16, and passes the 32-row gate (31/32 top-1, max Δp 0.0837).
pub const DEFAULT_INT8_GROUP_EXCLUDE: &str = "";

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: SocketAddr,
    pub model_id: String,
    pub checkpoint: PathBuf,
    /// Explicit tokenizer path; defaults to `<checkpoint>/tokenizer.json`.
    pub tokenizer: Option<PathBuf>,
    /// Checkpoint opened by the native engine (must contain the packed W4A8
    /// weights, `readout.safetensors`, and `decision_config.json`).
    pub native_model: PathBuf,
    pub native_cache: bool,
    /// Repack to the Marlin W4A8 layout at preload.
    pub native_marlin: bool,
    /// Dequant + cuBLAS reference path when Marlin is off.
    pub native_fused_w4a16: bool,
    pub native_gdn_chunked: bool,
    pub native_gdn_chunked_min_seq: usize,
    pub native_int8: bool,
    pub native_int8_clip: f32,
    pub native_int8_group_scale: bool,
    pub native_int8_group_exclude: Vec<String>,
    pub native_int8_exclude: Vec<String>,
    pub native_int8_all_layers: String,
    /// Keep the embedding table resident on the GPU (2.4 GiB).
    pub native_gpu_embed: bool,
    pub metrics_enabled: bool,
    pub max_body_bytes: usize,
    /// Requests waiting behind the single CUDA worker before 503.
    pub max_queue: usize,
    /// Maximum requests coalesced into one worker pass.
    pub batch_requests: usize,
    /// Soft question-row budget for one worker pass; the first job's question
    /// count selects how many requests fit.
    pub batch_rows: usize,
    /// Coalescing window in microseconds.
    pub batch_window_us: u64,
    /// Exact-state LRU entries shared across requests.
    pub state_cache_entries: usize,
    /// Approximate byte budget for cached state K/V/GDN buffers.
    pub state_cache_bytes: usize,
    /// Host-memory L2 byte budget for overflow states.
    pub host_state_cache_bytes: usize,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let bind_addr = env_string("DECIDRA_BIND_ADDR")
            .unwrap_or_else(|| DEFAULT_BIND_ADDR.to_owned())
            .parse::<SocketAddr>()
            .context("DECIDRA_BIND_ADDR must be a socket address such as 0.0.0.0:8080")?;
        let model_id = env_string("DECIDRA_MODEL").unwrap_or_else(|| DEFAULT_MODEL.to_owned());
        let checkpoint = PathBuf::from(
            env_string("DECIDRA_CHECKPOINT").unwrap_or_else(|| DEFAULT_CHECKPOINT.to_owned()),
        );
        let tokenizer = env_string("DECIDRA_TOKENIZER").map(PathBuf::from);
        let native_model = PathBuf::from(
            env_string("DECIDRA_NATIVE_MODEL").unwrap_or_else(|| checkpoint.display().to_string()),
        );
        let native_cache = env_bool("DECIDRA_NATIVE_CACHE")?.unwrap_or(true);
        let native_marlin = env_bool("DECIDRA_NATIVE_MARLIN")?.unwrap_or(true);
        let native_fused_w4a16 = env_bool("DECIDRA_NATIVE_FUSED_W4A16")?.unwrap_or(true);
        let native_gdn_chunked = env_bool("DECIDRA_NATIVE_GDN_CHUNKED")?.unwrap_or(true);
        let native_gdn_chunked_min_seq =
            env_usize("DECIDRA_NATIVE_GDN_CHUNKED_MIN_SEQ")?.unwrap_or(16);
        let native_int8 = env_bool("DECIDRA_NATIVE_INT8")?.unwrap_or(true);
        let native_int8_clip = match env_string("DECIDRA_NATIVE_INT8_CLIP") {
            Some(value) => value
                .parse::<f32>()
                .context("DECIDRA_NATIVE_INT8_CLIP must be a number in (0, 1]")?,
            None => 1.0,
        };
        if !(native_int8_clip > 0.0 && native_int8_clip <= 1.0) {
            bail!("DECIDRA_NATIVE_INT8_CLIP must be in (0, 1], got {native_int8_clip}");
        }
        let native_int8_group_scale = env_bool("DECIDRA_NATIVE_INT8_GROUP_SCALE")?.unwrap_or(true);
        let native_int8_group_exclude = list_env(
            "DECIDRA_NATIVE_INT8_GROUP_EXCLUDE",
            DEFAULT_INT8_GROUP_EXCLUDE,
        );
        let native_int8_exclude = list_env("DECIDRA_NATIVE_INT8_EXCLUDE", DEFAULT_INT8_EXCLUDE);
        let native_int8_all_layers =
            env_string("DECIDRA_NATIVE_INT8_ALL_LAYERS").unwrap_or_else(|| "last:8".to_owned());
        let native_gpu_embed = env_bool("DECIDRA_NATIVE_GPU_EMBED")?.unwrap_or(false);
        let metrics_enabled = env_bool("DECIDRA_METRICS_ENABLED")?.unwrap_or(true);
        let max_body_bytes = env_usize("DECIDRA_MAX_BODY_BYTES")?.unwrap_or(DEFAULT_MAX_BODY_BYTES);
        if max_body_bytes == 0 {
            bail!("DECIDRA_MAX_BODY_BYTES must be greater than 0");
        }
        let max_queue = env_usize("DECIDRA_MAX_QUEUE")?.unwrap_or(DEFAULT_MAX_QUEUE);
        if max_queue == 0 {
            bail!("DECIDRA_MAX_QUEUE must be at least 1");
        }
        let batch_requests = env_usize("DECIDRA_BATCH_REQUESTS")?.unwrap_or(DEFAULT_BATCH_REQUESTS);
        if batch_requests == 0 {
            bail!("DECIDRA_BATCH_REQUESTS must be at least 1");
        }
        let batch_rows = env_usize("DECIDRA_BATCH_ROWS")?.unwrap_or(DEFAULT_BATCH_ROWS);
        if batch_rows == 0 {
            bail!("DECIDRA_BATCH_ROWS must be at least 1");
        }
        let batch_window_us =
            env_u64("DECIDRA_BATCH_WINDOW_US")?.unwrap_or(DEFAULT_BATCH_WINDOW_US);
        if batch_window_us > 50_000 {
            bail!("DECIDRA_BATCH_WINDOW_US must be at most 50000");
        }
        let state_cache_entries =
            env_usize("DECIDRA_STATE_CACHE_ENTRIES")?.unwrap_or(DEFAULT_STATE_CACHE_ENTRIES);
        let state_cache_bytes =
            env_usize("DECIDRA_STATE_CACHE_BYTES")?.unwrap_or(DEFAULT_STATE_CACHE_BYTES);
        let host_state_cache_bytes =
            env_usize("DECIDRA_HOST_STATE_CACHE_BYTES")?.unwrap_or(DEFAULT_HOST_STATE_CACHE_BYTES);

        Ok(Self {
            bind_addr,
            model_id,
            checkpoint,
            tokenizer,
            native_model,
            native_cache,
            native_marlin,
            native_fused_w4a16,
            native_gdn_chunked,
            native_gdn_chunked_min_seq,
            native_int8,
            native_int8_clip,
            native_int8_group_scale,
            native_int8_group_exclude,
            native_int8_exclude,
            native_int8_all_layers,
            native_gpu_embed,
            metrics_enabled,
            max_body_bytes,
            max_queue,
            batch_requests,
            batch_rows,
            batch_window_us,
            state_cache_entries,
            state_cache_bytes,
            host_state_cache_bytes,
        })
    }
}

fn list_env(name: &str, default: &str) -> Vec<String> {
    env_string(name)
        .unwrap_or_else(|| default.to_owned())
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn env_string(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(value) if !value.is_empty() => Some(value),
        _ => None,
    }
}

fn env_u64(name: &str) -> Result<Option<u64>> {
    match env_string(name) {
        Some(value) => {
            Ok(Some(value.parse::<u64>().with_context(|| {
                format!("{name} must be an unsigned integer")
            })?))
        }
        None => Ok(None),
    }
}

fn env_usize(name: &str) -> Result<Option<usize>> {
    match env_string(name) {
        Some(value) => {
            Ok(Some(value.parse::<usize>().with_context(|| {
                format!("{name} must be an unsigned integer")
            })?))
        }
        None => Ok(None),
    }
}

fn env_bool(name: &str) -> Result<Option<bool>> {
    match env_string(name) {
        Some(value) => Ok(Some(match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => bail!("{name} must be one of true/false, 1/0, yes/no, on/off"),
        })),
        None => Ok(None),
    }
}

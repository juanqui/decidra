//! Environment configuration. Every deployment knob is explicit and validated at startup.

use std::{net::SocketAddr, path::PathBuf};

use anyhow::{Context, Result, bail};

pub const DEFAULT_BIND_ADDR: &str = "0.0.0.0:8080";
pub const DEFAULT_MODEL: &str = "jev-awq";
pub const DEFAULT_CHECKPOINT: &str = "/srv/models/jev-awq/awq-w4a16-v2";
pub const DEFAULT_HF_SNAPSHOT: &str = "/srv/models/hf/hub/models--autotrust--JEV-27B-VL/snapshots/ba3f0d584994b37998f235c0a3f6f1beff32ba1e";
pub const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
/// Exact-decision cache entries (`DECIDRA_DECISION_CACHE_SIZE`); 0 disables.
pub const DEFAULT_DECISION_CACHE_SIZE: usize = 256;
/// Measured throughput saturates near batch 8 (~57 req/s at c16 with 73-token
/// prompts). Batches of 12/16 only raised p50, so 8 is the default.
pub const DEFAULT_NATIVE_MAX_BATCH: usize = 8;
/// A lone request uses a 500 us grace; a queued burst uses the full window.
/// 2 ms gave the best burst coalescing without hurting steady throughput.
pub const DEFAULT_NATIVE_BATCH_WINDOW_US: u64 = 2_000;
pub const MAX_NATIVE_BATCH_WINDOW_US: u64 = 50_000;
pub const DEFAULT_NATIVE_MAX_BATCH_TOKENS: usize = 16_384;
pub const MIN_NATIVE_MAX_BATCH_TOKENS: usize = 512;
pub const DEFAULT_SERVED_IDENTITY: &str = "autotrust/JEV-27B-VL@ba3f0d58+awq-w4a16";
/// Module types kept on W4A16 when `DECIDRA_NATIVE_INT8` is enabled. With
/// `last:8` all-linear this leaves only the attention o/v/k projections on
/// W4A16 outside layers 56-63. Verified at batch=1: 28/28, max delta 1.027e-2,
/// MAE 7.166e-4, KL 3.269e-5.
pub const DEFAULT_INT8_EXCLUDE: &str = "self_attn.o_proj,self_attn.v_proj,self_attn.k_proj";
/// Modules that keep the faster per-token INT8 epilogue when group scales are
/// enabled. `mlp.gate_proj` is the largest INT8 module and is accurate enough
/// with one scale per row; keeping it on the int32 epilogue saves ~50 ms at 4K.
pub const DEFAULT_INT8_GROUP_EXCLUDE: &str = "mlp.gate_proj";

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: SocketAddr,
    /// Model id advertised by `/v1/models` (`DECIDRA_MODEL`).
    pub model_id: String,
    pub checkpoint: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub calibration: Option<PathBuf>,
    pub calibration_fitted: Option<PathBuf>,
    pub hf_snapshot: PathBuf,
    /// AWQ artifact opened by the native engine (`DECIDRA_NATIVE_MODEL`).
    pub native_model: PathBuf,
    /// Keep packed decoder weights resident on the GPU (`DECIDRA_NATIVE_CACHE`).
    pub native_cache: bool,
    /// Use the repacked Marlin W4A16 kernel (`DECIDRA_NATIVE_MARLIN`). On by
    /// default; the engine repacks once during `preload_all`.
    pub native_marlin: bool,
    /// Use the fused W4A16 kernel/hybrid policy (`DECIDRA_NATIVE_FUSED_W4A16`).
    /// On by default: the native policy uses fused for `N >= 5120 && M <= 256`
    /// and falls back to dequant + cuBLAS otherwise.
    pub native_fused_w4a16: bool,
    /// Enable the native request batcher (`DECIDRA_NATIVE_BATCH`).
    pub native_batch: bool,
    /// Maximum requests per batch (`DECIDRA_NATIVE_MAX_BATCH`). Effective value is
    /// 1 when batching is disabled.
    pub native_max_batch: usize,
    /// Batch window in microseconds (`DECIDRA_NATIVE_BATCH_WINDOW_US`). Effective
    /// value is 0 when batching is disabled.
    pub native_batch_window_us: u64,
    /// Maximum total prompt tokens per batch (`DECIDRA_NATIVE_MAX_BATCH_TOKENS`).
    /// A single larger request runs alone.
    pub native_max_batch_tokens: usize,
    /// Use the chunked TF32 GDN prefill for sequences >= 64 tokens
    /// (`DECIDRA_NATIVE_GDN_CHUNKED`). Shorter sequences keep the existing
    /// shared/split recurrence.
    pub native_gdn_chunked: bool,
    /// Minimum sequence length for the chunked GDN path
    /// (`DECIDRA_NATIVE_GDN_CHUNKED_MIN_SEQ`, default 64).
    pub native_gdn_chunked_min_seq: usize,
    /// Enable selective W4A8-INT8 activations (`DECIDRA_NATIVE_INT8`, default on;
    /// the excluded modules in `DEFAULT_INT8_EXCLUDE` keep W4A16).
    pub native_int8: bool,
    /// Abs-max clip factor for INT8 activation scales
    /// (`DECIDRA_NATIVE_INT8_CLIP`, default 1.0 = off).
    pub native_int8_clip: f32,
    /// Use per-K-group (64) INT8 activation scales in the W4A8 Marlin path
    /// (`DECIDRA_NATIVE_INT8_GROUP_SCALE`, default on).
    pub native_int8_group_scale: bool,
    /// Module-name substrings that keep per-token INT8 scales
    /// (`DECIDRA_NATIVE_INT8_GROUP_EXCLUDE`).
    pub native_int8_group_exclude: Vec<String>,
    /// Module-name substrings kept on W4A16 (`DECIDRA_NATIVE_INT8_EXCLUDE`).
    pub native_int8_exclude: Vec<String>,
    /// Layers that use all-linear INT8 (`DECIDRA_NATIVE_INT8_ALL_LAYERS`).
    pub native_int8_all_layers: String,
    /// Exact-decision logits cache entries (`DECIDRA_DECISION_CACHE_SIZE`).
    /// Zero disables the cache.
    pub decision_cache_size: usize,
    /// Keep the embedding table resident on the GPU (`DECIDRA_NATIVE_GPU_EMBED`,
    /// default on; costs 2.4 GiB and removes the host row gather).
    pub native_gpu_embed: bool,
    pub signing_key: Option<Vec<u8>>,
    pub metrics_enabled: bool,
    pub max_body_bytes: usize,
    /// Identity reported by the xPNP `/v1/systemone` endpoint.
    pub served_identity: String,
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
        let hf_snapshot = PathBuf::from(
            env_string("DECIDRA_HF_SNAPSHOT").unwrap_or_else(|| DEFAULT_HF_SNAPSHOT.to_owned()),
        );
        let tokenizer = env_string("DECIDRA_TOKENIZER").map(PathBuf::from);
        let calibration = env_string("DECIDRA_CALIBRATION").map(PathBuf::from);
        let calibration_fitted = env_string("DECIDRA_CALIBRATION_FITTED").map(PathBuf::from);

        let native_model = PathBuf::from(
            env_string("DECIDRA_NATIVE_MODEL").unwrap_or_else(|| checkpoint.display().to_string()),
        );
        let native_cache = env_bool("DECIDRA_NATIVE_CACHE")?.unwrap_or(true);
        let native_marlin = env_bool("DECIDRA_NATIVE_MARLIN")?.unwrap_or(true);
        let native_fused_w4a16 = env_bool("DECIDRA_NATIVE_FUSED_W4A16")?.unwrap_or(true);
        let native_batch = env_bool("DECIDRA_NATIVE_BATCH")?.unwrap_or(true);
        let mut native_max_batch =
            env_usize("DECIDRA_NATIVE_MAX_BATCH")?.unwrap_or(DEFAULT_NATIVE_MAX_BATCH);
        if native_max_batch == 0 {
            bail!("DECIDRA_NATIVE_MAX_BATCH must be at least 1");
        }
        let mut native_batch_window_us =
            env_u64("DECIDRA_NATIVE_BATCH_WINDOW_US")?.unwrap_or(DEFAULT_NATIVE_BATCH_WINDOW_US);
        if native_batch_window_us > MAX_NATIVE_BATCH_WINDOW_US {
            bail!(
                "DECIDRA_NATIVE_BATCH_WINDOW_US must be at most {MAX_NATIVE_BATCH_WINDOW_US} (got {native_batch_window_us})"
            );
        }
        let native_max_batch_tokens = env_usize("DECIDRA_NATIVE_MAX_BATCH_TOKENS")?
            .unwrap_or(DEFAULT_NATIVE_MAX_BATCH_TOKENS);
        if native_max_batch_tokens < MIN_NATIVE_MAX_BATCH_TOKENS {
            bail!(
                "DECIDRA_NATIVE_MAX_BATCH_TOKENS must be at least {MIN_NATIVE_MAX_BATCH_TOKENS} (got {native_max_batch_tokens})"
            );
        }
        if !native_batch {
            native_max_batch = 1;
            native_batch_window_us = 0;
        }
        let native_gdn_chunked = env_bool("DECIDRA_NATIVE_GDN_CHUNKED")?.unwrap_or(true);
        let native_gdn_chunked_min_seq =
            env_usize("DECIDRA_NATIVE_GDN_CHUNKED_MIN_SEQ")?.unwrap_or(64);
        let native_int8 = env_bool("DECIDRA_NATIVE_INT8")?.unwrap_or(true);
        let native_int8_clip = match env_string("DECIDRA_NATIVE_INT8_CLIP") {
            Some(v) => v
                .parse::<f32>()
                .context("DECIDRA_NATIVE_INT8_CLIP must be a number in (0, 1]")?,
            None => 1.0,
        };
        if !(native_int8_clip > 0.0 && native_int8_clip <= 1.0) {
            bail!("DECIDRA_NATIVE_INT8_CLIP must be in (0, 1], got {native_int8_clip}");
        }
        let native_int8_group_scale = env_bool("DECIDRA_NATIVE_INT8_GROUP_SCALE")?.unwrap_or(true);
        let native_int8_group_exclude = env_string("DECIDRA_NATIVE_INT8_GROUP_EXCLUDE")
            .unwrap_or_else(|| DEFAULT_INT8_GROUP_EXCLUDE.to_owned())
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        let native_int8_exclude = env_string("DECIDRA_NATIVE_INT8_EXCLUDE")
            .unwrap_or_else(|| DEFAULT_INT8_EXCLUDE.to_owned())
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        let native_int8_all_layers =
            env_string("DECIDRA_NATIVE_INT8_ALL_LAYERS").unwrap_or_else(|| "last:8".to_owned());
        let decision_cache_size =
            env_usize("DECIDRA_DECISION_CACHE_SIZE")?.unwrap_or(DEFAULT_DECISION_CACHE_SIZE);
        let native_gpu_embed = env_bool("DECIDRA_NATIVE_GPU_EMBED")?.unwrap_or(true);

        let signing_key = match std::env::var("DECIDRA_SIGNING_KEY") {
            Ok(key) if !key.is_empty() => Some(key.into_bytes()),
            _ => None,
        };

        let metrics_enabled = env_bool("DECIDRA_METRICS_ENABLED")?.unwrap_or(true);
        let max_body_bytes = env_usize("DECIDRA_MAX_BODY_BYTES")?.unwrap_or(DEFAULT_MAX_BODY_BYTES);
        if max_body_bytes == 0 {
            bail!("DECIDRA_MAX_BODY_BYTES must be greater than 0");
        }
        let served_identity = env_string("DECIDRA_SERVED_IDENTITY")
            .unwrap_or_else(|| DEFAULT_SERVED_IDENTITY.to_owned());

        Ok(Self {
            bind_addr,
            model_id,
            checkpoint,
            tokenizer,
            calibration,
            calibration_fitted,
            hf_snapshot,
            native_model,
            native_cache,
            native_marlin,
            native_fused_w4a16,
            native_batch,
            native_max_batch,
            native_batch_window_us,
            native_max_batch_tokens,
            native_gdn_chunked,
            native_gdn_chunked_min_seq,
            native_int8,
            native_int8_clip,
            native_int8_group_scale,
            native_int8_group_exclude,
            native_int8_exclude,
            native_int8_all_layers,
            decision_cache_size,
            native_gpu_embed,
            signing_key,
            metrics_enabled,
            max_body_bytes,
            served_identity,
        })
    }

    pub fn signing_enabled(&self) -> bool {
        self.signing_key.is_some()
    }
}

fn env_string(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(value) if !value.is_empty() => Some(value),
        _ => None,
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

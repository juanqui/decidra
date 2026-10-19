//! Environment configuration. Every deployment knob is explicit and validated at startup.

use std::{net::SocketAddr, path::PathBuf};

use anyhow::{Context, Result, bail};

pub const DEFAULT_BIND_ADDR: &str = "0.0.0.0:8080";
pub const DEFAULT_MODEL: &str = "jev-awq";
pub const DEFAULT_CHECKPOINT: &str = "/srv/models/jev-awq/awq-w4a16-v2";
pub const DEFAULT_HF_SNAPSHOT: &str = "/srv/models/hf/hub/models--autotrust--JEV-27B-VL/snapshots/ba3f0d584994b37998f235c0a3f6f1beff32ba1e";
pub const DEFAULT_MAX_CONCURRENCY: usize = 8;
pub const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
/// Measured throughput saturates at batch 4 (~2,130-2,180 tok/s on ~71-token
/// requests); 8 is kept so a burst drains in one pass without costing steady
/// throughput. Batches of 12/16 only raised p50 in the sweep.
pub const DEFAULT_NATIVE_MAX_BATCH: usize = 8;
/// The smallest window that keeps a lone request near 62 ms p50. Windows of
/// 5/10 ms fill 8-wide batches under load but did not improve throughput.
pub const DEFAULT_NATIVE_BATCH_WINDOW_US: u64 = 2_000;
pub const MAX_NATIVE_BATCH_WINDOW_US: u64 = 50_000;
pub const DEFAULT_NATIVE_MAX_BATCH_TOKENS: usize = 16_384;
pub const MIN_NATIVE_MAX_BATCH_TOKENS: usize = 512;
pub const DEFAULT_SERVED_IDENTITY: &str = "autotrust/JEV-27B-VL@ba3f0d58+awq-w4a16";

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: SocketAddr,
    /// Model id advertised by `/v1/models` (`JEV_MODEL`).
    pub model_id: String,
    pub checkpoint: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub calibration: Option<PathBuf>,
    pub calibration_fitted: Option<PathBuf>,
    pub hf_snapshot: PathBuf,
    /// AWQ artifact opened by the native engine (`JEV_NATIVE_MODEL`).
    pub native_model: PathBuf,
    /// Keep packed decoder weights resident on the GPU (`JEV_NATIVE_CACHE`).
    pub native_cache: bool,
    /// Use the repacked Marlin W4A16 kernel (`JEV_NATIVE_MARLIN`). On by
    /// default; the engine repacks once during `preload_all`.
    pub native_marlin: bool,
    /// Use the fused W4A16 kernel/hybrid policy (`JEV_NATIVE_FUSED_W4A16`).
    /// On by default: the native policy uses fused for `N >= 5120 && M <= 256`
    /// and falls back to dequant + cuBLAS otherwise.
    pub native_fused_w4a16: bool,
    /// Enable the native request batcher (`JEV_NATIVE_BATCH`).
    pub native_batch: bool,
    /// Maximum requests per batch (`JEV_NATIVE_MAX_BATCH`). Effective value is
    /// 1 when batching is disabled.
    pub native_max_batch: usize,
    /// Batch window in microseconds (`JEV_NATIVE_BATCH_WINDOW_US`). Effective
    /// value is 0 when batching is disabled.
    pub native_batch_window_us: u64,
    /// Maximum total prompt tokens per batch (`JEV_NATIVE_MAX_BATCH_TOKENS`).
    /// A single larger request runs alone.
    pub native_max_batch_tokens: usize,
    pub max_concurrency: usize,
    pub signing_key: Option<Vec<u8>>,
    pub metrics_enabled: bool,
    pub max_body_bytes: usize,
    /// Identity reported by the xPNP `/v1/systemone` endpoint.
    pub served_identity: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let bind_addr = env_string("JEV_BIND_ADDR")
            .unwrap_or_else(|| DEFAULT_BIND_ADDR.to_owned())
            .parse::<SocketAddr>()
            .context("JEV_BIND_ADDR must be a socket address such as 0.0.0.0:8080")?;

        let model_id = env_string("JEV_MODEL").unwrap_or_else(|| DEFAULT_MODEL.to_owned());

        let checkpoint = PathBuf::from(
            env_string("JEV_CHECKPOINT").unwrap_or_else(|| DEFAULT_CHECKPOINT.to_owned()),
        );
        let hf_snapshot = PathBuf::from(
            env_string("JEV_HF_SNAPSHOT").unwrap_or_else(|| DEFAULT_HF_SNAPSHOT.to_owned()),
        );
        let tokenizer = env_string("JEV_TOKENIZER").map(PathBuf::from);
        let calibration = env_string("JEV_CALIBRATION").map(PathBuf::from);
        let calibration_fitted = env_string("JEV_CALIBRATION_FITTED").map(PathBuf::from);

        let native_model = PathBuf::from(
            env_string("JEV_NATIVE_MODEL").unwrap_or_else(|| checkpoint.display().to_string()),
        );
        let native_cache = env_bool("JEV_NATIVE_CACHE")?.unwrap_or(true);
        let native_marlin = env_bool("JEV_NATIVE_MARLIN")?.unwrap_or(true);
        let native_fused_w4a16 = env_bool("JEV_NATIVE_FUSED_W4A16")?.unwrap_or(true);
        let native_batch = env_bool("JEV_NATIVE_BATCH")?.unwrap_or(true);
        let mut native_max_batch =
            env_usize("JEV_NATIVE_MAX_BATCH")?.unwrap_or(DEFAULT_NATIVE_MAX_BATCH);
        if native_max_batch == 0 {
            bail!("JEV_NATIVE_MAX_BATCH must be at least 1");
        }
        let mut native_batch_window_us =
            env_u64("JEV_NATIVE_BATCH_WINDOW_US")?.unwrap_or(DEFAULT_NATIVE_BATCH_WINDOW_US);
        if native_batch_window_us > MAX_NATIVE_BATCH_WINDOW_US {
            bail!(
                "JEV_NATIVE_BATCH_WINDOW_US must be at most {MAX_NATIVE_BATCH_WINDOW_US} (got {native_batch_window_us})"
            );
        }
        let native_max_batch_tokens =
            env_usize("JEV_NATIVE_MAX_BATCH_TOKENS")?.unwrap_or(DEFAULT_NATIVE_MAX_BATCH_TOKENS);
        if native_max_batch_tokens < MIN_NATIVE_MAX_BATCH_TOKENS {
            bail!(
                "JEV_NATIVE_MAX_BATCH_TOKENS must be at least {MIN_NATIVE_MAX_BATCH_TOKENS} (got {native_max_batch_tokens})"
            );
        }
        if !native_batch {
            native_max_batch = 1;
            native_batch_window_us = 0;
        }

        let max_concurrency = env_usize("JEV_MAX_CONCURRENCY")?.unwrap_or(DEFAULT_MAX_CONCURRENCY);
        if max_concurrency == 0 {
            bail!("JEV_MAX_CONCURRENCY must be at least 1");
        }

        let signing_key = match std::env::var("JEV_SIGNING_KEY") {
            Ok(key) if !key.is_empty() => Some(key.into_bytes()),
            _ => None,
        };

        let metrics_enabled = env_bool("JEV_METRICS_ENABLED")?.unwrap_or(true);
        let max_body_bytes = env_usize("JEV_MAX_BODY_BYTES")?.unwrap_or(DEFAULT_MAX_BODY_BYTES);
        if max_body_bytes == 0 {
            bail!("JEV_MAX_BODY_BYTES must be greater than 0");
        }
        let served_identity =
            env_string("JEV_SERVED_IDENTITY").unwrap_or_else(|| DEFAULT_SERVED_IDENTITY.to_owned());

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
            max_concurrency,
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

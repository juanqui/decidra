//! Environment configuration. Every deployment knob is explicit and validated at startup.

use std::{net::SocketAddr, path::PathBuf, time::Duration};

use anyhow::{Context, Result, bail};

pub const DEFAULT_BIND_ADDR: &str = "0.0.0.0:8080";
pub const DEFAULT_VLLM_URL: &str = "http://127.0.0.1:8002";
pub const DEFAULT_MODEL: &str = "jev-awq";
pub const DEFAULT_CHECKPOINT: &str = "/srv/models/jev-awq/awq-w4a16-v2";
pub const DEFAULT_HF_SNAPSHOT: &str = "/srv/models/hf/hub/models--autotrust--JEV-27B-VL/snapshots/ba3f0d584994b37998f235c0a3f6f1beff32ba1e";
pub const DEFAULT_MAX_CONCURRENCY: usize = 8;
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 600;
pub const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;
pub const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: SocketAddr,
    pub vllm_url: String,
    pub vllm_model: String,
    pub checkpoint: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub calibration: Option<PathBuf>,
    pub calibration_fitted: Option<PathBuf>,
    pub hf_snapshot: PathBuf,
    pub max_concurrency: usize,
    pub request_timeout: Duration,
    pub connect_timeout: Duration,
    pub signing_key: Option<Vec<u8>>,
    pub metrics_enabled: bool,
    pub max_body_bytes: usize,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let bind_addr = env_string("JEV_BIND_ADDR")
            .unwrap_or_else(|| DEFAULT_BIND_ADDR.to_owned())
            .parse::<SocketAddr>()
            .context("JEV_BIND_ADDR must be a socket address such as 0.0.0.0:8080")?;

        let vllm_url = env_string("JEV_VLLM_URL")
            .unwrap_or_else(|| DEFAULT_VLLM_URL.to_owned())
            .trim_end_matches('/')
            .to_owned();
        validate_url(&vllm_url)?;

        let vllm_model = env_string("JEV_VLLM_MODEL")
            .or_else(|| env_string("JEV_MODEL"))
            .unwrap_or_else(|| DEFAULT_MODEL.to_owned());

        let checkpoint = PathBuf::from(
            env_string("JEV_CHECKPOINT").unwrap_or_else(|| DEFAULT_CHECKPOINT.to_owned()),
        );
        let hf_snapshot = PathBuf::from(
            env_string("JEV_HF_SNAPSHOT").unwrap_or_else(|| DEFAULT_HF_SNAPSHOT.to_owned()),
        );
        let tokenizer = env_string("JEV_TOKENIZER").map(PathBuf::from);
        let calibration = env_string("JEV_CALIBRATION").map(PathBuf::from);
        let calibration_fitted = env_string("JEV_CALIBRATION_FITTED").map(PathBuf::from);

        let max_concurrency = env_usize("JEV_MAX_CONCURRENCY")?.unwrap_or(DEFAULT_MAX_CONCURRENCY);
        if max_concurrency == 0 {
            bail!("JEV_MAX_CONCURRENCY must be at least 1");
        }
        let request_timeout = Duration::from_secs(
            env_u64("JEV_REQUEST_TIMEOUT_SECS")?.unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS),
        );
        if request_timeout.is_zero() {
            bail!("JEV_REQUEST_TIMEOUT_SECS must be greater than 0");
        }
        let connect_timeout = Duration::from_secs(
            env_u64("JEV_CONNECT_TIMEOUT_SECS")?.unwrap_or(DEFAULT_CONNECT_TIMEOUT_SECS),
        );
        if connect_timeout.is_zero() {
            bail!("JEV_CONNECT_TIMEOUT_SECS must be greater than 0");
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

        Ok(Self {
            bind_addr,
            vllm_url,
            vllm_model,
            checkpoint,
            tokenizer,
            calibration,
            calibration_fitted,
            hf_snapshot,
            max_concurrency,
            request_timeout,
            connect_timeout,
            signing_key,
            metrics_enabled,
            max_body_bytes,
        })
    }

    pub fn signing_enabled(&self) -> bool {
        self.signing_key.is_some()
    }
}

fn validate_url(url: &str) -> Result<()> {
    let parsed = reqwest::Url::parse(url)
        .with_context(|| format!("JEV_VLLM_URL is not a valid URL: {url}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        bail!(
            "JEV_VLLM_URL must use http or https, got {}",
            parsed.scheme()
        );
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        bail!("JEV_VLLM_URL must not contain a query or fragment");
    }
    Ok(())
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

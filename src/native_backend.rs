//! In-process backend: the native CUDA engine (`jev-native`) and the existing
//! `tokenizers` tokenizer. No HTTP hop, no Python, no external model server.
//!
//! `jev_native::Engine` is `Send` but owns one CUDA context/stream and is not
//! safe for concurrent forwards. Calls are serialized through a mutex and moved
//! to `spawn_blocking`, so the async reactor is never stalled by a forward.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::{
    backend::{Backend, LogitsFuture},
    config::Config,
    error::EngineError,
    prompt,
};

pub struct NativeBackend {
    tokenizer: Tokenizer,
    engine: Arc<Mutex<jev_native::Engine>>,
}

impl NativeBackend {
    /// Open the checkpoint, optionally upload every decoder layer once, and
    /// keep the packed W4A16 weights resident on the selected GPU.
    ///
    /// The device is chosen before process start with `CUDA_VISIBLE_DEVICES`
    /// (the native engine opens ordinal `0` of the visible set).
    pub fn load(config: &Config) -> Result<Self> {
        let (tokenizer, tokenizer_path) = prompt::load_tokenizer(
            &config.checkpoint,
            config.tokenizer.as_deref(),
            &config.hf_snapshot,
        )?;
        let started = Instant::now();
        let engine = jev_native::Engine::new(&config.native_model).with_context(|| {
            format!(
                "opening native engine checkpoint {}",
                config.native_model.display()
            )
        })?;
        // Must be set before preload_all: Marlin repacks into its resident layout.
        engine.set_marlin(config.native_marlin);
        engine.set_cache_weights(config.native_cache);
        if !config.native_marlin {
            engine.set_fused_w4a16(config.native_fused_w4a16);
        }
        if config.native_cache {
            engine.preload_all().with_context(|| {
                format!(
                    "preloading resident weights from {}",
                    config.native_model.display()
                )
            })?;
        }
        tracing::info!(
            model = %config.native_model.display(),
            tokenizer = %tokenizer_path.display(),
            cache = config.native_cache,
            marlin = config.native_marlin,
            fused_w4a16 = config.native_fused_w4a16 && !config.native_marlin,
            hidden = engine.cfg.hidden_size,
            layers = engine.cfg.num_hidden_layers,
            elapsed_secs = started.elapsed().as_secs_f64(),
            "native engine ready"
        );
        Ok(Self {
            tokenizer,
            engine: Arc::new(Mutex::new(engine)),
        })
    }

    fn encode(&self, prompt: &str) -> Result<Vec<u32>, EngineError> {
        self.tokenizer
            .encode(prompt, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| EngineError::Config(format!("tokenizing prompt: {error}")))
    }
}

impl Backend for NativeBackend {
    fn logits<'a>(&'a self, prompt: &'a str, ids: &'a [u32]) -> LogitsFuture<'a> {
        Box::pin(async move {
            let tokens = self.encode(prompt)?;
            let ids = ids.to_vec();
            let expected = ids.len();
            let engine = self.engine.clone();
            let started = Instant::now();
            let result = tokio::task::spawn_blocking(move || {
                let engine = engine.lock().map_err(|_| {
                    EngineError::Internal("native engine mutex poisoned".to_owned())
                })?;
                engine
                    .decision_logits(&tokens, &ids)
                    .map_err(|error| EngineError::Backend(error.to_string()))
            })
            .await
            .map_err(|error| {
                EngineError::Internal(format!("native forward task failed: {error}"))
            })?;
            metrics::histogram!("jev_backend_forward_seconds")
                .record(started.elapsed().as_secs_f64());
            match &result {
                Ok(_) => {
                    metrics::counter!("jev_backend_requests_total", "status" => "ok").increment(1)
                }
                Err(_) => metrics::counter!("jev_backend_requests_total", "status" => "error")
                    .increment(1),
            }
            let logits = result?;
            if logits.len() != expected {
                return Err(EngineError::BadResponse(format!(
                    "native engine returned {} logits for {} requested ids",
                    logits.len(),
                    expected
                )));
            }
            Ok(logits)
        })
    }

    fn prompt_tokens(&self, prompt: &str) -> u64 {
        self.encode(prompt)
            .map(|tokens| tokens.len() as u64)
            .unwrap_or(0)
    }

    fn name(&self) -> &str {
        "native"
    }
}

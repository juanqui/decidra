//! In-process backend: the native CUDA engine (`decidra-native`) and the existing
//! `tokenizers` tokenizer. No HTTP hop, no Python, no external model server.
//!
//! `decidra_native::Engine` is `Send` but owns one CUDA context/stream and is not
//! safe for concurrent forwards. It is moved into a dedicated worker thread
//! (see [`crate::batch`]); requests are tokenized on the async reactor, queued,
//! and executed in small FIFO batches. The reactor never blocks on a forward.

use std::time::Instant;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::{
    backend::{Backend, LogitsFuture},
    batch::{BatchExecutor, BatchHandle, BatchOptions, ExecResult},
    cache::DecisionCache,
    config::Config,
    error::EngineError,
    prompt,
};

/// Owns the engine on the batch worker thread.
struct EngineExecutor {
    engine: decidra_native::Engine,
    /// Whether the queue may submit multi-request batches. When false the
    /// queue is configured for single requests anyway; this flag also lets the
    /// adapter take the single-call path explicitly.
    batch_enabled: bool,
}

impl BatchExecutor for EngineExecutor {
    fn execute(&mut self, tokens: &[&[u32]], ids: &[&[u32]]) -> Vec<ExecResult> {
        if tokens.len() != ids.len() {
            return vec![
                Err(format!(
                    "batch has {} token sequences but {} id sets",
                    tokens.len(),
                    ids.len()
                ));
                tokens.len()
            ];
        }
        // A one-item batch uses the single-forward path for short prompts
        // (lower launch overhead); long prompts use the batch API, which has
        // the lower long-context peak with FlashInfer.
        let long_single = tokens.len() == 1 && tokens[0].len() >= 1024;
        if !self.batch_enabled || (tokens.len() == 1 && !long_single) {
            // Single-call path: used only when batching is disabled. One-item
            // batches use the batch API as well; it is documented as
            // bit-identical and has a lower long-context peak (FlashInfer's
            // workspace) than the single forward path.
            return tokens
                .iter()
                .zip(ids)
                .map(|(tokens, ids)| {
                    self.engine
                        .decision_logits(tokens, ids)
                        .map_err(|error| error.to_string())
                })
                .collect();
        }
        match self.engine.decision_logits_batch(tokens, ids) {
            Ok(results) if results.len() == tokens.len() => results
                .into_iter()
                .zip(ids)
                .map(|(logits, ids)| {
                    if logits.len() == ids.len() {
                        Ok(logits)
                    } else {
                        Err(format!(
                            "native batch returned {} logits for {} requested ids",
                            logits.len(),
                            ids.len()
                        ))
                    }
                })
                .collect(),
            Ok(results) => vec![
                Err(format!(
                    "native batch returned {} results for {} requests",
                    results.len(),
                    tokens.len()
                ));
                tokens.len()
            ],
            Err(error) => vec![Err(error.to_string()); tokens.len()],
        }
    }
}

pub struct NativeBackend {
    tokenizer: Tokenizer,
    queue: BatchHandle,
    /// Exact `(tokens, head ids)` logits cache. Capacity 0 disables it.
    cache: DecisionCache,
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
        let engine = decidra_native::Engine::new(&config.native_model).with_context(|| {
            format!(
                "opening native engine checkpoint {}",
                config.native_model.display()
            )
        })?;
        // Must be set before preload_all: Marlin repacks into its resident layout.
        engine.set_marlin(config.native_marlin);
        engine.set_cache_weights(config.native_cache);
        engine.set_gpu_embed(config.native_gpu_embed);
        // Chunked prefill engages at >=64 tokens; shorter prompts keep the
        // shared/split recurrence. Both flags must be set before preload_all.
        engine.set_gdn_chunked(config.native_gdn_chunked);
        engine.set_gdn_chunked_tc(config.native_gdn_chunked);
        engine.set_gdn_chunked_min_seq(config.native_gdn_chunked_min_seq);
        engine.set_marlin_int8(config.native_int8);
        engine.set_int8_clip(config.native_int8_clip);
        engine.set_int8_group_scale(config.native_int8_group_scale);
        engine.set_int8_group_exclude(config.native_int8_group_exclude.clone());
        engine.set_int8_exclude(config.native_int8_exclude.clone());
        engine.set_int8_all_layers(&config.native_int8_all_layers);
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
            gpu_embed = config.native_gpu_embed,
            marlin = config.native_marlin,
            fused_w4a16 = config.native_fused_w4a16 && !config.native_marlin,
            hidden = engine.cfg.hidden_size,
            layers = engine.cfg.num_hidden_layers,
            batching = config.native_batch,
            max_batch = config.native_max_batch,
            batch_window_us = config.native_batch_window_us,
            max_batch_tokens = config.native_max_batch_tokens,
            gdn_chunked = config.native_gdn_chunked,
            gdn_chunked_min_seq = config.native_gdn_chunked_min_seq,
            int8 = config.native_int8,
            int8_clip = config.native_int8_clip,
            int8_group_scale = config.native_int8_group_scale,
            int8_group_exclude = config.native_int8_group_exclude.len(),
            int8_all_layers = config.native_int8_all_layers.as_str(),
            decision_cache_entries = config.decision_cache_size,
            elapsed_secs = started.elapsed().as_secs_f64(),
            "native engine ready"
        );
        let options = if config.native_batch {
            BatchOptions::new(
                config.native_max_batch,
                config.native_max_batch_tokens,
                std::time::Duration::from_micros(config.native_batch_window_us),
            )
        } else {
            BatchOptions::single(config.native_max_batch_tokens)
        };
        let executor = EngineExecutor {
            engine,
            batch_enabled: config.native_batch,
        };
        Ok(Self {
            tokenizer,
            queue: BatchHandle::spawn(executor, options),
            cache: DecisionCache::new(config.decision_cache_size),
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
            let started = Instant::now();
            let tokens = self.encode(prompt)?;
            let ids = ids.to_vec();
            let expected = ids.len();
            // Look up before enqueueing so a hit never waits on the batch worker.
            // The key is the exact token/head-id pair; no prefix matching.
            let cache_key = if self.cache.enabled() {
                if let Some(logits) = self.cache.get(&tokens, &ids) {
                    metrics::counter!("decidra_decision_cache_hits_total").increment(1);
                    return Ok(logits);
                }
                metrics::counter!("decidra_decision_cache_misses_total").increment(1);
                Some((tokens.clone(), ids.clone()))
            } else {
                None
            };
            let receiver = self.queue.submit(tokens, ids)?;
            let result = receiver.await.map_err(|_| {
                EngineError::Internal("native batch worker dropped the request".to_owned())
            })?;
            metrics::histogram!("decidra_backend_forward_seconds")
                .record(started.elapsed().as_secs_f64());
            match &result {
                Ok(_) => metrics::counter!("decidra_backend_requests_total", "status" => "ok")
                    .increment(1),
                Err(_) => metrics::counter!("decidra_backend_requests_total", "status" => "error")
                    .increment(1),
            }
            let logits = result.map_err(EngineError::Backend)?;
            if logits.len() != expected {
                return Err(EngineError::BadResponse(format!(
                    "native engine returned {} logits for {} requested ids",
                    logits.len(),
                    expected
                )));
            }
            // Only successful, length-checked logits are cached.
            if let Some((key_tokens, key_ids)) = cache_key {
                self.cache.insert(&key_tokens, &key_ids, logits.clone());
                metrics::gauge!("decidra_decision_cache_entries").set(self.cache.len() as f64);
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

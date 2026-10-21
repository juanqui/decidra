//! In-process backend: the native CUDA engine (`decidra-native`) and the existing
//! `tokenizers` tokenizer. No HTTP hop, no Python, no external model server.
//!
//! `decidra_native::Engine` is `Send` but owns one CUDA context/stream and is not
//! safe for concurrent forwards. It is moved into a dedicated worker thread
//! (see [`crate::batch`]); requests are tokenized on the async reactor, queued,
//! and executed in small FIFO batches. The reactor never blocks on a forward.

use std::{pin::Pin, time::Instant};

use serde_json::{Map, Value, json};

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
    fn execute_state_once(
        &mut self,
        state: &[u32],
        questions: &[Vec<u32>],
        _codes: &[u32],
    ) -> Result<Vec<Vec<f32>>, String> {
        let cache = self
            .engine
            .capture_state(state)
            .map_err(|error| error.to_string())?;
        let views: Vec<&[u32]> = questions.iter().map(|q| q.as_slice()).collect();
        self.engine
            .readout_state_once_batch(&cache, &views)
            .map_err(|error| error.to_string())
    }

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

#[derive(Clone)]
struct D3Config {
    codes: Vec<String>,
    temperature: f64,
}

pub struct NativeBackend {
    tokenizer: Tokenizer,
    queue: BatchHandle,
    /// Exact `(tokens, head ids)` logits cache. Capacity 0 disables it.
    cache: DecisionCache,
    d3: Option<D3Config>,
}

fn load_d3_config(dir: &std::path::Path) -> Result<Option<D3Config>> {
    let path = dir.join("decision_config.json");
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    let value: Value = serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    if value.get("attention_mode").and_then(Value::as_str) != Some("prefix_bidir") {
        return Ok(None);
    }
    let codes = value
        .get("codes")
        .and_then(Value::as_array)
        .context("decision_config.json codes")?
        .iter()
        .map(|code| code.as_str().unwrap_or_default().to_owned())
        .collect::<Vec<_>>();
    let temperature = value
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    Ok(Some(D3Config { codes, temperature }))
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
        let d3 = load_d3_config(&config.native_model)?;
        if std::env::var("DECIDRA_D3").map(|v| v != "0").unwrap_or(false) && d3.is_none() {
            anyhow::bail!(
                "DECIDRA_D3 is set but {} has no prefix_bidir decision_config.json",
                config.native_model.display()
            );
        }
        if let Some(d3) = &d3 {
            // Startup self-test through the exact state-once path. A broken
            // mask, artifact, or FlashInfer build must fail here, not on a
            // production request.
            let question = crate::d3::noul_question("self test", None);
            let text = crate::d3::render(&json!("ok"), &question, &d3.codes)?;
            let (ids, split) = crate::d3::encode_split(&tokenizer, &text, &json!("ok"))?;
            let cache = engine
                .capture_state(&ids[..split])
                .context("d3 startup state capture")?;
            let rows = engine
                .readout_state_once_batch(&cache, &[&ids[split..]])
                .context("d3 startup question pass")?;
            if rows.first().map_or(true, |row| row.iter().any(|v| !v.is_finite())) {
                anyhow::bail!("d3 startup self-test returned non-finite logits");
            }
            let codes: Vec<u32> = (0..255).collect();
            let reference = engine
                .decision_logits_split(&ids, split, &codes)
                .context("d3 startup per-row block reference")?;
            let softmax = |values: &[f32]| -> Vec<f32> {
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut out: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
                let total: f32 = out.iter().sum();
                for v in &mut out {
                    *v /= total;
                }
                out
            };
            let state_once = softmax(&rows[0]);
            let block = softmax(&reference);
            let delta = state_once
                .iter()
                .zip(&block)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            // W4A8 Marlin picks a different split-K shape for the two paths;
            // the smoke gate catches a mask/cache reversal, not quantization
            // noise (the offline EVAL gate covers weights).
            if delta > 0.12 {
                anyhow::bail!(
                    "d3 startup self-test disagrees with the per-row block path by {delta}"
                );
            }
            let yes = softmax(&rows[0][..2])[1];
            // BF16 reference for this fixed prompt is 0.803; the W4A8 artifact
            // lands at 0.735. A random-init or wrong-mask artifact leaves this
            // band. The offline EVAL gate remains the quality authority.
            if !(0.65..=0.85).contains(&yes) {
                anyhow::bail!("d3 startup self-test noul probability {yes} is out of band");
            }
            tracing::info!(delta, yes, "d3 state-once self-test passed");
        }
        let executor = EngineExecutor {
            engine,
            batch_enabled: config.native_batch,
        };
        Ok(Self {
            tokenizer,
            queue: BatchHandle::spawn(executor, options),
            cache: DecisionCache::new(config.decision_cache_size),
            d3,
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

    fn d3_enabled(&self) -> bool {
        self.d3.is_some()
    }

    fn d3_answers<'a>(
        &'a self,
        state: &'a Value,
        questions: &'a Map<String, Value>,
    ) -> Pin<Box<dyn Future<Output = Result<Value, EngineError>> + Send + 'a>> {
        Box::pin(async move {
            let config = self
                .d3
                .as_ref()
                .ok_or_else(|| EngineError::Unsupported("d3 is not configured".to_owned()))?;
            if questions.is_empty() {
                return Err(EngineError::Config("questions must not be empty".to_owned()));
            }
            if questions.len() > 64 {
                return Err(EngineError::Config(
                    "at most 64 questions per state-once request".to_owned(),
                ));
            }
            if crate::d3::contains_media(state) {
                return Err(EngineError::Unsupported(
                    "image and video states are not supported by the native d3 path".to_owned(),
                ));
            }
            if serde_json::to_string(state).map_or(true, |text| text.len() > (1 << 20)) {
                return Err(EngineError::Config(
                    "state is over the 1 MiB rendering cap".to_owned(),
                ));
            }
            let mut state_ids: Option<Vec<u32>> = None;
            let mut blocks = Vec::with_capacity(questions.len());
            let mut counts = Vec::with_capacity(questions.len());
            for (name, question) in questions {
                let parsed = crate::d3::parse_question(question)
                    .map_err(|error| EngineError::Config(format!("question {name}: {error}")))?;
                let text = crate::d3::render(state, &parsed, &config.codes)
                    .map_err(|error| EngineError::Config(format!("question {name}: {error}")))?;
                if text.len() > (1 << 20) {
                    return Err(EngineError::Config(format!(
                        "question {name} renders over the 1 MiB cap"
                    )));
                }
                let (ids, split) = crate::d3::encode_split(&self.tokenizer, &text, state)
                    .map_err(|error| EngineError::Config(format!("question {name}: {error}")))?;
                if split > 32_768 || ids.len() > 32_768 {
                    return Err(EngineError::Config(format!(
                        "question {name} has {split}+{} tokens, over the 32K serving cap",
                        ids.len() - split
                    )));
                }
                let head = ids[..split].to_vec();
                match &state_ids {
                    Some(existing) if *existing != head => {
                        return Err(EngineError::BadResponse(
                            "the tokenized state differs across questions".to_owned(),
                        ));
                    }
                    None => state_ids = Some(head),
                    _ => {}
                }
                blocks.push(ids[split..].to_vec());
                counts.push(parsed.keys.len());
            }
            let state_ids = state_ids.expect("at least one question");
            let usage_tokens = state_ids.len() as u64
                + blocks.iter().map(|block| block.len() as u64).sum::<u64>();
            let codes: Vec<u32> = (0..255).collect();
            let receiver = self
                .queue
                .submit_state_once(state_ids.clone(), blocks.clone(), codes)
                .map_err(|error| EngineError::Internal(error.to_string()))?;
            let rows = receiver
                .await
                .map_err(|_| EngineError::Internal("native worker dropped the d3 reply".to_owned()))?
                .map_err(EngineError::Backend)?;
            if rows.len() != blocks.len() {
                return Err(EngineError::BadResponse(format!(
                    "native engine returned {} rows for {} questions",
                    rows.len(),
                    blocks.len()
                )));
            }
            let mut out = Map::new();
            for ((name, count), row) in questions.keys().zip(counts).zip(rows) {
                if row.len() < count {
                    return Err(EngineError::BadResponse(
                        "readout returned fewer logits than options".to_owned(),
                    ));
                }
                let scaled: Vec<f32> = row[..count]
                    .iter()
                    .map(|value| (value / config.temperature as f32).exp())
                    .collect();
                let total: f32 = scaled.iter().sum();
                if !(total > 0.0) || !total.is_finite() {
                    return Err(EngineError::BadResponse("non-finite d3 logits".to_owned()));
                }
                let probabilities: Vec<f32> = scaled.iter().map(|value| value / total).collect();
                let question = crate::d3::parse_question(&questions[name])
                    .map_err(|error| EngineError::Internal(error.to_string()))?;
                let answer = crate::d3::to_answer(&question, &probabilities)
                    .map_err(|error| EngineError::Internal(error.to_string()))?;
                out.insert(name.clone(), answer);
            }
            Ok(json!({
                "answers": out,
                "usage": {"input_tokens": usage_tokens, "output_tokens": 0},
            }))
        })
    }
}

//! In-process d3-turbo backend: the native CUDA engine, the checkpoint
//! tokenizer, the `prefix_bidir` plan, and the separate F32 readout.
//!
//! The engine owns one CUDA stream on the selected device and is moved to the
//! dedicated worker in [`crate::batch`]. The HTTP reactor tokenizes and builds
//! question blocks; the worker captures the state once and answers every
//! question from that cache.

use std::{path::Path, sync::Arc, time::Instant};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use decidra_native::Engine;

use crate::{
    backend::{Backend, D3Future},
    batch::BatchHandle,
    config::{Config, MAX_QUESTIONS, MAX_STATE_TOKENS},
    d3,
    d3::ImageLayout,
    error::EngineError,
    image_media::{ImageMedia, MAX_IMAGE_TOKENS, MAX_IMAGES},
    telemetry::{self, EngineInfo, GpuInfo},
};

#[derive(Clone)]
struct D3Config {
    codes: Vec<String>,
    temperature: f64,
}

pub struct NativeBackend {
    tokenizer: Arc<Tokenizer>,
    queue: BatchHandle,
    d3: D3Config,
    supports_vision: bool,
    image_slots: Arc<tokio::sync::Semaphore>,
}

impl NativeBackend {
    /// Open the d3-turbo checkpoint, preload the resident W4A8 weights, and
    /// run the exact state-once self-test before serving.
    pub fn load(config: &Config) -> Result<Self> {
        let d3 = load_d3_config(&config.native_model)?;
        let readout = config.native_model.join("readout.safetensors");
        if !readout.is_file() {
            bail!(
                "{} has no readout.safetensors; only d3-turbo prefix_bidir packages are supported",
                config.native_model.display()
            );
        }
        let tokenizer_path = config
            .tokenizer
            .clone()
            .unwrap_or_else(|| config.checkpoint.join("tokenizer.json"));
        let tokenizer =
            Arc::new(Tokenizer::from_file(&tokenizer_path).map_err(|error| {
                anyhow::anyhow!("loading {}: {error}", tokenizer_path.display())
            })?);

        let started = Instant::now();
        let engine = Engine::new(&config.native_model).with_context(|| {
            format!(
                "opening native d3-turbo checkpoint {}",
                config.native_model.display()
            )
        })?;
        if !matches!(
            engine.cfg.attention_mode,
            decidra_native::config::AttentionMode::BlockCascade
        ) {
            bail!(
                "{} is not a d3-turbo prefix_bidir checkpoint (attention_mode {:?})",
                config.native_model.display(),
                engine.cfg.attention_mode
            );
        }
        // All flags must be set before preload: Marlin repacks in place.
        engine.set_marlin(config.native_marlin);
        engine.set_cache_weights(config.native_cache);
        engine.set_gpu_embed(config.native_gpu_embed);
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
                    "preloading d3-turbo weights from {}",
                    config.native_model.display()
                )
            })?;
        }
        tracing::info!(
            model = %config.native_model.display(),
            tokenizer = %tokenizer_path.display(),
            arch = engine.gpu.arch.label(),
            sms = engine.gpu.sms,
            cache = config.native_cache,
            gpu_embed = config.native_gpu_embed,
            marlin = config.native_marlin,
            int8 = config.native_int8,
            int8_group_scale = config.native_int8_group_scale,
            int8_exclude = config.native_int8_exclude.len(),
            int8_all_layers = config.native_int8_all_layers.as_str(),
            gdn_chunked = config.native_gdn_chunked,
            gdn_chunked_min_seq = config.native_gdn_chunked_min_seq,
            hidden = engine.cfg.hidden_size,
            layers = engine.cfg.num_hidden_layers,
            elapsed_secs = started.elapsed().as_secs_f64(),
            "d3-turbo engine ready"
        );

        self_test(&engine, &tokenizer, &d3)?;
        let supports_vision = engine.supports_vision();
        let gpu_name = match engine.gpu.device_name() {
            Ok(name) => name,
            Err(error) => {
                tracing::warn!(%error, "metrics: failed to read GPU name");
                "unknown".to_owned()
            }
        };
        let compute_capability = match engine.gpu.compute_capability() {
            Ok((major, minor)) => format!("{major}.{minor}"),
            Err(error) => {
                tracing::warn!(%error, "metrics: failed to read compute capability");
                "unknown".to_owned()
            }
        };
        telemetry::record_engine_info(
            &EngineInfo {
                attention_mode: "prefix_bidir",
                quantization: quantization_label(config),
                vision: supports_vision,
                gpu_embed: config.native_gpu_embed,
                gdn_chunked: config.native_gdn_chunked,
                weights_cached: config.native_cache,
            },
            &GpuInfo {
                device: "0",
                name: gpu_name,
                arch: engine.gpu.arch.arch_label(),
                sms: engine.gpu.sms,
                compute_capability,
            },
        );
        match engine.gpu.mem_info() {
            Ok((free, total)) => telemetry::set_gpu_memory(free, total),
            Err(error) => tracing::warn!(%error, "metrics: failed to read GPU memory"),
        }
        let queue = BatchHandle::spawn(
            engine,
            config.max_queue,
            config.batch_requests,
            config.batch_rows,
            config.batch_window_us,
            config.state_cache_entries,
            config.state_cache_bytes,
            config.host_state_cache_bytes,
        );
        Ok(Self {
            tokenizer,
            queue,
            d3,
            supports_vision,
            image_slots: Arc::new(tokio::sync::Semaphore::new(config.image_decode_concurrency)),
        })
    }
}

fn quantization_label(config: &Config) -> &'static str {
    if config.native_marlin && config.native_int8 {
        if config.native_int8_group_scale {
            "w4a8_g64"
        } else {
            "w4a8"
        }
    } else if config.native_marlin {
        "w4a16"
    } else if config.native_fused_w4a16 {
        "w4a16_fused"
    } else {
        "bf16"
    }
}

fn load_d3_config(dir: &Path) -> Result<D3Config> {
    let path = dir.join("decision_config.json");
    let raw =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let value: Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    if value.get("attention_mode").and_then(Value::as_str) != Some("prefix_bidir") {
        bail!(
            "{} does not declare attention_mode=prefix_bidir; only d3-turbo is supported",
            path.display()
        );
    }
    if let Some(format) = value.get("format_id").and_then(Value::as_str) {
        if format != "d3-code-readout-v1" {
            bail!("{} has unsupported format_id {format:?}", path.display());
        }
    }
    let codes = value
        .get("codes")
        .and_then(Value::as_array)
        .context("decision_config.json codes")?
        .iter()
        .map(|code| code.as_str().unwrap_or_default().to_owned())
        .collect::<Vec<_>>();
    if codes.len() < 2 {
        bail!("decision_config.json must declare at least two readout codes");
    }
    let temperature = value
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    Ok(D3Config { codes, temperature })
}

fn self_test(engine: &Engine, tokenizer: &Tokenizer, d3_config: &D3Config) -> Result<()> {
    let state = json!("ok");
    let question = d3::noul_question("self test", None);
    let text = d3::render(&state, &question, &d3_config.codes)?;
    let (ids, split) = d3::encode_split(tokenizer, &text, &state)?;
    let cache = engine
        .capture_state(&ids[..split])
        .context("d3 state-once self-test: capture_state")?;
    let rows = engine
        .readout_state_once_batch(&cache, &[&ids[split..]])
        .context("d3 state-once self-test: question pass")?;
    if rows
        .first()
        .map_or(true, |row| row.iter().any(|v| !v.is_finite()))
    {
        bail!("d3 state-once self-test returned non-finite logits");
    }
    let codes: Vec<u32> = (0..d3_config.codes.len().min(255))
        .map(|i| i as u32)
        .collect();
    let reference = engine
        .decision_logits_split(&ids, split, &codes)
        .context("d3 state-once self-test: per-row block reference")?;
    let softmax = |values: &[f32]| -> Vec<f32> {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut out: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
        let total: f32 = out.iter().sum();
        for v in &mut out {
            *v /= total;
        }
        out
    };
    let state_once = softmax(&rows[0][..codes.len()]);
    let block = softmax(&reference[..codes.len()]);
    let delta = state_once
        .iter()
        .zip(&block)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // The startup gate catches a reversed mask/cache or random artifact. The
    // offline 32-row gate remains the quality authority.
    if delta > 0.12 {
        bail!("d3 state-once self-test disagrees with the block reference by {delta}");
    }
    let yes = softmax(&rows[0][..2])[1];
    if !(0.55..=0.98).contains(&yes) {
        bail!("d3 state-once self-test noul probability {yes} is out of band");
    }
    tracing::info!(delta, yes, "d3 state-once self-test passed");
    Ok(())
}

struct Prepared {
    state_ids: Vec<u32>,
    blocks: Vec<Vec<u32>>,
    counts: Vec<usize>,
    usage: Value,
    question_token_count: usize,
    images: Vec<ImageMedia>,
    state_positions: Vec<[u32; 3]>,
    image_spans: Vec<(usize, usize)>,
    digest: [u8; 32],
}

fn prepare_request(
    tokenizer: &Tokenizer,
    codes: &[String],
    state: &Value,
    questions: &Map<String, Value>,
    images: Vec<ImageMedia>,
) -> Result<Prepared, EngineError> {
    let mut state_ids: Option<Vec<u32>> = None;
    let mut blocks = Vec::with_capacity(questions.len());
    let mut counts = Vec::with_capacity(questions.len());
    let mut state_positions = Vec::new();
    let mut image_spans = Vec::new();
    let image_layouts: Vec<ImageLayout> = images
        .iter()
        .map(|image| ImageLayout {
            tokens: image.tokens(),
            grid_h: image.grid_h,
            grid_w: image.grid_w,
        })
        .collect();
    for (index, (name, question)) in questions.iter().enumerate() {
        let parsed = d3::parse_question(question)
            .map_err(|error| EngineError::Config(format!("question {name}: {error}")))?;
        let text = if images.is_empty() {
            d3::render(state, &parsed, codes)
        } else {
            d3::render_image(state, &parsed, codes, images.len())
        }
        .map_err(|error| EngineError::Config(format!("question {name}: {error}")))?;
        if text.len() > (1 << 20) {
            return Err(EngineError::Config(format!(
                "question {name} renders over the 1 MiB cap"
            )));
        }
        // The first question establishes the exact state token ids. The
        // state block is question-independent; subsequent questions tokenize
        // only their suffix and prepend those ids. This removes 63/64 copies
        // of a 2K-token state from a 64-question request.
        let (head, tail, positions, spans) = if images.is_empty() {
            if index == 0 {
                let (ids, split) = d3::encode_split(tokenizer, &text, state)
                    .map_err(|error| EngineError::Config(format!("question {name}: {error}")))?;
                if split >= ids.len() {
                    return Err(EngineError::Config(format!(
                        "question {name} has no tokens after the state"
                    )));
                }
                let pos = (0..split as u32).map(|t| [t, t, t]).collect();
                (
                    ids[..split].to_vec(),
                    ids[split..].to_vec(),
                    pos,
                    Vec::new(),
                )
            } else {
                let state_ids = state_ids.as_ref().ok_or_else(|| {
                    EngineError::Internal("state tokens missing for a d3 question".to_owned())
                })?;
                let end = d3::state_end_char(&text, state)
                    .map_err(|error| EngineError::Config(format!("question {name}: {error}")))?;
                let suffix = tokenizer.encode(&text[end..], false).map_err(|error| {
                    EngineError::Config(format!("question {name}: tokenizing suffix: {error}"))
                })?;
                let tail = suffix.get_ids().to_vec();
                if tail.is_empty() {
                    return Err(EngineError::Config(format!(
                        "question {name} has no tokens after the state"
                    )));
                }
                (state_ids.clone(), tail, Vec::new(), Vec::new())
            }
        } else {
            let encoded = d3::encode_image_split(tokenizer, &text, state, &image_layouts)
                .map_err(|error| EngineError::Config(format!("question {name}: {error}")))?;
            let state_len = encoded.state_len;
            if state_len >= encoded.ids.len() {
                return Err(EngineError::Config(format!(
                    "question {name} has no tokens after the image state"
                )));
            }
            (
                encoded.ids[..state_len].to_vec(),
                encoded.ids[state_len..].to_vec(),
                encoded.positions[..state_len].to_vec(),
                encoded.image_spans,
            )
        };
        let split = head.len();
        if split > MAX_STATE_TOKENS || split + tail.len() > MAX_STATE_TOKENS {
            return Err(EngineError::Config(format!(
                "question {name} has {split}+{} tokens, over the {MAX_STATE_TOKENS} serving cap",
                tail.len()
            )));
        }
        match &state_ids {
            Some(existing) if *existing != head => {
                return Err(EngineError::BadResponse(
                    "the tokenized state differs across questions".to_owned(),
                ));
            }
            None => {
                state_positions = positions;
                image_spans = spans;
                state_ids = Some(head);
            }
            _ => {}
        }
        blocks.push(tail);
        counts.push(parsed.keys.len());
    }
    let state_ids = state_ids.expect("at least one question");
    let question_token_count = blocks.iter().map(|block| block.len()).sum::<usize>();
    let usage = json!({
        "input_tokens": state_ids.len() as u64 + question_token_count as u64,
        "output_tokens": 0u64,
    });
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(images.len() as u64).to_le_bytes());
    for image in &images {
        hasher.update(&image.digest);
    }
    let digest = *hasher.finalize().as_bytes();
    Ok(Prepared {
        state_ids,
        blocks,
        counts,
        usage,
        question_token_count,
        images,
        state_positions,
        image_spans,
        digest,
    })
}

impl Backend for NativeBackend {
    fn name(&self) -> &str {
        "native"
    }

    fn d3_answers<'a>(
        &'a self,
        state: &'a Value,
        questions: &'a Map<String, Value>,
        images: Vec<String>,
    ) -> D3Future<'a> {
        Box::pin(async move { self.answers(state, questions, images).await })
    }
}

impl NativeBackend {
    async fn answers(
        &self,
        state: &Value,
        questions: &Map<String, Value>,
        raw_images: Vec<String>,
    ) -> Result<Value, EngineError> {
        if questions.is_empty() {
            return Err(EngineError::Config(
                "questions must not be empty".to_owned(),
            ));
        }
        if questions.len() > MAX_QUESTIONS {
            return Err(EngineError::Config(format!(
                "at most {MAX_QUESTIONS} questions per state-once request"
            )));
        }
        if d3::contains_media(state) {
            return Err(EngineError::Unsupported(
                "image and video states are not supported by this text engine".to_owned(),
            ));
        }
        if serde_json::to_string(state).map_or(true, |text| text.len() > (1 << 20)) {
            return Err(EngineError::Config(
                "state is over the 1 MiB rendering cap".to_owned(),
            ));
        }

        let tokenize_started = Instant::now();
        let tokenizer = self.tokenizer.clone();
        let codes = self.d3.codes.clone();
        let state = state.clone();
        let questions_for_prepare = questions.clone();
        if !raw_images.is_empty() && !self.supports_vision {
            return Err(EngineError::Config(
                "this checkpoint has no vision tower; image inputs are unavailable".to_owned(),
            ));
        }
        metrics::histogram!("decidra_request_images").record(raw_images.len() as f64);
        let image_slot = if raw_images.is_empty() {
            None
        } else {
            Some(
                tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    self.image_slots.clone().acquire_owned(),
                )
                .await
                .map_err(|_| {
                    metrics::counter!(
                        "decidra_engine_admission_rejections_total",
                        "reason" => "image_busy"
                    )
                    .increment(1);
                    EngineError::Overloaded(
                        "image decoder queue is saturated; retry shortly".to_owned(),
                    )
                })?
                .map_err(|_| EngineError::Internal("image decoder stopped".to_owned()))?,
            )
        };
        let prepared_result = tokio::task::spawn_blocking(move || {
            // The permit lives in the blocking task: client cancellation must
            // not release a decoder slot while the decode is still running.
            let _image_slot = image_slot;
            if raw_images.len() > MAX_IMAGES {
                return Err(EngineError::Config(format!(
                    "at most {MAX_IMAGES} images per request"
                )));
            }
            let mut images = Vec::with_capacity(raw_images.len());
            let mut image_tokens = 0usize;
            for (i, value) in raw_images.iter().enumerate() {
                let decode_started = Instant::now();
                let decoded = crate::image_media::decode_data_url(value);
                metrics::histogram!("decidra_image_decode_seconds")
                    .record(decode_started.elapsed().as_secs_f64());
                let image = decoded
                    .map_err(|error| EngineError::Config(format!("images[{i}]: {error}")))?;
                image_tokens += image.tokens();
                if image_tokens > MAX_IMAGE_TOKENS {
                    return Err(EngineError::Config(format!(
                        "images expand to {image_tokens} vision tokens, over the {MAX_IMAGE_TOKENS} image-token budget"
                    )));
                }
                images.push(image);
            }
            prepare_request(&tokenizer, &codes, &state, &questions_for_prepare, images)
        })
        .await;
        metrics::histogram!("decidra_request_tokenize_seconds")
            .record(tokenize_started.elapsed().as_secs_f64());
        let prepared = prepared_result
            .map_err(|error| EngineError::Internal(format!("tokenizer task failed: {error}")))??;
        let tokenize_ms = tokenize_started.elapsed().as_secs_f64() * 1e3;
        let state_token_count = prepared.state_ids.len();
        let question_token_count = prepared.question_token_count;
        let block_count = prepared.blocks.len();
        metrics::histogram!("decidra_request_state_tokens").record(state_token_count as f64);
        metrics::histogram!("decidra_request_question_tokens").record(question_token_count as f64);
        metrics::histogram!("decidra_request_questions").record(block_count as f64);
        let counts = prepared.counts;
        let usage = prepared.usage;
        let submit_started = Instant::now();
        let receiver = match self
            .queue
            .submit_state_once(
                prepared.state_ids,
                prepared.blocks,
                prepared.images,
                prepared.state_positions,
                prepared.image_spans,
                prepared.digest,
            )
            .await
        {
            Ok(receiver) => receiver,
            Err(error) => {
                metrics::histogram!("decidra_request_engine_seconds")
                    .record(submit_started.elapsed().as_secs_f64());
                return Err(match error {
                    EngineError::Overloaded(message) => EngineError::Overloaded(message),
                    other => EngineError::Internal(other.to_string()),
                });
            }
        };
        let queue_submit_ms = submit_started.elapsed().as_secs_f64() * 1e3;
        let reply = receiver.await;
        metrics::histogram!("decidra_request_engine_seconds")
            .record(submit_started.elapsed().as_secs_f64());
        let rows = match reply {
            Ok(rows) => rows.map_err(EngineError::Backend)?,
            Err(_) => {
                metrics::counter!(
                    "decidra_engine_failures_total",
                    "stage" => "scheduler"
                )
                .increment(1);
                return Err(EngineError::Internal(
                    "d3 CUDA worker dropped the reply".to_owned(),
                ));
            }
        };
        tracing::debug!(
            questions = block_count,
            state_tokens = state_token_count,
            question_tokens = question_token_count,
            tokenize_ms,
            queue_submit_ms,
            "d3 request"
        );
        if rows.len() != block_count {
            return Err(EngineError::BadResponse(format!(
                "native engine returned {} rows for {} questions",
                rows.len(),
                block_count
            )));
        }

        let mut answers = Map::with_capacity(questions.len());
        for ((name, count), row) in questions.keys().zip(counts).zip(rows) {
            if row.len() < count {
                return Err(EngineError::BadResponse(
                    "readout returned fewer logits than options".to_owned(),
                ));
            }
            let scaled: Vec<f32> = row[..count]
                .iter()
                .map(|value| (value / self.d3.temperature as f32).exp())
                .collect();
            let total: f32 = scaled.iter().sum();
            if !total.is_finite() || total <= 0.0 {
                return Err(EngineError::BadResponse("non-finite d3 logits".to_owned()));
            }
            let probabilities: Vec<f32> = scaled.iter().map(|value| value / total).collect();
            let question = d3::parse_question(&questions[name])
                .map_err(|error| EngineError::Internal(error.to_string()))?;
            let answer = d3::to_answer(&question, &probabilities)
                .map_err(|error| EngineError::Internal(error.to_string()))?;
            answers.insert(name.clone(), answer);
        }
        Ok(json!({
            "answers": answers,
            "usage": usage,
        }))
    }
}

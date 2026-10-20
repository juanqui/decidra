//! Axum application: JEV decision endpoints, OpenAI-compatible decisions,
//! HMAC signing, and Prometheus metrics. The OpenAI routes execute the same
//! native decision as `/v1/decide`; they do not generate free-form text.

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Instant};

use anyhow::{Context, Result, bail};
use axum::{
    Json, Router,
    body::{Body, Bytes, to_bytes},
    extract::{DefaultBodyLimit, Request, State, rejection::JsonRejection},
    http::{
        HeaderValue, StatusCode,
        header::{CONTENT_LENGTH, CONTENT_TYPE},
    },
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use hmac::{Hmac, Mac};
use metrics_exporter_prometheus::PrometheusHandle;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::Sha256;
use uuid::Uuid;

use crate::{
    DEFAULT_STRATEGY, MAX_OPTIONS, NATIVE_CHOICE_OPTIONS, PROTOCOL, PUBLIC_ALIAS, PUBLIC_MODEL,
    backend::Backend,
    config::Config,
    engine::{DecisionEngine, Kind, Strategy, Usage},
    error::{ApiError, EngineError},
    head::{self, DecisionHead, FittedKind, Temperatures},
    prompt::{self, LABEL_CONTEXT},
};

const SIGNATURE_HEADER: &str = "x-jev-signature";
const SIGNATURE_BODY_LIMIT: usize = 128 * 1024 * 1024;

/// Startup diagnostics, also printed by `--check-config`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LoadReport {
    pub checkpoint: PathBuf,
    pub native_model: PathBuf,
    pub tokenizer: PathBuf,
    pub head: PathBuf,
    pub calibration: PathBuf,
    pub fitted: Option<PathBuf>,
    pub max_options: usize,
    pub native_choice_options: usize,
    pub temperatures: Value,
    pub backend: String,
    pub batching: bool,
    pub max_batch: usize,
    pub batch_window_us: u64,
    pub max_batch_tokens: usize,
    pub signing_enabled: bool,
}

/// Everything loaded from disk that shapes probabilities, without an inference
/// backend. `--check-config` stops here; `serve` adds the native engine.
pub struct Metadata {
    pub head: DecisionHead,
    pub temperatures: Temperatures,
    pub fitted: Option<HashMap<String, FittedKind>>,
    pub label_ids: Vec<u32>,
    pub report: LoadReport,
}

/// Load and validate the tokenizer, decision head, calibration, and labels.
pub fn load_metadata(config: &Config) -> Result<Metadata> {
    let loaded = head::load(
        &config.checkpoint,
        config.calibration.as_deref(),
        config.calibration_fitted.as_deref(),
        &config.hf_snapshot,
    )
    .with_context(|| {
        format!(
            "loading decision head/calibration from {} (HF fallback {})",
            config.checkpoint.display(),
            config.hf_snapshot.display()
        )
    })?;
    let (tokenizer, tokenizer_path) = prompt::load_tokenizer(
        &config.checkpoint,
        config.tokenizer.as_deref(),
        &config.hf_snapshot,
    )?;
    let label_ids = prompt::derive_label_ids(&tokenizer, LABEL_CONTEXT, MAX_OPTIONS)?;
    prompt::validate_label_parity(&label_ids, &loaded.head)?;
    if label_ids.len() < MAX_OPTIONS {
        tracing::warn!(
            labels = label_ids.len(),
            "tokenizer yields fewer than 256 wide-choice labels; max_options is reduced"
        );
    }
    if label_ids.len() < 2 {
        bail!("tokenizer must yield at least 2 single-token labels");
    }

    let native_choice_options = loaded.head.choice[1] - loaded.head.choice[0];
    let report = LoadReport {
        checkpoint: config.checkpoint.clone(),
        native_model: config.native_model.clone(),
        tokenizer: tokenizer_path,
        head: loaded.head_path.clone(),
        calibration: loaded.calibration_path.clone(),
        fitted: loaded.fitted_path.clone(),
        max_options: label_ids.len().min(MAX_OPTIONS),
        native_choice_options,
        temperatures: loaded.temperatures.raw.clone(),
        backend: "not loaded (check-config)".to_owned(),
        batching: config.native_batch,
        max_batch: config.native_max_batch,
        batch_window_us: config.native_batch_window_us,
        max_batch_tokens: config.native_max_batch_tokens,
        signing_enabled: config.signing_enabled(),
    };
    Ok(Metadata {
        head: loaded.head,
        temperatures: loaded.temperatures,
        fitted: loaded.fitted,
        label_ids,
        report,
    })
}

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<DecisionEngine>,
    pub config: Arc<Config>,
    pub metrics: Option<PrometheusHandle>,
    pub signing_key: Option<Arc<Vec<u8>>>,
    pub load_report: LoadReport,
}

impl AppState {
    /// Build the application around an inference backend.
    pub fn load(config: Config, backend: Arc<dyn Backend>) -> Result<Self> {
        let metadata = load_metadata(&config)?;
        Ok(Self::from_metadata(config, metadata, backend))
    }

    pub fn from_metadata(config: Config, metadata: Metadata, backend: Arc<dyn Backend>) -> Self {
        let mut report = metadata.report;
        report.backend = backend.name().to_owned();
        let fitted = metadata.fitted.map(Arc::new);
        let engine = Arc::new(DecisionEngine::new(
            Arc::new(metadata.head),
            Arc::new(metadata.temperatures),
            fitted,
            metadata.label_ids,
            backend,
        ));
        Self {
            engine,
            signing_key: config.signing_key.clone().map(Arc::new),
            config: Arc::new(config),
            metrics: None,
            load_report: report,
        }
    }

    pub fn with_metrics(mut self, handle: PrometheusHandle) -> Self {
        self.metrics = Some(handle);
        self
    }

    pub fn router(self) -> Router {
        let state = self.clone();
        Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(models))
            .route("/v1/decide/info", get(decide_info))
            .route("/v1/decide", post(decide))
            .route("/v1/decisions", post(decisions))
            .route("/v1/systemone", post(systemone))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .route("/metrics", get(metrics_endpoint))
            .layer(DefaultBodyLimit::max(state.config.max_body_bytes))
            .layer(middleware::from_fn_with_state(
                state.clone(),
                sign_responses,
            ))
            .layer(middleware::from_fn_with_state(state.clone(), observe_http))
            .with_state(state)
    }
}

async fn health(State(_state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ready",
        "model": PUBLIC_MODEL,
        "head": "merged-lm_head",
    }))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    let data: Vec<Value> = [PUBLIC_MODEL, PUBLIC_ALIAS, state.config.model_id.as_str()]
        .into_iter()
        .map(|name| {
            json!({
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "autotrust",
            })
        })
        .collect();
    Json(json!({"object": "list", "data": data}))
}

async fn decide_info(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "protocol": PROTOCOL,
        "model": PUBLIC_MODEL,
        "max_options": state.engine.max_options(),
        "native_choice_options": state.engine.head.choice[1] - state.engine.head.choice[0],
        "temperatures": state.engine.temperatures.raw,
        "defaults": {"strategy": DEFAULT_STRATEGY},
    }))
}

async fn metrics_endpoint(State(state): State<AppState>) -> Response {
    let body = match (&state.config.metrics_enabled, &state.metrics) {
        (true, Some(handle)) => handle.render(),
        (true, None) => "# metrics recorder not installed\n".to_owned(),
        (false, _) => "# metrics disabled by JEV_METRICS_ENABLED\n".to_owned(),
    };
    (
        [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
        .into_response()
}

#[derive(Debug, Deserialize)]
struct DecideRequest {
    kind: String,
    #[serde(default = "empty_state")]
    state: Value,
    question: String,
    #[serde(default)]
    options: Option<Vec<String>>,
    #[serde(default)]
    strategy: Strategy,
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
}

fn empty_state() -> Value {
    Value::String(String::new())
}

async fn decide(
    State(state): State<AppState>,
    payload: Result<Json<DecideRequest>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(request) = payload.map_err(json_rejection)?;
    decide_value(&state, request).await.map(Json)
}

/// Shared decision path used by `/v1/decide` and the OpenAI-compatible routes.
async fn decide_value(state: &AppState, request: DecideRequest) -> Result<Value, ApiError> {
    let kind = Kind::parse(&request.kind)
        .ok_or_else(|| ApiError::bad_request("kind must be noul|score|choice"))?;
    prompt::validate_state(&request.state).map_err(ApiError::bad_request)?;
    if prompt::image_count(&request.state) > 0 {
        return Err(ApiError::bad_request(
            "native vision not implemented: image states are not supported",
        ));
    }

    let max_options = state.engine.max_options();
    let options: Vec<String> = match kind {
        Kind::Choice => {
            let options = request.options.unwrap_or_default();
            if !(2..=max_options).contains(&options.len()) {
                return Err(ApiError::bad_request(format!(
                    "choice needs 2-{max_options} options in this front, got {}",
                    options.len()
                )));
            }
            options
        }
        Kind::Noul => vec!["false".to_owned(), "true".to_owned()],
        Kind::Score => (0..6).map(|index| index.to_string()).collect(),
    };

    tracing::debug!(
        kind = kind.as_str(),
        options = options.len(),
        strategy = request.strategy.as_str(),
        "decision request"
    );
    let outcome = state
        .engine
        .decide(
            kind,
            &request.state,
            &request.question,
            &options,
            request.strategy,
        )
        .await
        .map_err(ApiError::from)?;
    let choice_index = head::argmax_first(&outcome.probabilities)
        .ok_or_else(|| ApiError::internal("decision returned no probabilities"))?;
    let confidence = outcome.probabilities[choice_index];
    let adaptation = if kind == Kind::Choice && options.len() > NATIVE_CHOICE_OPTIONS {
        outcome.effective_strategy.as_str()
    } else {
        "native"
    };
    let usage = outcome.usage;

    Ok(json!({
        "kind": kind.as_str(),
        "effective_kind": kind.as_str(),
        "options": options,
        "probabilities": outcome.probabilities,
        "choice_index": choice_index,
        "choice": options[choice_index],
        "confidence": confidence,
        "adaptation": adaptation,
        "protocol": PROTOCOL,
        "model": PUBLIC_MODEL,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens(),
        },
        "num_model_requests": usage.requests,
    }))
}

#[derive(Debug, Deserialize)]
struct DecisionsRequest {
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    #[serde(default)]
    input: Value,
    questions: Vec<Value>,
    #[serde(default)]
    #[allow(dead_code)]
    safety_identifier: Option<String>,
}

async fn decisions(
    State(state): State<AppState>,
    payload: Result<Json<DecisionsRequest>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(request) = payload.map_err(json_rejection)?;
    let state_text = decisions_state(&request.input)?.ok_or_else(|| {
        ApiError::bad_request(
            "native vision not implemented: input_image content parts are not supported",
        )
    })?;
    let state_value = Value::String(state_text);
    let max_choice = state.engine.max_options();
    let score_levels = state.engine.head.score[1] - state.engine.head.score[0];
    let mut usage = Usage::default();
    let mut answers = Vec::with_capacity(request.questions.len());

    for question in &request.questions {
        let question = question
            .as_object()
            .ok_or_else(|| ApiError::bad_request("each question must be a JSON object"))?;
        let type_value = question.get("type").cloned().unwrap_or(Value::Null);
        let kind = type_value.as_str().unwrap_or("");
        let name = question.get("name").cloned().unwrap_or(Value::Null);
        let instructions = match question.get("instructions") {
            Some(value) if crate::pyjson::py_truthy(value) => crate::pyjson::py_str(value),
            _ => String::new(),
        };

        match kind {
            "predicate" => {
                let options = ["false".to_owned(), "true".to_owned()];
                let outcome = state
                    .engine
                    .decide(
                        Kind::Noul,
                        &state_value,
                        &instructions,
                        &options,
                        Strategy::Tournament,
                    )
                    .await
                    .map_err(ApiError::from)?;
                usage.add(outcome.usage);
                let probability = outcome.probabilities.get(1).copied().unwrap_or(0.0);
                metrics::counter!("jev_decisions_total", "kind" => "predicate").increment(1);
                answers.push(json!({
                    "type": "predicate",
                    "name": name,
                    "probability": probability,
                }));
            }
            "choice" => {
                let choices = question
                    .get("choices")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if !(2..=max_choice).contains(&choices.len()) {
                    return Err(ApiError::bad_request(format!(
                        "choice needs 2-{max_choice} choices, got {}",
                        choices.len()
                    )));
                }
                let mut options = Vec::with_capacity(choices.len());
                let mut values = Vec::with_capacity(choices.len());
                for choice in &choices {
                    let choice = choice.as_object().ok_or_else(|| {
                        ApiError::bad_request("each choice must be a JSON object")
                    })?;
                    let value = choice.get("value").cloned().unwrap_or(Value::Null);
                    let description = choice.get("description").cloned().unwrap_or(Value::Null);
                    options.push(option_line(&value, &description));
                    values.push(value);
                }
                let outcome = state
                    .engine
                    .decide(
                        Kind::Choice,
                        &state_value,
                        &instructions,
                        &options,
                        Strategy::Tournament,
                    )
                    .await
                    .map_err(ApiError::from)?;
                usage.add(outcome.usage);
                let best = head::argmax_first(&outcome.probabilities)
                    .ok_or_else(|| ApiError::internal("choice returned no probabilities"))?;
                let confidence = outcome.probabilities[best];
                let probabilities: Vec<Value> = values
                    .iter()
                    .zip(&outcome.probabilities)
                    .map(|(value, probability)| json!({"value": value, "probability": probability}))
                    .collect();
                metrics::counter!("jev_decisions_total", "kind" => "choice").increment(1);
                answers.push(json!({
                    "type": "choice",
                    "name": name,
                    "choice": values[best],
                    "probabilities": probabilities,
                    "confidence": confidence,
                }));
            }
            "score" => {
                let levels = question
                    .get("levels")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if !(2..=score_levels).contains(&levels.len()) {
                    return Err(ApiError::bad_request(format!(
                        "score needs 2-{score_levels} levels in this front, got {}",
                        levels.len()
                    )));
                }
                let mut options = Vec::with_capacity(levels.len());
                let mut labels = Vec::with_capacity(levels.len());
                for level in &levels {
                    let level = level
                        .as_object()
                        .ok_or_else(|| ApiError::bad_request("each level must be a JSON object"))?;
                    let label = level.get("label").cloned().unwrap_or(Value::Null);
                    let description = level.get("description").cloned().unwrap_or(Value::Null);
                    options.push(option_line(&label, &description));
                    labels.push(label);
                }
                let outcome = state
                    .engine
                    .decide(
                        Kind::Score,
                        &state_value,
                        &instructions,
                        &options,
                        Strategy::Tournament,
                    )
                    .await
                    .map_err(ApiError::from)?;
                usage.add(outcome.usage);
                let score: f64 = outcome
                    .probabilities
                    .iter()
                    .enumerate()
                    .map(|(index, probability)| index as f64 * probability)
                    .sum();
                let confidence = outcome
                    .probabilities
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let probabilities: Vec<Value> = outcome
                    .probabilities
                    .iter()
                    .enumerate()
                    .map(|(index, probability)| {
                        json!({
                            "value": index,
                            "label": crate::pyjson::py_str(&labels[index]),
                            "probability": probability,
                        })
                    })
                    .collect();
                metrics::counter!("jev_decisions_total", "kind" => "score").increment(1);
                answers.push(json!({
                    "type": "score",
                    "name": name,
                    "score": score,
                    "probabilities": probabilities,
                    "confidence": confidence,
                }));
            }
            _ => {
                return Err(ApiError::bad_request(format!(
                    "unsupported question type {}",
                    crate::pyjson::py_repr(&type_value)
                )));
            }
        }
    }

    Ok(Json(json!({
        "model": PUBLIC_MODEL,
        "answers": answers,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens(),
        },
    })))
}

/// Port of `_decisions_state`: text of the input, or `None` when it contains images.
fn decisions_state(input: &Value) -> Result<Option<String>, ApiError> {
    match input {
        Value::String(text) => Ok(Some(text.clone())),
        Value::Array(messages) => {
            let mut parts = Vec::new();
            for message in messages {
                let Some(message) = message.as_object() else {
                    continue;
                };
                match message.get("content") {
                    Some(Value::String(text)) => parts.push(text.clone()),
                    Some(Value::Array(content)) => {
                        for part in content {
                            let Some(part) = part.as_object() else {
                                continue;
                            };
                            match part.get("type").and_then(Value::as_str) {
                                Some("input_image") => return Ok(None),
                                Some("input_text") => {
                                    let text = part
                                        .get("text")
                                        .cloned()
                                        .unwrap_or_else(|| Value::String(String::new()));
                                    parts.push(crate::pyjson::py_str(&text));
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Some(parts.join("\n\n")))
        }
        _ => Ok(Some(String::new())),
    }
}

/// Port of `_option_line`: `"{value}: {description}"` when the description is truthy.
fn option_line(value: &Value, description: &Value) -> String {
    if crate::pyjson::py_truthy(description) {
        format!(
            "{}: {}",
            crate::pyjson::py_str(value),
            crate::pyjson::py_str(description)
        )
    } else {
        crate::pyjson::py_str(value)
    }
}

const MAX_SYSTEMONE_QUESTIONS: usize = 8;
const SYSTEMONE_PROVIDER: &str = "xPNP Hosted";

struct SystemonePlan {
    name: String,
    kind: &'static str,
    state_text: String,
    instructions: String,
    choice: Option<(Vec<String>, Vec<String>)>,
}

fn s1_error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message.into(),
                "code": status.as_u16(),
            }
        })),
    )
        .into_response()
}

/// Port of `_s1_text`: None becomes "", strings are verbatim, other JSON is dumped
/// with CPython-compatible formatting.
fn s1_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(value) => crate::pyjson::dumps(value),
    }
}

/// Port of `_s1_split_state`: text plus at most one normalized image part.
fn s1_split_state(state: &Value) -> Result<(String, Option<Value>), String> {
    match state {
        Value::String(text) => Ok((text.clone(), None)),
        Value::Object(_) => Ok((crate::pyjson::dumps(state), None)),
        Value::Array(parts) => {
            let mut texts = Vec::new();
            let mut images: Vec<Value> = Vec::new();
            for part in parts {
                match part {
                    Value::String(text) => texts.push(text.clone()),
                    Value::Object(map)
                        if map.get("type") == Some(&Value::String("text".to_owned())) =>
                    {
                        texts.push(s1_text(map.get("text")));
                    }
                    Value::Object(map)
                        if map.get("type") == Some(&Value::String("image_url".to_owned())) =>
                    {
                        let url = match map.get("image_url") {
                            Some(Value::Object(image_url)) => image_url.get("url"),
                            other => other,
                        };
                        let Some(url) = url.and_then(Value::as_str).filter(|url| !url.is_empty())
                        else {
                            return Err("an image_url part needs a url".to_owned());
                        };
                        let mut image_url = serde_json::Map::new();
                        image_url.insert("url".to_owned(), Value::String(url.to_owned()));
                        let mut image = serde_json::Map::new();
                        image.insert("type".to_owned(), Value::String("image_url".to_owned()));
                        image.insert("image_url".to_owned(), Value::Object(image_url));
                        images.push(Value::Object(image));
                    }
                    Value::Object(map) if map.get("image").is_some_and(Value::is_string) => {
                        let mut image_url = serde_json::Map::new();
                        image_url.insert("url".to_owned(), map["image"].clone());
                        let mut image = serde_json::Map::new();
                        image.insert("type".to_owned(), Value::String("image_url".to_owned()));
                        image.insert("image_url".to_owned(), Value::Object(image_url));
                        images.push(Value::Object(image));
                    }
                    other => texts.push(crate::pyjson::dumps(other)),
                }
            }
            if images.len() > 1 {
                return Err(format!(
                    "this backend reads one image per request, got {}",
                    images.len()
                ));
            }
            Ok((texts.join("\n"), images.pop()))
        }
        _ => Err("state must be a string, an object, or an array".to_owned()),
    }
}

fn valid_choice_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 64
        && key
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

/// xPNP hosted System One: one decision pass per question, no tournament/permute.
async fn systemone(State(state): State<AppState>, body: Bytes) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(_) => {
            return s1_error(
                StatusCode::BAD_REQUEST,
                "send a JSON body with state and questions",
            );
        }
    };
    let Some(object) = body.as_object() else {
        return s1_error(StatusCode::BAD_REQUEST, "send a JSON object");
    };
    let Some(questions) = object.get("questions").and_then(Value::as_object) else {
        return s1_error(
            StatusCode::BAD_REQUEST,
            "questions must be a non-empty object",
        );
    };
    if questions.is_empty() {
        return s1_error(
            StatusCode::BAD_REQUEST,
            "questions must be a non-empty object",
        );
    }
    if questions.len() > MAX_SYSTEMONE_QUESTIONS {
        return s1_error(StatusCode::BAD_REQUEST, "at most 8 questions per request");
    }
    let state_value = object
        .get("state")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let (text, image) = match s1_split_state(&state_value) {
        Ok(parts) => parts,
        Err(message) => return s1_error(StatusCode::BAD_REQUEST, message),
    };
    if image.is_some() {
        return s1_error(
            StatusCode::BAD_REQUEST,
            "native vision not implemented: image states are not supported",
        );
    }

    let mut plans = Vec::with_capacity(questions.len());
    for (name, question) in questions {
        let Some(question) = question.as_object() else {
            return s1_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "question {} must be an object",
                    crate::pyjson::py_repr(&Value::String(name.clone()))
                ),
            );
        };
        let type_value = question.get("type").cloned().unwrap_or(Value::Null);
        let kind = type_value.as_str().unwrap_or("");
        let instructions = s1_text(question.get("instructions")).trim().to_owned();
        if instructions.is_empty() {
            return s1_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "question {} needs instructions",
                    crate::pyjson::py_repr(&Value::String(name.clone()))
                ),
            );
        }
        let criteria = question.get("criteria");
        match kind {
            "noul" => {
                let mut lines = text.clone();
                if let Some(criteria) = criteria.and_then(Value::as_object) {
                    let true_value = criteria.get("true");
                    let false_value = criteria.get("false");
                    if true_value.is_some_and(|value| !value.is_null())
                        || false_value.is_some_and(|value| !value.is_null())
                    {
                        lines = format!(
                            "{text}\nAnswer true when: {}\nAnswer false when: {}",
                            s1_text(true_value),
                            s1_text(false_value)
                        );
                    }
                }
                plans.push(SystemonePlan {
                    name: name.clone(),
                    kind: "noul",
                    state_text: format!("{lines}\n"),
                    instructions,
                    choice: None,
                });
            }
            "choice" => {
                let Some(criteria) = criteria.and_then(Value::as_object) else {
                    return s1_error(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "choice question {} needs 2 to 16 criteria",
                            crate::pyjson::py_repr(&Value::String(name.clone()))
                        ),
                    );
                };
                if !(2..=16).contains(&criteria.len()) {
                    return s1_error(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "choice question {} needs 2 to 16 criteria",
                            crate::pyjson::py_repr(&Value::String(name.clone()))
                        ),
                    );
                }
                let keys: Vec<String> = criteria.keys().cloned().collect();
                if let Some(bad) = keys.iter().find(|key| !valid_choice_key(key)) {
                    return s1_error(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "choice keys must match [a-z0-9-]+, got {}",
                            crate::pyjson::py_repr(&Value::String(bad.clone()))
                        ),
                    );
                }
                let options = keys
                    .iter()
                    .map(|key| match criteria.get(key) {
                        Some(value) if !value.is_null() => {
                            format!("{key}: {}", s1_text(Some(value)))
                        }
                        _ => key.clone(),
                    })
                    .collect();
                plans.push(SystemonePlan {
                    name: name.clone(),
                    kind: "choice",
                    state_text: format!("{text}\n"),
                    instructions,
                    choice: Some((keys, options)),
                });
            }
            "score" => {
                return s1_error(
                    StatusCode::BAD_REQUEST,
                    "score questions are not served by this backend",
                );
            }
            _ => {
                return s1_error(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "question {} has an unknown type {}",
                        crate::pyjson::py_repr(&Value::String(name.clone())),
                        crate::pyjson::py_repr(&type_value)
                    ),
                );
            }
        }
    }

    metrics::counter!("jev_systemone_requests_total").increment(1);
    let mut usage = Usage::default();
    let mut answers = serde_json::Map::with_capacity(plans.len());
    for plan in plans {
        let kind = if plan.kind == "noul" {
            Kind::Noul
        } else {
            Kind::Choice
        };
        let options = if kind == Kind::Noul {
            vec!["false".to_owned(), "true".to_owned()]
        } else {
            plan.choice
                .as_ref()
                .map(|(_, options)| options.clone())
                .unwrap_or_default()
        };
        let mut state_parts = vec![Value::String(plan.state_text.clone())];
        let state_value = Value::Array(std::mem::take(&mut state_parts));
        let (probabilities, pass_usage) = match state
            .engine
            .s1_pass(kind, &state_value, &plan.instructions, &options)
            .await
        {
            Ok(result) => result,
            Err(error) => {
                let status = if matches!(&error, EngineError::Timeout(_)) {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::BAD_GATEWAY
                };
                return s1_error(status, format!("backend: {error}"));
            }
        };
        usage.add(pass_usage);
        metrics::counter!("jev_systemone_questions_total", "kind" => plan.kind).increment(1);

        if kind == Kind::Noul {
            answers.insert(
                plan.name,
                json!({
                    "type": "noul",
                    "noul": probabilities.get(1).copied().unwrap_or(0.0),
                }),
            );
        } else {
            let Some((keys, _)) = plan.choice.as_ref() else {
                return s1_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal: choice plan missing option keys",
                );
            };
            let Some(best) = head::argmax_first(&probabilities) else {
                return s1_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal: choice returned no probabilities",
                );
            };
            let (Some(key), Some(confidence)) = (keys.get(best), probabilities.get(best).copied())
            else {
                return s1_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal: choice probabilities do not match option keys",
                );
            };
            let mut distribution = serde_json::Map::with_capacity(keys.len());
            for (key, probability) in keys.iter().zip(probabilities.iter()) {
                distribution.insert(key.clone(), json!(probability));
            }
            answers.insert(
                plan.name,
                json!({
                    "type": "choice",
                    "choice": key,
                    "confidence": confidence,
                    "probabilities": distribution,
                }),
            );
        }
    }

    Json(json!({
        "id": format!("xpnp-hosted-{}", Uuid::new_v4().simple()),
        "model": state.config.served_identity,
        "provider": SYSTEMONE_PROVIDER,
        "answers": answers,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": 0,
            "cost": 0,
            "num_model_requests": usage.requests,
        },
    }))
    .into_response()
}

/// OpenAI-compatible chat completions. The last `user` message must be a JSON
/// decision request (`{"kind","state","question","options"}`); the assistant
/// content is the compact decision JSON. This is the route LiteLLM proxies.
#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Option<ChatContent>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ChatContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

#[derive(Debug, Deserialize)]
struct ChatContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: CompletionPrompt,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CompletionPrompt {
    Text(String),
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn message_text(message: &ChatMessage) -> Result<String, ApiError> {
    match &message.content {
        Some(ChatContent::Text(text)) => Ok(text.clone()),
        Some(ChatContent::Parts(parts)) => {
            let mut out = String::new();
            for part in parts {
                if part.kind != "text" {
                    return Err(ApiError::bad_request(format!(
                        "only text content parts are supported, got {:?}",
                        part.kind
                    )));
                }
                if let Some(text) = &part.text {
                    out.push_str(text);
                }
            }
            Ok(out)
        }
        None => Ok(String::new()),
    }
}

/// Parse a decision request from user text, optionally inside a Markdown fence.
fn parse_decision_text(text: &str) -> Result<DecideRequest, ApiError> {
    let trimmed = text.trim();
    let unwrapped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|rest| rest.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(trimmed);
    serde_json::from_str(unwrapped).map_err(|error| {
        ApiError::bad_request(format!(
            "user message must be a JSON decision request \
             {{\"kind\":\"noul|score|choice\",\"state\":...,\"question\":...,\"options\":[...]}}: {error}"
        ))
    })
}

fn openai_usage(decision: &Value) -> Value {
    decision
        .get("usage")
        .cloned()
        .unwrap_or_else(|| json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}))
}

fn decision_content(decision: &Value) -> String {
    serde_json::to_string(decision).unwrap_or_default()
}

fn sse_events(events: Vec<Value>) -> Response {
    use axum::response::sse::{Event, Sse};
    let items: Vec<Result<Event, std::convert::Infallible>> = events
        .into_iter()
        .map(|value| {
            let data = if value.is_string() {
                value.as_str().unwrap_or_default().to_owned()
            } else {
                value.to_string()
            };
            Ok(Event::default().data(data))
        })
        .collect();
    Sse::new(futures::stream::iter(items)).into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(rejection) => return json_rejection(rejection).into_response(),
    };
    let text = match request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(message_text)
        .transpose()
    {
        Ok(Some(text)) => text,
        Ok(None) => {
            return ApiError::bad_request("chat request needs a user message").into_response();
        }
        Err(error) => return error.into_response(),
    };
    let decision_request = match parse_decision_text(&text) {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    let decision = match decide_value(&state, decision_request).await {
        Ok(value) => value,
        Err(error) => return error.into_response(),
    };
    let model = request
        .model
        .unwrap_or_else(|| state.config.model_id.clone());
    let content = decision_content(&decision);
    let usage = openai_usage(&decision);
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = unix_now();
    if request.stream {
        let include_usage = request
            .stream_options
            .map(|options| options.include_usage)
            .unwrap_or(false);
        let mut events = vec![
            json!({
                "id": id, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
            }),
            json!({
                "id": id, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}]
            }),
        ];
        let mut finish = json!({
            "id": id, "object": "chat.completion.chunk", "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        });
        if include_usage {
            finish["usage"] = usage;
        }
        events.push(finish);
        events.push(Value::String("[DONE]".to_owned()));
        return sse_events(events);
    }
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": usage,
    }))
    .into_response()
}

async fn completions(
    State(state): State<AppState>,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(rejection) => return json_rejection(rejection).into_response(),
    };
    let CompletionPrompt::Text(text) = request.prompt;
    let decision_request = match parse_decision_text(&text) {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    let decision = match decide_value(&state, decision_request).await {
        Ok(value) => value,
        Err(error) => return error.into_response(),
    };
    let model = request
        .model
        .unwrap_or_else(|| state.config.model_id.clone());
    let content = decision_content(&decision);
    let usage = openai_usage(&decision);
    let id = format!("cmpl-{}", Uuid::new_v4().simple());
    let created = unix_now();
    if request.stream {
        let include_usage = request
            .stream_options
            .map(|options| options.include_usage)
            .unwrap_or(false);
        let mut events = vec![json!({
            "id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": content, "finish_reason": null, "logprobs": null}]
        })];
        let mut finish = json!({
            "id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop", "logprobs": null}]
        });
        if include_usage {
            finish["usage"] = usage;
        }
        events.push(finish);
        events.push(Value::String("[DONE]".to_owned()));
        return sse_events(events);
    }
    Json(json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "text": content,
            "finish_reason": "stop",
            "logprobs": null
        }],
        "usage": usage,
    }))
    .into_response()
}

fn json_rejection(rejection: JsonRejection) -> ApiError {
    ApiError::bad_request(format!("invalid JSON body: {rejection}"))
}

/// Sign `/v1/decide*` and `/v1/decisions` bodies with HMAC-SHA256 over the exact bytes.
async fn sign_responses(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let path = request.uri().path().to_owned();
    let response = next.run(request).await;
    let Some(key) = state.signing_key.as_deref() else {
        return response;
    };
    if !(path.starts_with("/v1/decide") || path.starts_with("/v1/decisions")) {
        return response;
    }

    let (mut parts, body) = response.into_parts();
    let bytes = match to_bytes(body, SIGNATURE_BODY_LIMIT).await {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::error!(%error, "could not buffer response body for signing");
            return ApiError::internal("could not buffer response body for signing")
                .into_response();
        }
    };
    let mut mac = match Hmac::<Sha256>::new_from_slice(key) {
        Ok(mac) => mac,
        Err(error) => {
            tracing::error!(%error, "invalid HMAC key");
            return ApiError::internal("invalid signing key").into_response();
        }
    };
    mac.update(&bytes);
    let signature = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));
    parts.headers.remove(CONTENT_LENGTH);
    match HeaderValue::from_str(&signature) {
        Ok(value) => {
            parts.headers.insert(SIGNATURE_HEADER, value);
        }
        Err(error) => {
            tracing::error!(%error, "could not encode signature header");
            return ApiError::internal("could not encode signature header").into_response();
        }
    }
    Response::from_parts(parts, Body::from(bytes))
}

/// Low-cardinality HTTP metrics. Paths are fixed route templates on this server.
async fn observe_http(State(_state): State<AppState>, request: Request, next: Next) -> Response {
    let method = request.method().as_str().to_owned();
    let path = request.uri().path().to_owned();
    let started = Instant::now();
    metrics::gauge!("jev_http_requests_in_flight").increment(1.0);
    let response = next.run(request).await;
    metrics::gauge!("jev_http_requests_in_flight").decrement(1.0);
    metrics::counter!(
        "jev_http_requests_total",
        "method" => method,
        "path" => path.clone(),
        "status" => response.status().as_u16().to_string()
    )
    .increment(1);
    metrics::histogram!("jev_http_request_duration_seconds", "path" => path)
        .record(started.elapsed().as_secs_f64());
    response
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{decisions_state, option_line};

    #[test]
    fn decisions_state_joins_text_messages_and_parts() {
        let input = json!([
            {"role": "user", "content": "first"},
            {"role": "user", "content": [
                {"type": "input_text", "text": "second"},
                {"type": "input_text", "text": "third"}
            ]}
        ]);
        assert_eq!(
            decisions_state(&input).unwrap(),
            Some("first\n\nsecond\n\nthird".to_owned())
        );
    }

    #[test]
    fn decisions_state_rejects_images() {
        let input = json!([{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        ]}]);
        assert_eq!(decisions_state(&input).unwrap(), None);
    }

    #[test]
    fn option_line_matches_python_truthiness() {
        assert_eq!(
            option_line(&json!("a"), &json!("description")),
            "a: description"
        );
        assert_eq!(option_line(&json!("a"), &json!("")), "a");
        assert_eq!(option_line(&json!("a"), &json!(null)), "a");
        assert_eq!(option_line(&json!(1), &json!(0)), "1");
        assert_eq!(option_line(&json!(1), &json!("x")), "1: x");
    }
}

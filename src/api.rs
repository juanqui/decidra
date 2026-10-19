//! Axum application: JEV decision endpoints, OpenAI-compatible decisions, passthrough,
//! HMAC signing, and Prometheus metrics.

use std::{path::PathBuf, sync::Arc, time::Instant};

use anyhow::{Context, Result, bail};
use axum::{
    Json, Router,
    body::{Body, Bytes, to_bytes},
    extract::{DefaultBodyLimit, Request, State, rejection::JsonRejection},
    http::{
        HeaderValue,
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

use crate::{
    DEFAULT_STRATEGY, MAX_OPTIONS, NATIVE_CHOICE_OPTIONS, PROTOCOL, PUBLIC_ALIAS, PUBLIC_MODEL,
    config::Config,
    engine::{DecisionEngine, Kind, Strategy, Usage},
    error::ApiError,
    head::{self, FittedKind},
    prompt::{self, LABEL_CONTEXT},
    upstream::Upstream,
};

const SIGNATURE_HEADER: &str = "x-jev-signature";
const SIGNATURE_BODY_LIMIT: usize = 128 * 1024 * 1024;

/// Startup diagnostics, also printed by `--check-config`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LoadReport {
    pub checkpoint: PathBuf,
    pub tokenizer: PathBuf,
    pub head: PathBuf,
    pub calibration: PathBuf,
    pub fitted: Option<PathBuf>,
    pub max_options: usize,
    pub native_choice_options: usize,
    pub temperatures: Value,
    pub vllm_url: String,
    pub model: String,
    pub signing_enabled: bool,
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
    /// Load and validate the tokenizer, decision head, calibration, and HTTP client.
    pub fn load(config: Config) -> Result<Self> {
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

        let upstream = Arc::new(
            Upstream::new(
                config.vllm_url.clone(),
                config.vllm_model.clone(),
                config.max_concurrency,
                config.request_timeout,
                config.connect_timeout,
            )
            .context("building vLLM client")?,
        );

        let native_choice_options = loaded.head.choice[1] - loaded.head.choice[0];
        let report = LoadReport {
            checkpoint: config.checkpoint.clone(),
            tokenizer: tokenizer_path,
            head: loaded.head_path.clone(),
            calibration: loaded.calibration_path.clone(),
            fitted: loaded.fitted_path.clone(),
            max_options: label_ids.len().min(MAX_OPTIONS),
            native_choice_options,
            temperatures: loaded.temperatures.raw.clone(),
            vllm_url: config.vllm_url.clone(),
            model: config.vllm_model.clone(),
            signing_enabled: config.signing_enabled(),
        };

        let fitted: Option<Arc<std::collections::HashMap<String, FittedKind>>> =
            loaded.fitted.map(Arc::new);
        let engine = Arc::new(DecisionEngine::new(
            Arc::new(loaded.head),
            Arc::new(loaded.temperatures),
            fitted,
            label_ids,
            upstream,
        ));

        Ok(Self {
            engine,
            signing_key: config.signing_key.clone().map(Arc::new),
            config: Arc::new(config),
            metrics: None,
            load_report: report,
        })
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
    let data: Vec<Value> = [
        PUBLIC_MODEL,
        PUBLIC_ALIAS,
        state.engine.upstream.model.as_str(),
    ]
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
    let kind = Kind::parse(&request.kind)
        .ok_or_else(|| ApiError::bad_request("kind must be noul|score|choice"))?;
    prompt::validate_state(&request.state).map_err(ApiError::bad_request)?;
    let images = prompt::image_count(&request.state);
    if images > 1 {
        return Err(ApiError::bad_request(format!(
            "this backend reads one image per request, got {images}"
        )));
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

    Ok(Json(json!({
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
    })))
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
            "input_image content parts are not supported: this front is text-only",
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

async fn chat_completions(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Response, ApiError> {
    proxy(state, "/v1/chat/completions", body).await
}

async fn completions(State(state): State<AppState>, body: Bytes) -> Result<Response, ApiError> {
    proxy(state, "/v1/completions", body).await
}

async fn proxy(state: AppState, path: &'static str, body: Bytes) -> Result<Response, ApiError> {
    let mut value: Value = serde_json::from_slice(&body)
        .map_err(|error| ApiError::bad_request(format!("invalid JSON body: {error}")))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("request body must be a JSON object"))?;
    object.insert(
        "model".to_owned(),
        Value::String(state.engine.upstream.model.clone()),
    );
    let streaming = object.get("stream").and_then(Value::as_bool) == Some(true);

    if streaming {
        return state
            .engine
            .upstream
            .proxy_stream(path, &value)
            .await
            .map_err(ApiError::from);
    }

    let upstream = state
        .engine
        .upstream
        .proxy_json(path, &value)
        .await
        .map_err(ApiError::from)?;
    let mut response = Response::new(Body::from(upstream.body));
    *response.status_mut() = upstream.status;
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_str(&upstream.content_type)
            .unwrap_or_else(|_| HeaderValue::from_static("application/json")),
    );
    Ok(response)
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

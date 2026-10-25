//! d3-turbo HTTP API: `/v1/systemone`, health, models, metrics.
//!
//! There is no generic `/v1/decide`, OpenAI-compatible route, signing, or
//! model-family switch. This server accepts only d3-turbo `prefix_bidir`
//! packages with a separate F32 readout.

use std::{sync::Arc, time::SystemTime};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use metrics_exporter_prometheus::PrometheusHandle;
use serde_json::{Value, json};

use crate::{
    backend::Backend,
    config::Config,
    error::EngineError,
    image_media::{MAX_ENCODED, MAX_IMAGES},
};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub backend: Arc<dyn Backend>,
    pub metrics: Option<PrometheusHandle>,
}

impl AppState {
    pub fn new(config: Config, backend: Arc<dyn Backend>) -> Self {
        Self {
            config: Arc::new(config),
            backend,
            metrics: None,
        }
    }

    pub fn with_metrics(mut self, handle: PrometheusHandle) -> Self {
        self.metrics = Some(handle);
        self
    }

    pub fn router(self) -> Router {
        let body_limit = self.config.max_body_bytes;
        let state = self;
        Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(models))
            .route("/v1/systemone", post(systemone))
            .route("/metrics", get(metrics_endpoint))
            .layer(DefaultBodyLimit::max(body_limit))
            .with_state(state)
    }
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ready",
        "model": state.config.model_id,
        "head": "d3-code-readout",
    }))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.config.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "vllm-sr",
        }],
    }))
}

async fn metrics_endpoint(State(state): State<AppState>) -> Response {
    match &state.metrics {
        Some(handle) => (StatusCode::OK, handle.render()).into_response(),
        None => (StatusCode::NOT_FOUND, "metrics disabled").into_response(),
    }
}

async fn systemone(State(state): State<AppState>, body: Bytes) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(_) => {
            return error(
                StatusCode::BAD_REQUEST,
                "send a JSON body with state and questions",
            );
        }
    };
    let Some(object) = body.as_object() else {
        return error(StatusCode::BAD_REQUEST, "send a JSON object");
    };
    let Some(questions) = object.get("questions").and_then(Value::as_object) else {
        return error(
            StatusCode::BAD_REQUEST,
            "questions must be a non-empty object",
        );
    };
    if questions.is_empty() {
        return error(
            StatusCode::BAD_REQUEST,
            "questions must be a non-empty object",
        );
    }
    let state_value = object
        .get("state")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let images: Vec<String> = match object.get("images") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(values)) => {
            if values.len() > MAX_IMAGES {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!("at most {MAX_IMAGES} images per request"),
                );
            }
            let max_encoded = MAX_ENCODED.saturating_mul(4).div_ceil(3) + 128;
            let mut out = Vec::with_capacity(values.len());
            for (i, value) in values.iter().enumerate() {
                let Some(value) = value.as_str() else {
                    return error(
                        StatusCode::BAD_REQUEST,
                        format!("images[{i}] must be a data URL string"),
                    );
                };
                if value.len() > max_encoded {
                    return error(
                        StatusCode::BAD_REQUEST,
                        format!("images[{i}] exceeds the encoded image limit"),
                    );
                }
                out.push(value.to_owned());
            }
            out
        }
        Some(_) => return error(StatusCode::BAD_REQUEST, "images must be an array"),
    };

    metrics::counter!("decidra_systemone_requests_total").increment(1);
    match state
        .backend
        .d3_answers(&state_value, questions, images)
        .await
    {
        Ok(mut value) => {
            if let Some(object) = value.as_object_mut() {
                object.insert(
                    "object".to_owned(),
                    Value::String("structured_decision".to_owned()),
                );
                object.insert(
                    "model".to_owned(),
                    Value::String(state.config.model_id.clone()),
                );
                object.insert(
                    "created".to_owned(),
                    Value::from(
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0),
                    ),
                );
            }
            Json(value).into_response()
        }
        Err(EngineError::Config(message)) | Err(EngineError::Unsupported(message)) => {
            error(StatusCode::BAD_REQUEST, message)
        }
        Err(EngineError::Overloaded(message)) => error(StatusCode::SERVICE_UNAVAILABLE, message),
        Err(EngineError::Timeout(_)) => {
            error(StatusCode::GATEWAY_TIMEOUT, "d3-turbo request timed out")
        }
        Err(EngineError::Backend(message)) => {
            tracing::error!(%message, "d3-turbo backend failed");
            if message.contains("memory") || message.contains("out of capacity") {
                error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "d3-turbo is out of capacity",
                )
            } else {
                error(StatusCode::BAD_GATEWAY, "d3-turbo backend failed")
            }
        }
        Err(EngineError::BadResponse(message)) => {
            tracing::error!(%message, "d3-turbo returned an invalid response");
            error(StatusCode::BAD_GATEWAY, "d3-turbo backend failed")
        }
        Err(EngineError::Internal(message)) => {
            tracing::error!(%message, "d3-turbo internal error");
            error(StatusCode::INTERNAL_SERVER_ERROR, "internal error")
        }
    }
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({"detail": message.into()}))).into_response()
}

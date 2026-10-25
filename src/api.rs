//! d3-turbo HTTP API: `/v1/systemone`, health, models, metrics.
//!
//! There is no generic `/v1/decide`, OpenAI-compatible route, signing, or
//! model-family switch. This server accepts only d3-turbo `prefix_bidir`
//! packages with a separate F32 readout.

use std::{sync::Arc, time::SystemTime};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    middleware::{self, Next},
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
    request_slots: Arc<tokio::sync::Semaphore>,
}

impl AppState {
    pub fn new(config: Config, backend: Arc<dyn Backend>) -> Self {
        Self {
            config: Arc::new(config),
            backend,
            metrics: None,
            request_slots: Arc::new(tokio::sync::Semaphore::new(128)),
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
            .layer(middleware::from_fn_with_state(state.clone(), admit_request))
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

/// Bound in-flight requests before the body extractor allocates up to the
/// 64 MiB body limit. Image waiters retain their parsed bodies, so this is the
/// outer memory-admission gate; the 64 decoder slots queue behind it.
async fn admit_request(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let permit = match state.request_slots.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "server is busy; retry shortly",
            );
        }
    };
    let response = next.run(request).await;
    drop(permit);
    response
}

async fn systemone(State(state): State<AppState>, body: Bytes) -> Response {
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(_) => {
            return error(
                StatusCode::BAD_REQUEST,
                "send a JSON body with state and questions",
            );
        }
    };
    drop(body);
    let body = parsed;
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
    let mut state_value = object
        .get("state")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let mut images: Vec<String> = match object.get("images") {
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
    if let Err(detail) = extract_state_images(&mut state_value, &mut images) {
        return error(StatusCode::BAD_REQUEST, detail);
    }
    if images.len() > MAX_IMAGES {
        return error(
            StatusCode::BAD_REQUEST,
            format!("at most {MAX_IMAGES} images per request"),
        );
    }

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

/// OpenRouter/System One content parts may carry images inside `state`:
/// `[{"type":"text",...},{"type":"image_url","image_url":{"url":"data:..."}}]`.
/// Hoist those images into the engine's image list and leave the remaining
/// text as the state body. Arrays without image parts keep their existing
/// JSON state semantics.
fn extract_state_images(state: &mut Value, images: &mut Vec<String>) -> Result<(), String> {
    let Value::Array(parts) = state else {
        return Ok(());
    };
    let has_image = parts
        .iter()
        .any(|part| part.get("type").and_then(Value::as_str) == Some("image_url"));
    let unsupported_media =
        parts
            .iter()
            .find_map(|part| match part.get("type").and_then(Value::as_str) {
                Some(kind @ ("video_url" | "input_audio" | "file")) => Some(kind),
                _ => None,
            });
    if !has_image {
        if let Some(kind) = unsupported_media {
            return Err(format!(
                "{kind} content in state is not supported by this engine"
            ));
        }
        return Ok(());
    }
    let mut text = String::new();
    for (index, part) in parts.iter().enumerate() {
        if let Value::String(value) = part {
            text.push_str(value);
            continue;
        }
        let kind = part
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("state[{index}] is missing a content type"))?;
        match kind {
            "text" => {
                let value = part
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("state[{index}].text must be a string"))?;
                text.push_str(value);
            }
            "image_url" => {
                let value = match part.get("image_url") {
                    Some(Value::String(url)) => url,
                    Some(image) => image
                        .get("url")
                        .and_then(Value::as_str)
                        .ok_or_else(|| format!("state[{index}].image_url.url must be a string"))?,
                    None => {
                        return Err(format!("state[{index}].image_url is missing"));
                    }
                };
                images.push(value.to_owned());
            }
            other => {
                return Err(format!(
                    "state content type {other:?} is not supported by this engine"
                ));
            }
        }
    }
    *state = Value::String(text);
    Ok(())
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({"detail": message.into()}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::extract_state_images;
    use serde_json::json;

    #[test]
    fn state_content_parts_hoist_images_and_join_text() {
        let mut state = json!([
            "A held part.",
            {"type": "text", "text": " plus text."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "Second paragraph."}
        ]);
        let mut images = vec!["data:image/jpeg;base64,BBBB".to_owned()];
        extract_state_images(&mut state, &mut images).unwrap();
        assert_eq!(state, json!("A held part. plus text.Second paragraph."));
        assert_eq!(
            images,
            vec!["data:image/jpeg;base64,BBBB", "data:image/png;base64,AAAA"]
        );
    }

    #[test]
    fn image_only_state_becomes_empty() {
        let mut state = json!([
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ]);
        let mut images = Vec::new();
        extract_state_images(&mut state, &mut images).unwrap();
        assert_eq!(state, json!(""));
        assert_eq!(images.len(), 1);
    }

    #[test]
    fn plain_arrays_keep_their_json_state() {
        let mut state = json!(["one", 2, {"three": true}]);
        let original = state.clone();
        let mut images = Vec::new();
        extract_state_images(&mut state, &mut images).unwrap();
        assert_eq!(state, original);
        assert!(images.is_empty());
    }

    #[test]
    fn malformed_and_unsupported_parts_are_rejected() {
        let mut images = Vec::new();
        let mut missing = json!([{"type": "image_url"}]);
        assert!(extract_state_images(&mut missing, &mut images).is_err());
        for unsupported in [
            json!([{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]),
            json!([{"type": "input_audio", "input_audio": {}}]),
            json!([{"type": "file", "file": {}}]),
        ] {
            let mut unsupported = unsupported;
            assert!(extract_state_images(&mut unsupported, &mut images).is_err());
        }
    }
}

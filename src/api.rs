//! d3-turbo HTTP API: `/v1/systemone`, health, models, metrics.
//!
//! There is no generic `/v1/decide`, OpenAI-compatible route, signing, or
//! model-family switch. This server accepts only d3-turbo `prefix_bidir`
//! packages with a separate F32 readout.

use std::{
    sync::{Arc, Mutex, PoisonError},
    time::{Duration, Instant, SystemTime},
};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, MatchedPath, Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use metrics_exporter_prometheus::PrometheusHandle;
use serde_json::{Map, Value, json};

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
    metrics_cache: Arc<Mutex<Option<CachedMetrics>>>,
}

struct CachedMetrics {
    at: Instant,
    body: String,
}

impl AppState {
    pub fn new(config: Config, backend: Arc<dyn Backend>) -> Self {
        Self {
            config: Arc::new(config),
            backend,
            metrics: None,
            request_slots: Arc::new(tokio::sync::Semaphore::new(128)),
            metrics_cache: Arc::new(Mutex::new(None)),
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
            // Outermost: sees admission 503s and body-limit 413s.
            .layer(middleware::from_fn(track_http))
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
    let Some(handle) = &state.metrics else {
        return (
            StatusCode::NOT_FOUND,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            "metrics disabled\n",
        )
            .into_response();
    };
    let mut cache = state
        .metrics_cache
        .lock()
        .unwrap_or_else(|error: PoisonError<_>| error.into_inner());
    if let Some(cached) = cache.as_ref() {
        if cached.at.elapsed() < Duration::from_secs(1) {
            return metrics_response(cached.body.clone());
        }
    }
    let mut body = handle.render();
    if !body.ends_with('\n') {
        body.push('\n');
    }
    *cache = Some(CachedMetrics {
        at: Instant::now(),
        body: body.clone(),
    });
    metrics_response(body)
}

fn metrics_response(body: String) -> Response {
    (
        StatusCode::OK,
        [
            (
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            ),
            (header::CACHE_CONTROL, "no-store"),
        ],
        body,
    )
        .into_response()
}

/// Bound in-flight requests before the body extractor allocates up to the
/// 64 MiB body limit. Image waiters retain their parsed bodies, so this is the
/// outer memory-admission gate; decoder slots queue behind it. `/metrics` must
/// stay scrapeable when the API is saturated, so it bypasses the gate.
async fn admit_request(State(state): State<AppState>, request: Request, next: Next) -> Response {
    if request
        .extensions()
        .get::<MatchedPath>()
        .is_some_and(|path| path.as_str() == "/metrics")
    {
        return next.run(request).await;
    }
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

#[derive(Clone, Copy)]
enum Outcome {
    Success,
    InvalidRequest,
    Unsupported,
    Overloaded,
    Timeout,
    BackendError,
    InternalError,
}

impl Outcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::InvalidRequest => "invalid_request",
            Self::Unsupported => "unsupported",
            Self::Overloaded => "overloaded",
            Self::Timeout => "timeout",
            Self::BackendError => "backend_error",
            Self::InternalError => "internal_error",
        }
    }

    fn from_status(status: StatusCode) -> Self {
        match status {
            StatusCode::OK => Self::Success,
            StatusCode::BAD_REQUEST | StatusCode::PAYLOAD_TOO_LARGE => Self::InvalidRequest,
            StatusCode::SERVICE_UNAVAILABLE => Self::Overloaded,
            StatusCode::GATEWAY_TIMEOUT => Self::Timeout,
            StatusCode::BAD_GATEWAY => Self::BackendError,
            other if other.is_server_error() => Self::InternalError,
            _ => Self::InternalError,
        }
    }
}

struct InFlightGuard {
    route: String,
}

impl InFlightGuard {
    fn new(route: String) -> Self {
        metrics::gauge!(
            "decidra_http_requests_in_flight",
            "route" => route.clone()
        )
        .increment(1.0);
        Self { route }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        metrics::gauge!(
            "decidra_http_requests_in_flight",
            "route" => self.route.clone()
        )
        .decrement(1.0);
    }
}

fn method_label(method: &axum::http::Method) -> String {
    match method.as_str() {
        "GET" | "POST" | "HEAD" | "OPTIONS" => method.as_str().to_owned(),
        _ => "other".to_owned(),
    }
}

async fn track_http(request: Request, next: Next) -> Response {
    let method = method_label(request.method());
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(|path| path.as_str().to_owned())
        .unwrap_or_else(|| "unmatched".to_owned());
    if route == "/metrics" {
        return next.run(request).await;
    }
    let _in_flight = InFlightGuard::new(route.clone());
    let started = Instant::now();
    let response = next.run(request).await;
    let status = response.status().as_u16().to_string();
    metrics::counter!(
        "decidra_http_requests_total",
        "method" => method.clone(),
        "route" => route.clone(),
        "status" => status
    )
    .increment(1);
    metrics::histogram!(
        "decidra_http_request_duration_seconds",
        "method" => method.clone(),
        "route" => route.clone()
    )
    .record(started.elapsed().as_secs_f64());
    if route == "/v1/systemone" && method == "POST" {
        let outcome = response
            .extensions()
            .get::<Outcome>()
            .copied()
            .unwrap_or_else(|| Outcome::from_status(response.status()));
        metrics::counter!(
            "decidra_systemone_requests_total",
            "outcome" => outcome.as_str()
        )
        .increment(1);
        metrics::histogram!(
            "decidra_systemone_request_duration_seconds",
            "outcome" => outcome.as_str()
        )
        .record(started.elapsed().as_secs_f64());
    }
    response
}

async fn systemone(State(state): State<AppState>, body: Bytes) -> Response {
    let (mut response, outcome) = match parse_systemone(&body) {
        Ok((state_value, questions, images)) => {
            match state
                .backend
                .d3_answers(&state_value, &questions, images)
                .await
            {
                Ok(value) => (success_response(&state.config, value), Outcome::Success),
                Err(error) => error_response(error),
            }
        }
        Err(message) => (
            error(StatusCode::BAD_REQUEST, message),
            Outcome::InvalidRequest,
        ),
    };
    response.extensions_mut().insert(outcome);
    response
}

fn success_response(config: &Config, mut value: Value) -> Response {
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "object".to_owned(),
            Value::String("structured_decision".to_owned()),
        );
        object.insert("model".to_owned(), Value::String(config.model_id.clone()));
        object.insert(
            "created".to_owned(),
            Value::from(
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|duration| duration.as_secs())
                    .unwrap_or(0),
            ),
        );
    }
    Json(value).into_response()
}

fn error_response(failure: EngineError) -> (Response, Outcome) {
    match failure {
        EngineError::Config(message) => (
            error(StatusCode::BAD_REQUEST, message),
            Outcome::InvalidRequest,
        ),
        EngineError::Unsupported(message) => (
            error(StatusCode::BAD_REQUEST, message),
            Outcome::Unsupported,
        ),
        EngineError::Overloaded(message) => (
            error(StatusCode::SERVICE_UNAVAILABLE, message),
            Outcome::Overloaded,
        ),
        EngineError::Timeout(_) => (
            error(StatusCode::GATEWAY_TIMEOUT, "d3-turbo request timed out"),
            Outcome::Timeout,
        ),
        EngineError::Backend(message) => {
            tracing::error!(%message, "d3-turbo backend failed");
            if message.contains("memory") || message.contains("out of capacity") {
                (
                    error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "d3-turbo is out of capacity",
                    ),
                    Outcome::Overloaded,
                )
            } else {
                (
                    error(StatusCode::BAD_GATEWAY, "d3-turbo backend failed"),
                    Outcome::BackendError,
                )
            }
        }
        EngineError::BadResponse(message) => {
            tracing::error!(%message, "d3-turbo returned an invalid response");
            (
                error(StatusCode::BAD_GATEWAY, "d3-turbo backend failed"),
                Outcome::BackendError,
            )
        }
        EngineError::Internal(message) => {
            tracing::error!(%message, "d3-turbo internal error");
            (
                error(StatusCode::INTERNAL_SERVER_ERROR, "internal error"),
                Outcome::InternalError,
            )
        }
    }
}

type ParsedSystemone = (Value, Map<String, Value>, Vec<String>);

fn parse_systemone(body: &[u8]) -> Result<ParsedSystemone, String> {
    let parsed: Value = serde_json::from_slice(body)
        .map_err(|_| "send a JSON body with state and questions".to_owned())?;
    let object = parsed
        .as_object()
        .ok_or_else(|| "send a JSON object".to_owned())?;
    let questions = object
        .get("questions")
        .and_then(Value::as_object)
        .ok_or_else(|| "questions must be a non-empty object".to_owned())?;
    if questions.is_empty() {
        return Err("questions must be a non-empty object".to_owned());
    }
    let mut state_value = object
        .get("state")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let mut images: Vec<String> = match object.get("images") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(values)) => {
            if values.len() > MAX_IMAGES {
                return Err(format!("at most {MAX_IMAGES} images per request"));
            }
            let max_encoded = MAX_ENCODED.saturating_mul(4).div_ceil(3) + 128;
            let mut out = Vec::with_capacity(values.len());
            for (i, value) in values.iter().enumerate() {
                let Some(value) = value.as_str() else {
                    return Err(format!("images[{i}] must be a data URL string"));
                };
                if value.len() > max_encoded {
                    return Err(format!("images[{i}] exceeds the encoded image limit"));
                }
                out.push(value.to_owned());
            }
            out
        }
        Some(_) => return Err("images must be an array".to_owned()),
    };
    extract_state_images(&mut state_value, &mut images)?;
    if images.len() > MAX_IMAGES {
        return Err(format!("at most {MAX_IMAGES} images per request"));
    }
    Ok((state_value, questions.clone(), images))
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
    use super::{AppState, extract_state_images};
    use crate::{
        backend::MockBackend,
        config::Config,
        telemetry::{self, BuildInfo},
    };
    use axum::{
        body::Body,
        http::{Request, StatusCode, header},
    };
    use serde_json::{Value, json};
    use std::sync::{Arc, LazyLock};
    use tower::ServiceExt;

    static METRICS: LazyLock<metrics_exporter_prometheus::PrometheusHandle> = LazyLock::new(|| {
        telemetry::install(
            &Config::for_test(),
            &BuildInfo {
                version: "test",
                profile: "debug",
                compiled_arch: "sm_89".to_owned(),
                native: false,
            },
        )
        .expect("installing test metrics recorder")
    });

    fn metrics_handle() -> &'static metrics_exporter_prometheus::PrometheusHandle {
        &METRICS
    }

    fn state(max_body_bytes: usize, metrics: bool) -> AppState {
        let mut config = Config::for_test();
        config.max_body_bytes = max_body_bytes;
        let backend = Arc::new(MockBackend::new(json!({
            "choice": "yes",
            "noul": 0.9,
            "score": 1.0,
        })));
        let state = AppState::new(config, backend);
        if metrics {
            state.with_metrics(metrics_handle().clone())
        } else {
            state
        }
    }

    async fn send(
        router: axum::Router,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> axum::response::Response {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                body.map_or_else(String::new, |body| body.to_string()),
            ))
            .expect("test request");
        router.oneshot(request).await.expect("router response")
    }

    async fn text(response: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body");
        String::from_utf8(bytes.to_vec()).expect("utf-8 response")
    }

    #[tokio::test]
    async fn metrics_contract_covers_success_invalid_body_limit_and_admission() {
        let valid = json!({"state": "ok", "questions": {"q": {"type": "noul"}}});

        let router = state(1024, true).router();
        let response = send(router.clone(), "POST", "/v1/systemone", Some(valid.clone())).await;
        assert_eq!(response.status(), StatusCode::OK);
        let response = send(router.clone(), "GET", "/metrics", None).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/plain; version=0.0.4; charset=utf-8"
        );
        let exposition = text(response).await;
        let cached = text(send(router, "GET", "/metrics", None).await).await;
        assert_eq!(
            exposition, cached,
            "the one-second scrape cache should single-flight"
        );
        assert!(exposition.contains(
            "decidra_systemone_requests_total{model_name=\"test/model\",outcome=\"success\"}"
        ));
        assert!(exposition.contains("route=\"/v1/systemone\""));
        assert!(exposition.contains("status=\"200\""));
        assert!(exposition.contains("decidra_systemone_request_duration_seconds_count"));
        assert!(!exposition.contains("route=\"/metrics\""));
        assert!(exposition.contains("decidra_engine_worker_alive"));

        let router = state(1024, true).router();
        let response = send(router.clone(), "POST", "/v1/systemone", Some(json!({}))).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let exposition = text(send(router, "GET", "/metrics", None).await).await;
        assert!(exposition.contains("outcome=\"invalid_request\""));

        let router = state(16, true).router();
        let response = send(
            router.clone(),
            "POST",
            "/v1/systemone",
            Some(json!({"state": "x".repeat(64), "questions": {"q": {}}})),
        )
        .await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let exposition = text(send(router, "GET", "/metrics", None).await).await;
        assert!(exposition.contains("outcome=\"invalid_request\""));

        let admission_state = state(1024, true);
        let mut held = Vec::new();
        for _ in 0..128 {
            held.push(
                admission_state
                    .request_slots
                    .clone()
                    .try_acquire_owned()
                    .expect("test request slot"),
            );
        }
        let router = admission_state.router();
        let response = send(router.clone(), "POST", "/v1/systemone", Some(valid)).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let response = send(router, "GET", "/metrics", None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let exposition = text(response).await;
        assert!(exposition.contains("outcome=\"overloaded\""));
        drop(held);

        let router = state(1024, false).router();
        let response = send(router, "GET", "/metrics", None).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

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

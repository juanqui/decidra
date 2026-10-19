//! Error types shared by the decision engine and the HTTP layer.

use std::time::Duration;

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

/// Errors raised while building requests to, or interpreting answers from, vLLM.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("upstream request failed: {0}")]
    Upstream(String),
    #[error("upstream request timed out after {0:?}")]
    Timeout(Duration),
    #[error("upstream returned HTTP {status}: {body}")]
    UpstreamStatus { status: u16, body: String },
    #[error("invalid upstream response: {0}")]
    BadResponse(String),
    #[error("internal error: {0}")]
    Internal(String),
}

/// An API error with an OpenAI-compatible envelope plus FastAPI's historical `detail` field.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    pub fn bad_gateway(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            kind: "upstream_error",
            message: message.into(),
        }
    }

    pub fn gateway_timeout(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            kind: "upstream_timeout",
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "internal_error",
            message: message.into(),
        }
    }
}

impl From<EngineError> for ApiError {
    fn from(error: EngineError) -> Self {
        match error {
            EngineError::Config(message) => ApiError::internal(message),
            EngineError::Timeout(timeout) => {
                ApiError::gateway_timeout(format!("upstream request timed out after {timeout:?}"))
            }
            EngineError::Upstream(message) => {
                ApiError::bad_gateway(format!("upstream request failed: {message}"))
            }
            EngineError::UpstreamStatus { status, body } => {
                ApiError::bad_gateway(format!("backend: HTTP {status}: {body}"))
            }
            EngineError::BadResponse(message) => {
                ApiError::bad_gateway(format!("invalid backend response: {message}"))
            }
            EngineError::Internal(message) => ApiError::internal(message),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let code = self.status.as_u16();
        let body = json!({
            "detail": self.message,
            "error": {
                "message": self.message,
                "type": self.kind,
                "code": code,
            }
        });
        (self.status, Json(body)).into_response()
    }
}

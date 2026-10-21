//! Error types shared by the decision engine and the HTTP layer.

use std::time::Duration;

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

/// Errors raised while loading the native backend or running a decision pass.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("backend request failed: {0}")]
    Backend(String),
    #[error("backend request timed out after {0:?}")]
    Timeout(Duration),
    #[error("invalid backend response: {0}")]
    BadResponse(String),
    #[error("unsupported by the native backend: {0}")]
    Unsupported(String),
    #[error("internal error: {0}")]
    Internal(String),
    #[error("overloaded: {0}")]
    Overloaded(String),
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
            kind: "backend_error",
            message: message.into(),
        }
    }

    pub fn gateway_timeout(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            kind: "backend_timeout",
            message: message.into(),
        }
    }

    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            kind: "not_implemented_error",
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
                ApiError::gateway_timeout(format!("backend request timed out after {timeout:?}"))
            }
            EngineError::Backend(message) => {
                ApiError::bad_gateway(format!("backend request failed: {message}"))
            }
            EngineError::BadResponse(message) => {
                ApiError::bad_gateway(format!("invalid backend response: {message}"))
            }
            EngineError::Unsupported(message) => ApiError::not_implemented(message),
            EngineError::Internal(message) => ApiError::internal(message),
            EngineError::Overloaded(message) => ApiError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                kind: "overloaded_error",
                message,
            },
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

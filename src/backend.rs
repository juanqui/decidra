//! The serving backend contract: one d3-turbo state-once decision request.

use std::{future::Future, pin::Pin};

use serde_json::{Map, Value};

use crate::error::EngineError;

pub type D3Future<'a> = Pin<Box<dyn Future<Output = Result<Value, EngineError>> + Send + 'a>>;

/// Typed decision backend for `POST /v1/systemone`.
///
/// `state` is the raw state value (string, JSON object, or an array containing
/// media; the native text path rejects media). `questions` is the ordered map
/// from question id to the typed question object. The returned value has
/// `answers` and `usage`; the HTTP layer adds the model/object envelope.
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;

    fn d3_answers<'a>(
        &'a self,
        state: &'a Value,
        questions: &'a Map<String, Value>,
    ) -> D3Future<'a>;
}

/// Deterministic test backend. Kept small so API tests do not need a GPU.
pub struct MockBackend {
    name: String,
    answer: Value,
}

impl MockBackend {
    pub fn new(answer: Value) -> Self {
        Self {
            name: "mock".to_owned(),
            answer,
        }
    }
}

impl Backend for MockBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn d3_answers<'a>(
        &'a self,
        _state: &'a Value,
        questions: &'a Map<String, Value>,
    ) -> D3Future<'a> {
        let answer = self.answer.clone();
        let questions = questions.clone();
        Box::pin(async move {
            let mut answers = Map::new();
            for name in questions.keys() {
                answers.insert(name.clone(), answer.clone());
            }
            Ok(serde_json::json!({
                "answers": answers,
                "usage": {"input_tokens": 0u64, "output_tokens": 0u64},
            }))
        })
    }
}

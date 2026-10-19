//! Logit backend abstraction.
//!
//! `DecisionEngine` only needs raw head logits for a prompt and a list of token
//! ids. The shipped implementation is [`NativeBackend`](crate::native_backend)
//! (in-process `jev-native` CUDA engine); [`MockBackend`] keeps the API tests
//! independent of a GPU and of the native crate.

use std::{future::Future, pin::Pin, sync::Arc};

use crate::error::EngineError;

/// Boxed future returned by [`Backend::logits`]. Keeping the trait object-safe
/// lets `AppState` hold `Arc<dyn Backend>`.
pub type LogitsFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<f32>, EngineError>> + Send + 'a>>;

/// Raw inference backend for the decision head.
///
/// `logits` receives the exact text prompt and the head token ids whose logits
/// are needed at the final position. The returned vector is positionally
/// aligned with `ids`. Values may be raw logits or log-probabilities: the head
/// softmax is insensitive to a constant shift (the normalizer cancels).
pub trait Backend: Send + Sync {
    fn logits<'a>(&'a self, prompt: &'a str, ids: &'a [u32]) -> LogitsFuture<'a>;

    /// Prompt token count for usage accounting. Best effort; `0` when the
    /// backend cannot report it without a full tokenizer pass.
    fn prompt_tokens(&self, _prompt: &str) -> u64 {
        0
    }

    /// Short backend name used in diagnostics (`/v1/models`, logs).
    fn name(&self) -> &str;
}

type MockFn = dyn Fn(&str, &[u32]) -> Result<Vec<f32>, EngineError> + Send + Sync;

/// Deterministic in-process backend for tests and local API development.
pub struct MockBackend {
    name: String,
    prompt_tokens: u64,
    logits: Arc<MockFn>,
}

impl MockBackend {
    pub fn new<F>(logits: F) -> Self
    where
        F: Fn(&str, &[u32]) -> Result<Vec<f32>, EngineError> + Send + Sync + 'static,
    {
        Self {
            name: "mock".to_owned(),
            prompt_tokens: 0,
            logits: Arc::new(logits),
        }
    }

    /// Fixed prompt-token count reported in usage (tests assert exact totals).
    pub fn with_prompt_tokens(mut self, prompt_tokens: u64) -> Self {
        self.prompt_tokens = prompt_tokens;
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl Backend for MockBackend {
    fn logits<'a>(&'a self, prompt: &'a str, ids: &'a [u32]) -> LogitsFuture<'a> {
        let result = (self.logits)(prompt, ids);
        Box::pin(async move { result })
    }

    fn prompt_tokens(&self, _prompt: &str) -> u64 {
        self.prompt_tokens
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::{Backend, MockBackend};

    #[tokio::test]
    async fn mock_backend_is_positional_and_configurable() {
        let backend = MockBackend::new(|prompt, ids| {
            assert_eq!(prompt, "p");
            Ok(ids.iter().map(|id| *id as f32).collect())
        })
        .with_prompt_tokens(7)
        .with_name("test");
        assert_eq!(backend.name(), "test");
        assert_eq!(backend.prompt_tokens("p"), 7);
        assert_eq!(backend.logits("p", &[3, 4]).await.unwrap(), vec![3.0, 4.0]);
    }
}

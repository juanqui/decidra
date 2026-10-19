//! vLLM HTTP client: 1-token decision read-outs, usage accounting, and OpenAI passthrough.
//!
//! One keep-alive `reqwest::Client` is shared by every request. A semaphore bounds the
//! number of concurrent upstream calls, including the parallel passes of `tournament`
//! and `permute`, so a wide choice cannot fan out without bound.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    http::{HeaderValue, StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures::{StreamExt, stream};
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{engine::Usage, error::EngineError, prompt::RAW_TEMPLATE};

/// Bound on upstream error bodies included in an error message (characters, like Python).
const ERROR_BODY_LIMIT: usize = 600;

pub struct ProxyResponse {
    pub status: StatusCode,
    pub content_type: String,
    pub body: Bytes,
}

pub struct Upstream {
    client: reqwest::Client,
    base_url: String,
    pub model: String,
    semaphore: Arc<Semaphore>,
    timeout: Duration,
}

impl Upstream {
    pub fn new(
        base_url: String,
        model: String,
        max_concurrency: usize,
        timeout: Duration,
        connect_timeout: Duration,
    ) -> Result<Self, EngineError> {
        let client = reqwest::Client::builder()
            .connect_timeout(connect_timeout)
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(60))
            .build()
            .map_err(|error| EngineError::Config(format!("building HTTP client: {error}")))?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_owned(),
            model,
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
            timeout,
        })
    }

    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    async fn acquire(&self) -> Result<OwnedSemaphorePermit, EngineError> {
        self.semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| EngineError::Internal("upstream concurrency limiter closed".to_owned()))
    }

    /// Read logprobs for up to 128 token ids per `/v1/completions` request.
    pub async fn logprobs_text(
        &self,
        prompt: &str,
        ids: &[u32],
    ) -> Result<(HashMap<u32, f64>, Usage), EngineError> {
        let mut logprobs = HashMap::with_capacity(ids.len());
        let mut usage = Usage::default();
        for chunk in ids.chunks(128) {
            let body = json!({
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "add_special_tokens": false,
                "return_tokens_as_token_ids": true,
                "logprobs": chunk.len(),
                "logprob_token_ids": chunk,
            });
            let data = self
                .send_json("/v1/completions", &body, "completions")
                .await?;
            let (prompt_tokens, completion_tokens) = parse_usage(&data);
            usage.prompt_tokens += prompt_tokens;
            usage.completion_tokens += completion_tokens;
            usage.requests += 1;
            metrics::counter!("jev_upstream_tokens_total", "direction" => "prompt")
                .increment(prompt_tokens);
            metrics::counter!("jev_upstream_tokens_total", "direction" => "completion")
                .increment(completion_tokens);
            logprobs.extend(parse_text_logprobs(&data)?);
        }
        Ok((logprobs, usage))
    }

    /// Read logprobs through the chat endpoint with the shim's raw template (image path).
    pub async fn logprobs_chat(
        &self,
        content: &[Value],
        ids: &[u32],
    ) -> Result<(HashMap<u32, f64>, Usage), EngineError> {
        let mut logprobs = HashMap::with_capacity(ids.len());
        let mut usage = Usage::default();
        for chunk in ids.chunks(128) {
            let body = json!({
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "chat_template": RAW_TEMPLATE,
                "add_generation_prompt": false,
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "add_special_tokens": false,
                "return_tokens_as_token_ids": true,
                "logprobs": true,
                "top_logprobs": 1,
                "logprob_token_ids": chunk,
            });
            let data = self
                .send_json("/v1/chat/completions", &body, "chat_completions")
                .await?;
            let (prompt_tokens, completion_tokens) = parse_usage(&data);
            usage.prompt_tokens += prompt_tokens;
            usage.completion_tokens += completion_tokens;
            usage.requests += 1;
            metrics::counter!("jev_upstream_tokens_total", "direction" => "prompt")
                .increment(prompt_tokens);
            metrics::counter!("jev_upstream_tokens_total", "direction" => "completion")
                .increment(completion_tokens);
            logprobs.extend(parse_chat_logprobs(&data)?);
        }
        Ok((logprobs, usage))
    }

    async fn send_json(
        &self,
        path: &str,
        body: &Value,
        endpoint: &'static str,
    ) -> Result<Value, EngineError> {
        let permit = self.acquire().await?;
        let started = Instant::now();
        let response = self
            .client
            .post(format!("{}{path}", self.base_url))
            .json(body)
            .timeout(self.timeout)
            .send()
            .await;
        let elapsed = started.elapsed().as_secs_f64();
        drop(permit);

        let response = match response {
            Ok(response) => response,
            Err(error) => {
                metrics::counter!(
                    "jev_upstream_requests_total",
                    "endpoint" => endpoint,
                    "status" => "error"
                )
                .increment(1);
                metrics::histogram!(
                    "jev_upstream_request_duration_seconds",
                    "endpoint" => endpoint
                )
                .record(elapsed);
                return Err(map_send_error(error, self.timeout));
            }
        };
        let status = response.status();
        metrics::counter!(
            "jev_upstream_requests_total",
            "endpoint" => endpoint,
            "status" => status.as_u16().to_string()
        )
        .increment(1);
        metrics::histogram!("jev_upstream_request_duration_seconds", "endpoint" => endpoint)
            .record(elapsed);

        let text = response.text().await.map_err(|error| {
            EngineError::Upstream(format!("reading upstream response: {error}"))
        })?;
        if !status.is_success() {
            return Err(EngineError::UpstreamStatus {
                status: status.as_u16(),
                body: text.chars().take(ERROR_BODY_LIMIT).collect(),
            });
        }
        serde_json::from_str(&text)
            .map_err(|error| EngineError::BadResponse(format!("not valid JSON: {error}")))
    }

    /// Non-streaming OpenAI passthrough. Upstream status and body are preserved.
    pub async fn proxy_json(&self, path: &str, body: &Value) -> Result<ProxyResponse, EngineError> {
        let permit = self.acquire().await?;
        let response = self
            .client
            .post(format!("{}{path}", self.base_url))
            .json(body)
            .send()
            .await
            .map_err(|error| map_send_error(error, self.timeout))?;
        drop(permit);
        let status = response.status();
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("application/json")
            .to_owned();
        let body = response.bytes().await.map_err(|error| {
            EngineError::Upstream(format!("reading upstream response: {error}"))
        })?;
        Ok(ProxyResponse {
            status,
            content_type,
            body,
        })
    }

    /// Streaming OpenAI passthrough. The concurrency permit is held for the lifetime of
    /// the relayed stream, so long generations remain inside the concurrency budget.
    pub async fn proxy_stream(&self, path: &str, body: &Value) -> Result<Response, EngineError> {
        let permit = self.acquire().await?;
        let response = self
            .client
            .post(format!("{}{path}", self.base_url))
            .json(body)
            .send()
            .await
            .map_err(|error| map_send_error(error, self.timeout))?;
        let status =
            StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("text/event-stream")
            .to_owned();
        metrics::counter!(
            "jev_upstream_requests_total",
            "endpoint" => "stream",
            "status" => status.as_u16().to_string()
        )
        .increment(1);

        let stream = response.bytes_stream();
        let guarded = stream::unfold((stream, permit), |(mut stream, permit)| async move {
            stream.next().await.map(|item| (item, (stream, permit)))
        });
        let mut response = Body::from_stream(guarded).into_response();
        *response.status_mut() = status;
        response.headers_mut().insert(
            CONTENT_TYPE,
            HeaderValue::from_str(&content_type)
                .unwrap_or_else(|_| HeaderValue::from_static("application/json")),
        );
        Ok(response)
    }
}

fn map_send_error(error: reqwest::Error, timeout: Duration) -> EngineError {
    if error.is_timeout() {
        EngineError::Timeout(timeout)
    } else if error.is_connect() {
        EngineError::Upstream(format!("connect: {error}"))
    } else {
        EngineError::Upstream(error.to_string())
    }
}

fn parse_usage(data: &Value) -> (u64, u64) {
    let usage = data.get("usage");
    (
        usage
            .and_then(|usage| usage.get("prompt_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
        usage
            .and_then(|usage| usage.get("completion_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
    )
}

fn parse_token_id(token: &str) -> Result<u32, EngineError> {
    token
        .split(':')
        .nth(1)
        .ok_or_else(|| {
            EngineError::BadResponse(format!("logprob token {token:?} has no ':id' suffix"))
        })?
        .parse::<u32>()
        .map_err(|_| {
            EngineError::BadResponse(format!("logprob token {token:?} has a non-numeric id"))
        })
}

fn parse_text_logprobs(data: &Value) -> Result<HashMap<u32, f64>, EngineError> {
    let entries = data
        .pointer("/choices/0/logprobs/top_logprobs/0")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            EngineError::BadResponse(
                "completion response has no choices[0].logprobs.top_logprobs[0] object".to_owned(),
            )
        })?;
    let mut logprobs = HashMap::with_capacity(entries.len());
    for (token, value) in entries {
        let id = parse_token_id(token)?;
        let logprob = value.as_f64().ok_or_else(|| {
            EngineError::BadResponse(format!("logprob for {token:?} is not a number"))
        })?;
        logprobs.insert(id, logprob);
    }
    Ok(logprobs)
}

fn parse_chat_logprobs(data: &Value) -> Result<HashMap<u32, f64>, EngineError> {
    let entries = data
        .pointer("/choices/0/logprobs/content/0/top_logprobs")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            EngineError::BadResponse(
                "chat response has no choices[0].logprobs.content[0].top_logprobs array".to_owned(),
            )
        })?;
    let mut logprobs = HashMap::with_capacity(entries.len());
    for entry in entries {
        let token = entry.get("token").and_then(Value::as_str).ok_or_else(|| {
            EngineError::BadResponse("chat logprob entry has no token string".to_owned())
        })?;
        let logprob = entry
            .get("logprob")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                EngineError::BadResponse("chat logprob entry has no numeric logprob".to_owned())
            })?;
        logprobs.insert(parse_token_id(token)?, logprob);
    }
    Ok(logprobs)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{parse_chat_logprobs, parse_text_logprobs};

    #[test]
    fn parses_token_id_keys_from_text_completions() {
        let data = json!({
            "choices": [{"logprobs": {"top_logprobs": [{
                "token_id:3721": -0.25,
                "token_id:1802": -1.75
            }]}}]
        });
        let logprobs = parse_text_logprobs(&data).unwrap();
        assert_eq!(logprobs[&3721], -0.25);
        assert_eq!(logprobs[&1802], -1.75);
    }

    #[test]
    fn parses_token_entries_from_chat_completions() {
        let data = json!({
            "choices": [{"logprobs": {"content": [{"top_logprobs": [
                {"token": "token_id:8", "logprob": -0.5}
            ]}]}}]
        });
        let logprobs = parse_chat_logprobs(&data).unwrap();
        assert_eq!(logprobs[&8], -0.5);
    }

    #[test]
    fn rejects_malformed_logprob_shapes() {
        assert!(parse_text_logprobs(&json!({"choices": []})).is_err());
        assert!(
            parse_chat_logprobs(&json!({
                "choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": "x"}]}]}}]
            }))
            .is_err()
        );
    }
}

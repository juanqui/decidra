//! End-to-end API tests against a self-contained mock vLLM upstream.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    Json, Router,
    body::{Body, Bytes, to_bytes},
    extract::State,
    http::{HeaderMap, Method, Request, StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response},
    routing::post,
};
use hmac::{Hmac, Mac};
use jev_engine::{api::AppState, config::Config, prompt::RAW_TEMPLATE};
use serde_json::{Map, Value, json};
use sha2::Sha256;
use tokio::net::TcpListener;
use tower::ServiceExt;

#[derive(Clone)]
struct MockState {
    requests: Arc<Mutex<Vec<Value>>>,
}

async fn mock_completion(State(state): State<MockState>, Json(body): Json<Value>) -> Response {
    state.requests.lock().unwrap().push(body.clone());
    let text = body_text(&body);
    if text.contains("FAIL") {
        return (StatusCode::INTERNAL_SERVER_ERROR, "upstream exploded").into_response();
    }
    if text.contains("SLOW") {
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    if text.contains("BADJSON") {
        return (StatusCode::OK, "this is not json").into_response();
    }
    let is_chat = body.get("messages").is_some();

    if body.get("logprob_token_ids").is_none() {
        if body.get("stream").and_then(Value::as_bool) == Some(true) {
            return Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n",
                ))
                .unwrap();
        }
        return Json(json!({
            "id": "mock-completion",
            "object": "text_completion",
            "model": body.get("model").cloned().unwrap_or(Value::Null),
            "choices": [{"text": "ok"}],
        }))
        .into_response();
    }

    let ids: Vec<u32> = body
        .get("logprob_token_ids")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(Value::as_u64)
        .map(|id| id as u32)
        .collect();
    let preferred = if text.starts_with("[kind] noul") {
        1802
    } else if text.starts_with("[kind] score") {
        18
    } else {
        34
    };
    let logprob = |id: u32| if id == preferred { 0.0 } else { -8.0 };

    if is_chat {
        let entries: Vec<Value> = ids
            .iter()
            .map(|id| json!({"token": format!("token_id:{id}"), "logprob": logprob(*id)}))
            .collect();
        Json(json!({
            "choices": [{"logprobs": {"content": [{"top_logprobs": entries}]}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 1},
        }))
        .into_response()
    } else {
        let mut entries = Map::new();
        for id in &ids {
            entries.insert(format!("token_id:{id}"), json!(logprob(*id)));
        }
        Json(json!({
            "choices": [{"logprobs": {"top_logprobs": [entries]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }))
        .into_response()
    }
}

fn body_text(body: &Value) -> String {
    if let Some(prompt) = body.get("prompt").and_then(Value::as_str) {
        return prompt.to_owned();
    }
    body.pointer("/messages/0/content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default()
}

async fn spawn_mock() -> (String, Arc<Mutex<Vec<Value>>>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/completions", post(mock_completion))
        .route("/v1/chat/completions", post(mock_completion))
        .with_state(MockState {
            requests: requests.clone(),
        });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}"), requests)
}

fn test_config(vllm_url: String) -> Config {
    Config {
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        vllm_url,
        vllm_model: "jev-awq".to_owned(),
        checkpoint: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        tokenizer: None,
        calibration: None,
        calibration_fitted: None,
        hf_snapshot: PathBuf::from("/nonexistent/hf-snapshot"),
        max_concurrency: 4,
        request_timeout: Duration::from_secs(5),
        connect_timeout: Duration::from_secs(2),
        signing_key: None,
        metrics_enabled: false,
        max_body_bytes: 8 * 1024 * 1024,
    }
}

async fn app_for() -> (Router, Arc<Mutex<Vec<Value>>>) {
    let (url, requests) = spawn_mock().await;
    let config = test_config(url);
    let state = AppState::load(config).unwrap();
    (state.router(), requests)
}

async fn app_with(config: Config) -> (Router, Arc<Mutex<Vec<Value>>>) {
    let (url, requests) = spawn_mock().await;
    let mut config = config;
    config.vllm_url = url;
    (AppState::load(config).unwrap().router(), requests)
}

async fn call(
    app: &Router,
    method: Method,
    uri: &str,
    body: Option<&Value>,
) -> (StatusCode, HeaderMap, Bytes) {
    let mut builder = Request::builder().method(method).uri(uri);
    let body = match body {
        Some(value) => {
            builder = builder.header(CONTENT_TYPE, "application/json");
            Body::from(serde_json::to_vec(value).unwrap())
        }
        None => Body::empty(),
    };
    let response = app
        .clone()
        .oneshot(builder.body(body).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, headers, bytes)
}

async fn call_raw(app: &Router, uri: &str, body: &str) -> (StatusCode, Bytes) {
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, bytes)
}

fn signature(key: &[u8], body: &[u8]) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).unwrap();
    mac.update(body);
    format!("sha256={}", hex::encode(mac.finalize().into_bytes()))
}

fn header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn decode(bytes: &Bytes) -> Value {
    serde_json::from_slice(bytes).unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[tokio::test]
async fn noul_decision_signing_metadata_and_metrics() {
    let (url, requests) = spawn_mock().await;
    let mut config = test_config(url);
    config.signing_key = Some(b"secret-key".to_vec());
    let state = AppState::load(config).unwrap();
    let app = state.router();

    let (status, headers, bytes) = call(&app, Method::GET, "/health", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(header(&headers, "x-jev-signature").is_none());
    assert_eq!(decode(&bytes)["status"], "ready");

    let (status, headers, bytes) = call(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(header(&headers, "x-jev-signature").is_none());
    let models = decode(&bytes);
    assert_eq!(models["data"].as_array().unwrap().len(), 3);
    assert_eq!(models["data"][0]["id"], "autotrust/JEV-27B-VL");

    let (status, headers, bytes) = call(&app, Method::GET, "/v1/decide/info", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        header(&headers, "x-jev-signature").as_deref(),
        Some(signature(b"secret-key", &bytes).as_str())
    );
    let info = decode(&bytes);
    assert_eq!(info["protocol"], "jev27-bare-v1");
    assert_eq!(info["max_options"], 32);
    assert_eq!(info["native_choice_options"], 16);
    assert_eq!(info["defaults"]["strategy"], "tournament");
    assert!(close(
        info["temperatures"]["noul"].as_f64().unwrap(),
        1.0143134751376188
    ));

    let body = json!({"kind": "noul", "state": "hello", "question": "Is it safe?"});
    let (status, headers, bytes) = call(&app, Method::POST, "/v1/decide", Some(&body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        header(&headers, "x-jev-signature").as_deref(),
        Some(signature(b"secret-key", &bytes).as_str())
    );
    let response = decode(&bytes);
    assert_eq!(response["kind"], "noul");
    assert_eq!(response["effective_kind"], "noul");
    assert_eq!(response["choice"], "true");
    assert_eq!(response["choice_index"], 1);
    assert_eq!(response["adaptation"], "native");
    assert_eq!(response["protocol"], "jev27-bare-v1");
    assert_eq!(response["usage"]["prompt_tokens"], 10);
    assert_eq!(response["usage"]["completion_tokens"], 1);
    assert_eq!(response["usage"]["total_tokens"], 11);
    assert_eq!(response["num_model_requests"], 1);
    assert!(response["confidence"].as_f64().unwrap() > 0.99);
    let probabilities = response["probabilities"].as_array().unwrap();
    assert_eq!(probabilities.len(), 2);
    assert!(close(
        probabilities
            .iter()
            .map(|value| value.as_f64().unwrap())
            .sum(),
        1.0
    ));

    {
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let upstream = &requests[0];
        assert_eq!(upstream["model"], "jev-awq");
        assert_eq!(upstream["max_tokens"], 1);
        assert_eq!(upstream["temperature"], 1.0);
        assert_eq!(upstream["top_p"], 1.0);
        assert_eq!(upstream["top_k"], 0);
        assert_eq!(upstream["min_p"], 0.0);
        assert_eq!(upstream["repetition_penalty"], 1.0);
        assert_eq!(upstream["add_special_tokens"], false);
        assert_eq!(upstream["return_tokens_as_token_ids"], true);
        assert_eq!(upstream["logprobs"], 2);
        assert_eq!(upstream["logprob_token_ids"], json!([3721, 1802]));
        assert_eq!(
            upstream["prompt"],
            "[kind] noul\n[state] hello\n[question] Is it safe?\n[options]\nfalse\ntrue\n[decision]:"
        );
    }

    let (status, headers, bytes) = call(&app, Method::GET, "/metrics", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(header(&headers, "x-jev-signature").is_none());
    assert!(String::from_utf8_lossy(&bytes).contains("metrics"));

    let body = json!({
        "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        ]}],
        "questions": [{"type": "predicate", "name": "p", "instructions": "is it?"}]
    });
    let (status, headers, bytes) = call(&app, Method::POST, "/v1/decisions", Some(&body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        header(&headers, "x-jev-signature").as_deref(),
        Some(signature(b"secret-key", &bytes).as_str())
    );
    assert!(String::from_utf8_lossy(&bytes).contains("input_image"));
}

#[tokio::test]
async fn choice_single_tournament_and_permute() {
    let (app, requests) = app_for().await;
    let options: Vec<String> = (0..3).map(|index| format!("option-{index}")).collect();
    let body = json!({"kind": "choice", "state": "s", "question": "q", "options": options});
    let (status, _, bytes) = call(&app, Method::POST, "/v1/decide", Some(&body)).await;
    assert_eq!(status, StatusCode::OK);
    let response = decode(&bytes);
    assert_eq!(response["adaptation"], "native");
    assert_eq!(response["choice"], "option-2");
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert_eq!(
        requests.lock().unwrap()[0]["prompt"],
        "[kind] choice\n[state] s\n[question] q\n[options]\nA) option-0\nB) option-1\nC) option-2\n[decision]:"
    );

    let wide: Vec<String> = (0..17).map(|index| format!("option-{index}")).collect();
    let body = json!({"kind": "choice", "state": "s", "question": "q", "options": wide});
    let (status, _, bytes) = call(&app, Method::POST, "/v1/decide", Some(&body)).await;
    assert_eq!(status, StatusCode::OK);
    let response = decode(&bytes);
    assert_eq!(response["adaptation"], "tournament");
    assert_eq!(response["probabilities"].as_array().unwrap().len(), 17);
    assert!(close(
        response["probabilities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .sum(),
        1.0
    ));
    // Two 9/8-option groups, then a 16-option final.
    assert_eq!(requests.lock().unwrap().len(), 1 + 3);
    assert!(String::from_utf8_lossy(&bytes).contains("tournament"));

    let body = json!({
        "kind": "choice",
        "state": "s",
        "question": "q",
        "options": wide,
        "strategy": "permute"
    });
    let (status, _, first_bytes) = call(&app, Method::POST, "/v1/decide", Some(&body)).await;
    assert_eq!(status, StatusCode::OK);
    let before = requests.lock().unwrap().len();
    let (status, _, second_bytes) = call(&app, Method::POST, "/v1/decide", Some(&body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(requests.lock().unwrap().len(), before + 4);
    let first = decode(&first_bytes);
    let second = decode(&second_bytes);
    assert_eq!(first["adaptation"], "permute");
    assert_eq!(first["probabilities"], second["probabilities"]);

    let thirty_two: Vec<String> = (0..32).map(|index| format!("option-{index}")).collect();
    let body = json!({"kind": "choice", "state": "s", "question": "q", "options": thirty_two});
    let (status, _, bytes) = call(&app, Method::POST, "/v1/decide", Some(&body)).await;
    assert_eq!(status, StatusCode::OK);
    let response = decode(&bytes);
    assert_eq!(response["choice_index"], 2);
    assert_eq!(response["choice"], "option-2");
    assert_eq!(response["adaptation"], "tournament");
}

#[tokio::test]
async fn decisions_predicate_choice_and_score_mapping() {
    let (app, requests) = app_for().await;
    let body = json!({
        "model": "client-model",
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "context one"},
            {"type": "input_text", "text": "context two"}
        ]}],
        "questions": [
            {"type": "predicate", "name": "safe", "instructions": "Is it safe?"},
            {"type": "choice", "name": "size", "instructions": "Pick a size", "choices": [
                {"value": "small", "description": "little"},
                {"value": "medium", "description": ""},
                {"value": "large", "description": "big"}
            ]},
            {"type": "score", "name": "quality", "instructions": "Rate it", "levels": [
                {"label": "bad"}, {"label": "ok"}, {"label": "good"}, {"label": "great"}
            ]}
        ]
    });
    let (status, _, bytes) = call(&app, Method::POST, "/v1/decisions", Some(&body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let response = decode(&bytes);
    assert_eq!(response["model"], "autotrust/JEV-27B-VL");
    let answers = response["answers"].as_array().unwrap();
    assert_eq!(answers.len(), 3);
    assert_eq!(answers[0]["type"], "predicate");
    assert!(answers[0]["probability"].as_f64().unwrap() > 0.99);
    assert_eq!(answers[1]["type"], "choice");
    assert_eq!(answers[1]["choice"], "large");
    assert!(answers[1]["confidence"].as_f64().unwrap() > 0.99);
    assert_eq!(answers[1]["probabilities"][2]["value"], "large");
    assert_eq!(answers[2]["type"], "score");
    assert!((answers[2]["score"].as_f64().unwrap() - 3.0).abs() < 0.01);
    assert_eq!(answers[2]["probabilities"][3]["label"], "great");
    assert_eq!(response["usage"]["input_tokens"], 30);
    assert_eq!(response["usage"]["output_tokens"], 3);
    assert_eq!(response["usage"]["total_tokens"], 33);

    {
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        for request in requests.iter() {
            let prompt = request["prompt"].as_str().unwrap();
            assert!(prompt.starts_with("[kind] "));
            assert!(prompt.contains("[state] context one\n\ncontext two\n[question]"));
        }
        assert!(
            requests[1]["prompt"]
                .as_str()
                .unwrap()
                .contains("A) small: little\nB) medium\nC) large: big")
        );
        assert!(
            requests[2]["prompt"]
                .as_str()
                .unwrap()
                .contains("bad\nok\ngood\ngreat")
        );
    }

    let body = json!({
        "input": "x",
        "questions": [{"type": "unknown", "name": "n"}]
    });
    let (status, bytes) = call_raw(
        &app,
        "/v1/decisions",
        &serde_json::to_string(&body).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&bytes).contains("unsupported question type 'unknown'"));
}

#[tokio::test]
async fn image_state_uses_chat_completions_with_raw_template() {
    let (app, requests) = app_for().await;
    let body = json!({
        "kind": "noul",
        "state": [
            "look at this",
            {"image": "data:image/png;base64,AAAA"}
        ],
        "question": "Is it a cat?"
    });
    let (status, _, bytes) = call(&app, Method::POST, "/v1/decide", Some(&body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert_eq!(decode(&bytes)["choice"], "true");

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let upstream = &requests[0];
    assert!(upstream.get("prompt").is_none());
    assert_eq!(upstream["chat_template"], RAW_TEMPLATE);
    assert_eq!(upstream["add_generation_prompt"], false);
    assert_eq!(upstream["logprobs"], true);
    assert_eq!(upstream["top_logprobs"], 1);
    assert_eq!(upstream["add_special_tokens"], false);
    assert_eq!(upstream["logprob_token_ids"], json!([3721, 1802]));
    assert_eq!(
        upstream["messages"][0]["content"][0]["text"],
        "[kind] noul\n[state] "
    );
    assert_eq!(
        upstream["messages"][0]["content"][1]["text"],
        "look at this"
    );
    assert_eq!(upstream["messages"][0]["content"][2]["type"], "image_url");
    assert_eq!(
        upstream["messages"][0]["content"][2]["image_url"]["url"],
        "data:image/png;base64,AAAA"
    );
    assert_eq!(
        upstream["messages"][0]["content"][3]["text"],
        "\n[question] Is it a cat?\n[options]\nfalse\ntrue\n[decision]:"
    );
}

#[tokio::test]
async fn proxy_rewrites_model_and_streams() {
    let (app, requests) = app_for().await;
    let body = json!({"model": "client-model", "prompt": "hello", "max_tokens": 4});
    let (status, headers, bytes) = call(&app, Method::POST, "/v1/completions", Some(&body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        header(&headers, "content-type").as_deref(),
        Some("application/json")
    );
    assert_eq!(decode(&bytes)["model"], "jev-awq");
    assert_eq!(requests.lock().unwrap()[0]["model"], "jev-awq");

    let body = json!({"model": "client-model", "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (status, headers, bytes) =
        call(&app, Method::POST, "/v1/chat/completions", Some(&body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        header(&headers, "content-type").as_deref(),
        Some("text/event-stream")
    );
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("data: {\"choices\""));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn validation_and_upstream_error_semantics() {
    let (app, _requests) = app_for().await;
    let (status, bytes) = call_raw(&app, "/v1/decide", "{not json").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&bytes).contains("invalid JSON body"));

    let body = json!({"kind": "choice", "state": "s", "question": "q", "options": ["only"]});
    let (status, bytes) =
        call_raw(&app, "/v1/decide", &serde_json::to_string(&body).unwrap()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&bytes).contains("choice needs 2-32 options"));

    let body = json!({"kind": "bogus", "state": "s", "question": "q"});
    let (status, bytes) =
        call_raw(&app, "/v1/decide", &serde_json::to_string(&body).unwrap()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&bytes).contains("kind must be noul|score|choice"));

    let body = json!({
        "kind": "noul",
        "state": [
            {"image": "data:image/png;base64,AAAA"},
            {"image": "data:image/png;base64,BBBB"}
        ],
        "question": "q"
    });
    let (status, bytes) =
        call_raw(&app, "/v1/decide", &serde_json::to_string(&body).unwrap()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&bytes).contains("one image per request"));

    let body = json!({"kind": "noul", "state": "s", "question": "FAIL"});
    let (status, bytes) =
        call_raw(&app, "/v1/decide", &serde_json::to_string(&body).unwrap()).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    let error = decode(&bytes);
    assert_eq!(error["error"]["type"], "upstream_error");
    assert!(error["detail"].as_str().unwrap().contains("500"));

    let body = json!({"kind": "noul", "state": "s", "question": "BADJSON"});
    let (status, _) = call_raw(&app, "/v1/decide", &serde_json::to_string(&body).unwrap()).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);

    let mut config = test_config(String::new());
    config.request_timeout = Duration::from_millis(50);
    let (app, _) = app_with(config).await;
    let body = json!({"kind": "noul", "state": "s", "question": "SLOW"});
    let (status, bytes) =
        call_raw(&app, "/v1/decide", &serde_json::to_string(&body).unwrap()).await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(decode(&bytes)["error"]["type"], "upstream_timeout");
}

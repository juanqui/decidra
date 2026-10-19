//! End-to-end API tests against a self-contained mock backend.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::{HeaderMap, Method, Request, StatusCode, header::CONTENT_TYPE},
};
use hmac::{Hmac, Mac};
use jev_engine::{
    api::AppState,
    backend::{Backend, MockBackend},
    config::Config,
    error::EngineError,
};
use serde_json::{Value, json};
use sha2::Sha256;
use tower::ServiceExt;

type Prompts = Arc<Mutex<Vec<(String, Vec<u32>)>>>;

fn mock_backend(prompts: Prompts) -> Arc<dyn Backend> {
    let backend = MockBackend::new(move |prompt: &str, ids: &[u32]| {
        prompts
            .lock()
            .unwrap()
            .push((prompt.to_owned(), ids.to_vec()));
        if prompt.contains("FAIL") {
            return Err(EngineError::Backend("mock backend exploded".to_owned()));
        }
        if prompt.contains("BADJSON") {
            return Err(EngineError::BadResponse("not valid logits".to_owned()));
        }
        if prompt.contains("SLOW") {
            return Err(EngineError::Timeout(Duration::from_secs(1)));
        }
        let preferred = if prompt.starts_with("[kind] noul") {
            1802
        } else if prompt.starts_with("[kind] score") {
            18
        } else {
            34
        };
        Ok(ids
            .iter()
            .map(|id| if *id == preferred { 0.0 } else { -8.0 })
            .collect())
    })
    .with_prompt_tokens(10);
    Arc::new(backend)
}

fn test_config() -> Config {
    Config {
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        model_id: "jev-awq".to_owned(),
        checkpoint: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        tokenizer: None,
        calibration: None,
        calibration_fitted: None,
        hf_snapshot: PathBuf::from("/nonexistent/hf-snapshot"),
        native_model: PathBuf::from("/nonexistent/native-model"),
        native_cache: true,
        native_marlin: true,
        native_fused_w4a16: true,
        max_concurrency: 4,
        signing_key: None,
        metrics_enabled: false,
        max_body_bytes: 8 * 1024 * 1024,
        served_identity: "test/JEV-27B-VL".to_owned(),
    }
}

async fn app_for() -> (Router, Prompts) {
    let prompts: Prompts = Arc::new(Mutex::new(Vec::new()));
    let state = AppState::load(test_config(), mock_backend(prompts.clone())).unwrap();
    (state.router(), prompts)
}

async fn app_with(config: Config) -> (Router, Prompts) {
    let prompts: Prompts = Arc::new(Mutex::new(Vec::new()));
    let state = AppState::load(config, mock_backend(prompts.clone())).unwrap();
    (state.router(), prompts)
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
    let mut config = test_config();
    config.signing_key = Some(b"secret-key".to_vec());
    let (app, prompts) = app_with(config).await;

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
        let prompts = prompts.lock().unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].1, vec![3721, 1802]);
        assert_eq!(
            prompts[0].0,
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
    assert!(String::from_utf8_lossy(&bytes).contains("native vision not implemented"));
    assert_eq!(prompts.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn choice_single_tournament_and_permute() {
    let (app, prompts) = app_for().await;
    let options: Vec<String> = (0..3).map(|index| format!("option-{index}")).collect();
    let body = json!({"kind": "choice", "state": "s", "question": "q", "options": options});
    let (status, _, bytes) = call(&app, Method::POST, "/v1/decide", Some(&body)).await;
    assert_eq!(status, StatusCode::OK);
    let response = decode(&bytes);
    assert_eq!(response["adaptation"], "native");
    assert_eq!(response["choice"], "option-2");
    assert_eq!(prompts.lock().unwrap().len(), 1);
    assert_eq!(
        prompts.lock().unwrap()[0].0,
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
    assert_eq!(prompts.lock().unwrap().len(), 1 + 3);
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
    let before = prompts.lock().unwrap().len();
    let (status, _, second_bytes) = call(&app, Method::POST, "/v1/decide", Some(&body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(prompts.lock().unwrap().len(), before + 4);
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
    let (app, prompts) = app_for().await;
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
        let prompts = prompts.lock().unwrap();
        assert_eq!(prompts.len(), 3);
        for (prompt, _) in prompts.iter() {
            assert!(prompt.starts_with("[kind] "));
            assert!(prompt.contains("[state] context one\n\ncontext two\n[question]"));
        }
        assert!(
            prompts[1]
                .0
                .contains("A) small: little\nB) medium\nC) large: big")
        );
        assert!(prompts[2].0.contains("bad\nok\ngood\ngreat"));
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
async fn image_states_are_rejected_without_a_backend_call() {
    let (app, prompts) = app_for().await;
    let body = json!({
        "kind": "noul",
        "state": [
            "look at this",
            {"image": "data:image/png;base64,AAAA"}
        ],
        "question": "Is it a cat?"
    });
    let (status, _, bytes) = call(&app, Method::POST, "/v1/decide", Some(&body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&bytes).contains("native vision not implemented"));

    let body = json!({
        "state": ["look at this", {"image": "data:image/png;base64,AAAA"}],
        "questions": {"notable": {"type": "noul", "instructions": "Is this notable?"}}
    });
    let (status, _, bytes) = call(&app, Method::POST, "/v1/systemone", Some(&body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&bytes).contains("native vision not implemented"));
    assert!(prompts.lock().unwrap().is_empty());
}

#[tokio::test]
async fn generation_endpoints_return_501() {
    let (app, prompts) = app_for().await;
    let body = json!({"model": "client-model", "prompt": "hello", "max_tokens": 4});
    let (status, _, bytes) = call(&app, Method::POST, "/v1/completions", Some(&body)).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    let error = decode(&bytes);
    assert_eq!(error["error"]["type"], "not_implemented_error");
    assert!(
        error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("native generation not implemented")
    );

    let body = json!({"messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (status, _, bytes) = call(&app, Method::POST, "/v1/chat/completions", Some(&body)).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert!(String::from_utf8_lossy(&bytes).contains("native generation not implemented"));
    assert!(prompts.lock().unwrap().is_empty());
}

#[tokio::test]
async fn systemone_noul_choice_and_validation() {
    let (app, prompts) = app_for().await;

    // noul with criteria, object state, and exact usage accounting.
    let body = json!({
        "state": {"ticket": "refund", "amount": 42},
        "questions": {
            "safe": {
                "type": "noul",
                "instructions": "Is this safe?",
                "criteria": {"true": "no abuse", "false": "abuse"}
            },
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {
                    "billing": "charges",
                    "support": "bugs",
                    "sales": null
                }
            }
        }
    });
    let (status, _, bytes) = call(&app, Method::POST, "/v1/systemone", Some(&body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let response = decode(&bytes);
    let id = response["id"].as_str().unwrap();
    assert!(id.starts_with("xpnp-hosted-"));
    assert_eq!(id.len(), "xpnp-hosted-".len() + 32);
    assert_eq!(response["model"], "test/JEV-27B-VL");
    assert_eq!(response["provider"], "xPNP Hosted");
    assert_eq!(response["answers"]["safe"]["type"], "noul");
    assert!(response["answers"]["safe"]["noul"].as_f64().unwrap() > 0.99);
    assert_eq!(response["answers"]["team"]["type"], "choice");
    assert_eq!(response["answers"]["team"]["choice"], "sales");
    assert!(response["answers"]["team"]["confidence"].as_f64().unwrap() > 0.99);
    let distribution = response["answers"]["team"]["probabilities"]
        .as_object()
        .unwrap();
    assert_eq!(
        distribution.keys().cloned().collect::<Vec<_>>(),
        ["billing", "support", "sales"]
    );
    assert!(close(
        distribution
            .values()
            .map(|value| value.as_f64().unwrap())
            .sum(),
        1.0
    ));
    assert_eq!(response["usage"]["input_tokens"], 20);
    assert_eq!(response["usage"]["output_tokens"], 0);
    assert_eq!(response["usage"]["cost"], 0);
    assert_eq!(response["usage"]["num_model_requests"], 2);

    {
        let prompts = prompts.lock().unwrap();
        assert_eq!(prompts.len(), 2);
        let noul = prompts[0].0.as_str();
        assert!(noul.contains("[state] {\"ticket\": \"refund\", \"amount\": 42}\nAnswer true when: no abuse\nAnswer false when: abuse\n\n[question] Is this safe?"));
        let choice = prompts[1].0.as_str();
        assert!(
            choice.contains(
                "[options]\nA) billing: charges\nB) support: bugs\nC) sales\n[decision]:"
            )
        );
    }

    // Validation failures use the xPNP error envelope and no backend calls.
    let mut nine = serde_json::Map::new();
    for index in 0..9 {
        nine.insert(
            format!("q{index}"),
            json!({"type": "noul", "instructions": "is it?"}),
        );
    }
    let before = prompts.lock().unwrap().len();
    let body = json!({"state": "x", "questions": nine});
    let (status, bytes) = call_raw(
        &app,
        "/v1/systemone",
        &serde_json::to_string(&body).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = decode(&bytes);
    assert_eq!(error["error"]["code"], 400);
    assert_eq!(error["error"]["message"], "at most 8 questions per request");

    let body = json!({
        "state": "x",
        "questions": {"q": {"type": "score", "instructions": "rate it"}}
    });
    let (status, bytes) = call_raw(
        &app,
        "/v1/systemone",
        &serde_json::to_string(&body).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        decode(&bytes)["error"]["message"],
        "score questions are not served by this backend"
    );

    let body = json!({
        "state": "x",
        "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"Bad_Key": "x", "ok": "y"}}}
    });
    let (status, bytes) = call_raw(
        &app,
        "/v1/systemone",
        &serde_json::to_string(&body).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        decode(&bytes)["error"]["message"],
        "choice keys must match [a-z0-9-]+, got 'Bad_Key'"
    );

    let body = json!({"state": "x", "questions": {}});
    let (status, bytes) = call_raw(
        &app,
        "/v1/systemone",
        &serde_json::to_string(&body).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        decode(&bytes)["error"]["message"],
        "questions must be a non-empty object"
    );

    let body = json!({"state": null, "questions": {"q": {"type": "noul", "instructions": "x"}}});
    let (status, bytes) = call_raw(
        &app,
        "/v1/systemone",
        &serde_json::to_string(&body).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        decode(&bytes)["error"]["message"],
        "state must be a string, an object, or an array"
    );
    assert_eq!(prompts.lock().unwrap().len(), before);
}

#[tokio::test]
async fn validation_and_backend_error_semantics() {
    let (app, _prompts) = app_for().await;
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
    assert!(String::from_utf8_lossy(&bytes).contains("native vision not implemented"));

    let body = json!({"kind": "noul", "state": "s", "question": "FAIL"});
    let (status, bytes) =
        call_raw(&app, "/v1/decide", &serde_json::to_string(&body).unwrap()).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    let error = decode(&bytes);
    assert_eq!(error["error"]["type"], "backend_error");
    assert!(
        error["detail"]
            .as_str()
            .unwrap()
            .contains("mock backend exploded")
    );

    let body = json!({"kind": "noul", "state": "s", "question": "BADJSON"});
    let (status, bytes) =
        call_raw(&app, "/v1/decide", &serde_json::to_string(&body).unwrap()).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(
        decode(&bytes)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid backend response")
    );

    let body = json!({"kind": "noul", "state": "s", "question": "SLOW"});
    let (status, bytes) =
        call_raw(&app, "/v1/decide", &serde_json::to_string(&body).unwrap()).await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(decode(&bytes)["error"]["type"], "backend_timeout");
}

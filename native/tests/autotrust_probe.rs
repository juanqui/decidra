//! AutoTrust-JEV-27B-v1.1 readout probe: the engine's lm_head code-token logits.
//!
//! The model's decision readout is the softmax over the first n answer-code token
//! ids at the last prompt position (no bias, temperature from vllm_readout.json).
//! This probe reads prompt token ids produced by the model's own tokenizer and the
//! code ids, runs the native forward, and writes the raw code logits.
//!
//! Run on GPU1 after quantization:
//! `DECIDRA_AUTOTRUST_MODEL=/srv/models/autotrust-jev-27b-v1.1-awq-g64 \
//!  DECIDRA_AUTOTRUST_PROBE=/tmp/autotrust_probe.json \
//!  DECIDRA_AUTOTRUST_OUT=/tmp/autotrust_engine.json \
//!  CUDA_VISIBLE_DEVICES=1 cargo test --release -p decidra-native \
//!    --test autotrust_probe -- --ignored --nocapture`

use std::path::Path;

use decidra_native::Engine;

#[test]
#[ignore = "requires the quantized AutoTrust checkpoint on GPU1"]
fn autotrust_readout() {
    let model = std::env::var("DECIDRA_AUTOTRUST_MODEL")
        .unwrap_or_else(|_| "/srv/models/autotrust-jev-27b-v1.1-awq-g64".to_string());
    let probe = std::env::var("DECIDRA_AUTOTRUST_PROBE")
        .unwrap_or_else(|_| "/tmp/autotrust_probe.json".to_string());
    let out = std::env::var("DECIDRA_AUTOTRUST_OUT")
        .unwrap_or_else(|_| "/tmp/autotrust_engine.json".to_string());

    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&probe).expect("probe json")).expect("json");
    let engine = Engine::new(Path::new(&model)).expect("engine");
    let mut results = Vec::new();
    for row in data["rows"].as_array().expect("rows") {
        let id = row["id"].as_str().unwrap_or_default().to_string();
        let tokens: Vec<u32> = row["tokens"]
            .as_array()
            .expect("tokens")
            .iter()
            .map(|v| v.as_u64().expect("token") as u32)
            .collect();
        let codes: Vec<u32> = row["codes"]
            .as_array()
            .expect("codes")
            .iter()
            .map(|v| v.as_u64().expect("code") as u32)
            .collect();
        let logits = engine
            .decision_logits(&tokens, &codes)
            .expect("decision logits");
        println!("{id}: {} codes, {} prompt tokens", codes.len(), tokens.len());
        results.push(serde_json::json!({"id": id, "logits": logits}));
    }
    std::fs::write(
        &out,
        serde_json::to_string(&serde_json::json!({"rows": results})).expect("json"),
    )
    .expect("write engine logits");
    println!("wrote {out}");
}

//! d3-turbo engine eval: decision_logits via the separate F32 readout head.
//! Input: /tmp/d3_probe.json (from /tmp/d3_eval.py). Output: /tmp/d3_engine_eval.json.

use std::path::Path;

use decidra_native::Engine;

#[test]
#[ignore = "requires the quantized d3-turbo checkpoint on GPU1"]
fn d3_eval() {
    let model = std::env::var("DECIDRA_D3_MODEL")
        .unwrap_or_else(|_| "/srv/models/d3-turbo-awq-g64".to_string());
    let probe = std::env::var("DECIDRA_D3_PROBE")
        .unwrap_or_else(|_| "/tmp/d3_probe.json".to_string());
    let out = std::env::var("DECIDRA_D3_OUT")
        .unwrap_or_else(|_| "/tmp/d3_engine_eval.json".to_string());
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&probe).expect("probe")).expect("json");
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
        let logits = engine.decision_logits(&tokens, &codes).expect("logits");
        results.push(serde_json::json!({"id": id, "logits": logits}));
    }
    std::fs::write(&out, serde_json::to_string(&serde_json::json!({"rows": results})).expect("json"))
        .expect("write");
    println!("wrote {out}");
}

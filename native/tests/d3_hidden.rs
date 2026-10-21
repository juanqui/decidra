//! d3-turbo final-norm hidden at the last position for the probe.
//! Input /tmp/d3_probe.json; output /tmp/d3_engine_hidden.json.

use std::path::Path;

use decidra_native::Engine;

#[test]
#[ignore = "requires the quantized d3-turbo checkpoint on GPU1"]
fn d3_hidden() {
    let model = std::env::var("DECIDRA_D3_MODEL")
        .unwrap_or_else(|_| "/srv/models/d3-turbo-w4a8-g64-v4".to_string());
    let probe =
        std::env::var("DECIDRA_D3_PROBE").unwrap_or_else(|_| "/tmp/d3_probe.json".to_string());
    let out = std::env::var("DECIDRA_D3_HIDDEN_OUT")
        .unwrap_or_else(|_| "/tmp/d3_engine_hidden.json".to_string());
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&probe).expect("probe")).expect("json");
    let engine = Engine::new(Path::new(&model)).expect("engine");
    let last = engine.cfg.num_hidden_layers.saturating_sub(1) as isize;
    let h = engine.cfg.hidden_size;
    let mut results = Vec::new();
    for row in data["rows"].as_array().expect("rows") {
        let id = row["id"].as_str().unwrap_or_default().to_string();
        let tokens: Vec<u32> = row["tokens"]
            .as_array()
            .expect("tokens")
            .iter()
            .map(|v| v.as_u64().expect("token") as u32)
            .collect();
        let split = row["split"].as_u64().expect("split") as usize;
        let all = engine
            .forward_split(&tokens, split, last, decidra_native::engine::Stage::FinalNorm, true)
            .expect("final norm hidden");
        assert_eq!(all.len(), h);
        let hidden = all;
        assert_eq!(hidden.len(), h);
        results.push(serde_json::json!({"id": id, "hidden": hidden}));
    }
    std::fs::write(
        &out,
        serde_json::to_string(&serde_json::json!({"rows": results})).expect("json"),
    )
    .expect("write");
    println!("wrote {out}");
}

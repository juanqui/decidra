//! Per-layer hidden-state probe: engine vs HF on the same quantized AutoTrust row.
//!
//! `DECIDRA_AUTOTRUST_MODEL` quantized dir, `DECIDRA_AUTOTRUST_HIDDEN_OUT` output,
//! `DECIDRA_AUTOTRUST_SAMPLE` JSONL (uses the first row's token_ids).

use std::path::Path;

use decidra_native::engine::Stage;
use decidra_native::Engine;

#[test]
#[ignore = "requires the quantized AutoTrust checkpoint on GPU1"]
fn autotrust_hidden_layers() {
    let model = std::env::var("DECIDRA_AUTOTRUST_MODEL")
        .unwrap_or_else(|_| "/srv/models/autotrust-jev-27b-v1.1-awq-g64".to_string());
    let sample = std::env::var("DECIDRA_AUTOTRUST_SAMPLE")
        .unwrap_or_else(|_| "/tmp/autotrust_sample256.jsonl".to_string());
    let out = std::env::var("DECIDRA_AUTOTRUST_HIDDEN_OUT")
        .unwrap_or_else(|_| "/tmp/autotrust_engine_hidden.json".to_string());
    let row: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(&sample)
            .expect("sample")
            .lines()
            .next()
            .expect("first row"),
    )
    .expect("json");
    let tokens: Vec<u32> = row["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|v| v.as_u64().expect("token") as u32)
        .collect();
    let engine = Engine::new(Path::new(&model)).expect("engine");
    let mut layers = serde_json::Map::new();
    for layer in [0usize, 1, 2, 3, 7, 15, 31, 63] {
        let hidden = engine
            .forward(&tokens, layer as isize, Stage::Layer, false)
            .expect("layer forward");
        println!("layer {layer}: {} values", hidden.len());
        layers.insert(layer.to_string(), serde_json::json!(hidden));
    }
    let embed = engine
        .forward(&tokens, -1, Stage::Embed, false)
        .expect("embed");
    let doc = serde_json::json!({"tokens": tokens, "embed": embed, "layers": layers});
    std::fs::write(&out, serde_json::to_string(&doc).expect("json")).expect("write");
    println!("wrote {out}");
}

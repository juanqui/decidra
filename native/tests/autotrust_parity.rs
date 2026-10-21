//! Teacher-forced per-layer parity: feed HF's layer input into the engine layer.
//!
//! `DECIDRA_AUTOTRUST_MODEL`, `DECIDRA_AUTOTRUST_HF_LAYERS` (from
//! `/tmp/autotrust_hf_layers.py`), `DECIDRA_AUTOTRUST_LAYERS_OUT`.

use std::path::Path;

use decidra_native::engine::Stage;
use decidra_native::Engine;

#[test]
#[ignore = "requires the quantized AutoTrust checkpoint and HF layer dump on GPU1"]
fn autotrust_teacher_forced_layers() {
    let model = std::env::var("DECIDRA_AUTOTRUST_MODEL")
        .unwrap_or_else(|_| "/srv/models/autotrust-jev-27b-v1.1-awq-g64".to_string());
    let hf_path = std::env::var("DECIDRA_AUTOTRUST_HF_LAYERS")
        .unwrap_or_else(|_| "/tmp/autotrust_hf_layers.json".to_string());
    let out_path = std::env::var("DECIDRA_AUTOTRUST_LAYERS_OUT")
        .unwrap_or_else(|_| "/tmp/autotrust_engine_layers.json".to_string());
    let hf: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&hf_path).expect("hf layers")).expect("json");
    let tokens: Vec<u32> = hf["tokens"]
        .as_array()
        .expect("tokens")
        .iter()
        .map(|v| v.as_u64().expect("token") as u32)
        .collect();
    let codes: Vec<u32> = hf["codes"]
        .as_array()
        .expect("codes")
        .iter()
        .map(|v| v.as_u64().expect("code") as u32)
        .collect();
    let seq = tokens.len();
    let mut engine = Engine::new(Path::new(&model)).expect("engine");
    // P0 can force causal (HF default); the default follows the checkpoint's
    // attention mode (AutoTrust is noncausal full).
    if std::env::var("DECIDRA_AUTOTRUST_CAUSAL").is_ok() {
        engine.noncausal_full = false;
    }
    let mut layers = serde_json::Map::new();
    for layer in [0usize, 1, 2, 3, 7, 15, 31, 47, 63] {
        let input: Vec<f32> = hf["layers"][layer.to_string()]["input"]
            .as_array()
            .expect("layer input")
            .iter()
            .map(|v| v.as_f64().expect("f32") as f32)
            .collect();
        let output = engine
            .forward_hidden(&input, seq, layer as isize, Stage::Layer, false)
            .expect("layer forward");
        println!("layer {layer}: {} values", output.len());
        layers.insert(layer.to_string(), serde_json::json!(output));
    }
    let last_input: Vec<f32> = hf["layers"]["63"]["input"]
        .as_array()
        .expect("last input")
        .iter()
        .map(|v| v.as_f64().expect("f32") as f32)
        .collect();
    let final_norm = engine
        .forward_hidden(&last_input, seq, 63, Stage::FinalNorm, false)
        .expect("final norm");
    let readout = engine
        .decision_logits(&tokens, &codes)
        .expect("readout logits");
    let doc = serde_json::json!({
        "layers": layers,
        "final_norm": final_norm,
        "readout": readout,
    });
    std::fs::write(&out_path, serde_json::to_string(&doc).expect("json")).expect("write");
    println!("wrote {out_path}");
}

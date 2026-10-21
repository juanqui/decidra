//! pplx marker readout: final-norm hidden at all positions, then the readout
//! at each `<|fim_suffix|>` marker. Input /tmp/pplx_probe.json.

use std::path::Path;

use decidra_native::Engine;
use decidra_native::engine::Stage;

#[test]
#[ignore = "requires the quantized pplx checkpoint on GPU1"]
fn pplx_marker_hidden() {
    let model = std::env::var("DECIDRA_PPLX_MODEL")
        .unwrap_or_else(|_| "/srv/models/pplx-decider-v1.2-awq-g64".to_string());
    let probe =
        std::env::var("DECIDRA_PPLX_PROBE").unwrap_or_else(|_| "/tmp/pplx_probe.json".to_string());
    let out = std::env::var("DECIDRA_PPLX_OUT")
        .unwrap_or_else(|_| "/tmp/pplx_engine_hidden.json".to_string());
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
        let markers: Vec<u32> = row["markers"]
            .as_array()
            .expect("markers")
            .iter()
            .map(|v| v.as_u64().expect("marker") as u32)
            .collect();
        let last = engine.cfg.num_hidden_layers.saturating_sub(1) as isize;
        let hidden = engine
            .forward(&tokens, last, Stage::FinalNorm, false)
            .expect("final norm hidden");
        let h = engine.cfg.hidden_size;
        let mut sites = Vec::new();
        for marker in &markers {
            let start = (*marker as usize) * h;
            sites.push(serde_json::json!(hidden[start..start + h].to_vec()));
        }
        results.push(serde_json::json!({"id": id, "sites": sites}));
    }
    std::fs::write(
        &out,
        serde_json::to_string(&serde_json::json!({"rows": results})).expect("json"),
    )
    .expect("write");
    println!("wrote {out}");
}

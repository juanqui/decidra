//! d3-turbo engine eval: decision_logits via the separate F32 readout head.
//! Input: /tmp/d3_probe.json (from /tmp/d3_eval.py). Output: /tmp/d3_engine_eval.json.

use std::path::Path;

use decidra_native::Engine;

#[test]
#[ignore = "requires the quantized d3-turbo checkpoint on GPU1"]
fn d3_eval() {
    let model = std::env::var("DECIDRA_D3_MODEL")
        .unwrap_or_else(|_| "/srv/models/d3-turbo-w4a8-g64-v4".to_string());
    let probe =
        std::env::var("DECIDRA_D3_PROBE").unwrap_or_else(|_| "/tmp/d3_probe.json".to_string());
    let out =
        std::env::var("DECIDRA_D3_OUT").unwrap_or_else(|_| "/tmp/d3_engine_eval.json".to_string());
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&probe).expect("probe")).expect("json");
    let engine = Engine::new(Path::new(&model)).expect("engine");
    if std::env::var("DECIDRA_NATIVE_CACHE")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        engine.set_cache_weights(false);
    }
    if std::env::var("DECIDRA_MARLIN")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        engine.set_marlin(true);
        engine.set_marlin_int8(
            std::env::var("DECIDRA_INT8")
                .map(|v| v != "0")
                .unwrap_or(true),
        );
        engine.set_int8_group_scale(
            std::env::var("DECIDRA_INT8_GROUP")
                .map(|v| v != "0")
                .unwrap_or(false),
        );
        let exclude = std::env::var("DECIDRA_INT8_EXCLUDE")
            .unwrap_or_else(|_| "self_attn.v_proj,self_attn.k_proj".to_owned());
        engine.set_int8_exclude(
            exclude
                .split(',')
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect(),
        );
        let group_exclude =
            std::env::var("DECIDRA_INT8_GROUP_EXCLUDE").unwrap_or_else(|_| String::new());
        engine.set_int8_group_exclude(
            group_exclude
                .split(',')
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect(),
        );
        engine.set_int8_all_layers("last:8");
        if let Ok(clip) = std::env::var("DECIDRA_INT8_CLIP") {
            if let Ok(clip) = clip.parse::<f32>() {
                engine.set_int8_clip(clip);
            }
        }
    }
    if std::env::var("DECIDRA_GDN_CHUNKED")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        engine.set_gdn_chunked(true);
        engine.set_gdn_chunked_tc(true);
    }
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
        let logits = match row.get("split").and_then(|v| v.as_u64()) {
            Some(split)
                if row
                    .get("state_once")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false) =>
            {
                let cache = engine
                    .capture_state(&tokens[..split as usize])
                    .expect("state cache");
                let question = &tokens[split as usize..];
                engine
                    .decision_logits_state_once_batch(&cache, &[question], &codes)
                    .expect("state-once logits")
                    .remove(0)
            }
            Some(split) => engine
                .decision_logits_split(&tokens, split as usize, &codes)
                .expect("split logits"),
            None => engine.decision_logits(&tokens, &codes).expect("logits"),
        };
        results.push(serde_json::json!({"id": id, "logits": logits}));
    }
    std::fs::write(
        &out,
        serde_json::to_string(&serde_json::json!({"rows": results})).expect("json"),
    )
    .expect("write");
    println!("wrote {out}");
}

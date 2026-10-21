//! State-once latency: capture once, then the batched question pass.
//! Input JSON: {"state": [ids], "questions": [[ids], ...], "codes": [ids]}.

use std::path::Path;
use std::time::Instant;

use decidra_native::Engine;

#[test]
#[ignore = "requires a d3-turbo checkpoint on a GPU"]
fn d3_state_once_perf() {
    let model = std::env::var("DECIDRA_D3_MODEL")
        .unwrap_or_else(|_| "/srv/models/d3-turbo-w4a8-g128".to_string());
    let probe =
        std::env::var("DECIDRA_D3_PROBE").unwrap_or_else(|_| "/tmp/d3_once_batch.json".to_string());
    let repeat: usize = std::env::var("DECIDRA_D3_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&probe).expect("probe")).expect("json");
    let state: Vec<u32> = data["state"]
        .as_array()
        .expect("state")
        .iter()
        .map(|v| v.as_u64().expect("token") as u32)
        .collect();
    let questions: Vec<Vec<u32>> = data["questions"]
        .as_array()
        .expect("questions")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("question")
                .iter()
                .map(|v| v.as_u64().expect("token") as u32)
                .collect()
        })
        .collect();
    let codes: Vec<u32> = data["codes"]
        .as_array()
        .expect("codes")
        .iter()
        .map(|v| v.as_u64().expect("code") as u32)
        .collect();
    let views: Vec<&[u32]> = questions.iter().map(|q| q.as_slice()).collect();

    let started = Instant::now();
    let engine = Engine::new(Path::new(&model)).expect("engine");
    if std::env::var("DECIDRA_MARLIN").map(|v| v != "0").unwrap_or(false) {
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
        let exclude = std::env::var("DECIDRA_INT8_EXCLUDE").unwrap_or_else(|_| {
            "self_attn.o_proj,self_attn.v_proj,self_attn.k_proj".to_owned()
        });
        engine.set_int8_exclude(
            exclude
                .split(',')
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect(),
        );
        engine.set_int8_group_exclude(vec!["mlp.gate_proj".to_owned()]);
        engine.set_int8_all_layers("last:8");
        if let Ok(clip) = std::env::var("DECIDRA_INT8_CLIP") {
            if let Ok(clip) = clip.parse::<f32>() {
                engine.set_int8_clip(clip);
            }
        }
    }
    if std::env::var("DECIDRA_GDN_CHUNKED").map(|v| v != "0").unwrap_or(false) {
        engine.set_gdn_chunked(true);
        engine.set_gdn_chunked_tc(true);
        if let Ok(min) = std::env::var("DECIDRA_GDN_CHUNK_MIN") {
            if let Ok(min) = min.parse::<usize>() {
                engine.set_gdn_chunked_min_seq(min);
            }
        }
    }
    println!("engine open: {:?}", started.elapsed());
    if std::env::var("DECIDRA_PRELOAD").map(|v| v != "0").unwrap_or(false) {
        engine.set_gpu_embed(
            std::env::var("DECIDRA_GPU_EMBED")
                .map(|v| v != "0")
                .unwrap_or(false),
        );
        let started = Instant::now();
        engine.preload_all().expect("preload");
        println!("preload: {:?}", started.elapsed());
    }
    let started = Instant::now();
    let cache = engine.capture_state(&state).expect("state cache");
    println!(
        "state capture: {:?} ({} tokens)",
        started.elapsed(),
        state.len()
    );
    let tokens: usize = questions.iter().map(|q| q.len()).sum();
    // Warm every kernel/shape first.
    let _ = engine
        .decision_logits_state_once_batch(&cache, &views, &codes)
        .expect("warmup");
    let mut times = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let started = Instant::now();
        let _ = engine
            .decision_logits_state_once_batch(&cache, &views, &codes)
            .expect("batch");
        times.push(started.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "batched questions: {} rows, {} tokens, p50 {:.2} ms, min {:.2} ms, max {:.2} ms",
        questions.len(),
        tokens,
        times[times.len() / 2],
        times[0],
        times[times.len() - 1]
    );
    let started = Instant::now();
    for q in &questions {
        let _ = engine
            .decision_logits_state_once(&cache, q, &codes)
            .expect("single");
    }
    println!(
        "single question passes: {} rows, total {:?}",
        questions.len(),
        started.elapsed()
    );
}

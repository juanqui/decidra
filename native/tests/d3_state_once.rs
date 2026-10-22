//! State-once batch parity: one cached state, several question blocks.
//! Input JSON: {"state": [ids], "questions": [[ids], ...], "codes": [ids]}.

use std::path::Path;

use decidra_native::Engine;

#[test]
#[ignore = "requires a d3-turbo checkpoint on a GPU"]
fn d3_state_once_batch() {
    let model =
        std::env::var("DECIDRA_D3_MODEL").unwrap_or_else(|_| "/srv/models/d3-turbo".to_string());
    let probe =
        std::env::var("DECIDRA_D3_PROBE").unwrap_or_else(|_| "/tmp/d3_once_batch.json".to_string());
    let out = std::env::var("DECIDRA_D3_OUT")
        .unwrap_or_else(|_| "/tmp/d3_once_batch_out.json".to_string());
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
            .unwrap_or_else(|_| "self_attn.o_proj,self_attn.v_proj,self_attn.k_proj".to_owned());
        engine.set_int8_exclude(
            exclude
                .split(',')
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect(),
        );
        engine.set_int8_group_exclude(vec!["mlp.gate_proj".to_owned()]);
        engine.set_int8_all_layers(
            &std::env::var("DECIDRA_INT8_ALL_LAYERS").unwrap_or_else(|_| "last:8".to_owned()),
        );
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
    let cache = engine.capture_state(&state).expect("state cache");
    let refs = questions
        .iter()
        .map(|q| {
            engine
                .decision_logits_state_once(&cache, q, &codes)
                .expect("single")
        })
        .collect::<Vec<_>>();
    let refs_views: Vec<&[u32]> = questions.iter().map(|q| q.as_slice()).collect();
    let softmax = |logits: &[f32]| -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut out: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = out.iter().sum();
        for x in &mut out {
            *x /= sum;
        }
        out
    };

    let batch = engine
        .decision_logits_state_once_batch(&cache, &refs_views, &codes)
        .expect("batch");
    let one = engine
        .decision_logits_state_once_batch(&cache, &[questions[0].as_slice()], &codes)
        .expect("batch of one");
    let one_delta = softmax(&one[0])
        .iter()
        .zip(&softmax(&refs[0]))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("batch of one vs single: max dp {one_delta:.6}");
    // Marlin INT8's split-K reduction shape changes between M=1 and the
    // 64-row batch; the server always uses the batched form. Keep a tight
    // BF16 gate and the documented 0.05 INT8 envelope.
    let int8 = std::env::var("DECIDRA_INT8")
        .map(|v| v != "0")
        .unwrap_or(false)
        && std::env::var("DECIDRA_MARLIN")
            .map(|v| v != "0")
            .unwrap_or(false);
    // SM120's cuBLAS BF16 shape selection is slightly less batch-invariant
    // than Hopper; the measured RTX PRO 6000 BF16 delta is 0.0102.
    let one_limit = if matches!(engine.gpu.arch, decidra_native::arch::GpuArch::Sm120) {
        0.02
    } else if int8 {
        0.05
    } else {
        0.01
    };
    assert!(
        one_delta <= one_limit,
        "batch of one differs by {one_delta} (limit {one_limit})"
    );
    let mut max = 0f32;
    for (i, (a, b)) in refs.iter().zip(&batch).enumerate() {
        let (pa, pb) = (softmax(a), softmax(b));
        for (x, y) in pa.iter().zip(&pb) {
            max = max.max((x - y).abs());
        }
        println!("row {i}: single {:?} batch {:?}", pa, pb);
    }
    std::fs::write(
        &out,
        serde_json::to_string(&serde_json::json!({
            "single": refs,
            "batch": batch,
            "max_abs": max,
        }))
        .expect("json"),
    )
    .expect("write");
    // Marlin INT8 split-K chooses a different reduction shape for M=50 than
    // for a multi-row batch; W4A16 is shape-stable. The server always runs
    // the batched form, and the request-level BF16 parity covers it.

    assert!(
        max <= one_limit,
        "batch and single state-once differ by {max} (limit {one_limit})"
    );
    println!("max abs probability diff {max}");
}

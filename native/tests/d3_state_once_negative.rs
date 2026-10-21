//! State-once negative controls: the seed, the convolution tail and the block
//! mask must each change the answer. Also checks the state-once fork against
//! the engine's per-row block reference.

use std::path::Path;

use decidra_native::Engine;

fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = p.iter().sum();
    for v in &mut p {
        *v /= s;
    }
    p
}

fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    softmax(a)
        .iter()
        .zip(softmax(b))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "requires a d3-turbo checkpoint on a GPU"]
fn d3_state_once_negative_controls() {
    let model = std::env::var("DECIDRA_D3_MODEL")
        .unwrap_or_else(|_| "/srv/models/d3-turbo".to_string());
    let probe =
        std::env::var("DECIDRA_D3_PROBE").unwrap_or_else(|_| "/tmp/d3_once_batch.json".to_string());
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&probe).expect("probe")).expect("json");
    let state: Vec<u32> = data["state"]
        .as_array()
        .expect("state")
        .iter()
        .map(|v| v.as_u64().expect("token") as u32)
        .collect();
    let question: Vec<u32> = data["questions"][0]
        .as_array()
        .expect("question")
        .iter()
        .map(|v| v.as_u64().expect("token") as u32)
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
    if std::env::var("DECIDRA_GDN_CHUNKED")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        engine.set_gdn_chunked(true);
        engine.set_gdn_chunked_tc(true);
        if let Ok(min) = std::env::var("DECIDRA_GDN_CHUNK_MIN") {
            if let Ok(min) = min.parse::<usize>() {
                engine.set_gdn_chunked_min_seq(min);
            }
        }
    }

    let mut cache = engine.capture_state(&state).expect("state cache");
    let base = engine
        .decision_logits_state_once(&cache, &question, &codes)
        .expect("state-once");

    // Per-row block reference over the same tokens.
    let mut full = state.clone();
    full.extend_from_slice(&question);
    let reference = engine
        .decision_logits_split(&full, state.len(), &codes)
        .expect("per-row block");
    let reference_delta = max_delta(&base, &reference);
    println!("state-once vs per-row block: max dp {reference_delta:.5}");
    // Chunk boundaries differ between a seeded question block and one full
    // [state || question] row; the reference runtime's own branching record
    // allows up to 0.032 for near-ties.
    assert!(
        reference_delta < 0.02,
        "state-once fork disagrees with the per-row block reference: {reference_delta}"
    );

    cache.zero_gdn_state(0).expect("zero seed");
    let no_seed = engine
        .decision_logits_state_once(&cache, &question, &codes)
        .expect("no seed");
    let seed_delta = max_delta(&base, &no_seed);
    println!("zero layer-0 GDN seed: max dp {seed_delta:.5}");
    assert!(
        seed_delta > 1e-3,
        "zeroing the GDN seed did not change the answer; the seed is not used"
    );

    let mut cache = engine.capture_state(&state).expect("state cache");
    cache.zero_gdn_hist(0).expect("zero tail");
    let no_tail = engine
        .decision_logits_state_once(&cache, &question, &codes)
        .expect("no tail");
    let tail_delta = max_delta(&base, &no_tail);
    println!("zero layer-0 convolution tail: max dp {tail_delta:.5}");
    assert!(
        tail_delta > 1e-3,
        "zeroing the convolution tail did not change the answer; the tail is not used"
    );
}

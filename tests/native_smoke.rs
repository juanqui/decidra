//! Real GPU smoke test. Run explicitly:
//!
//! ```sh
//! scripts/native-smoke.sh
//! ```
//!
//! Ignored by default: it needs the AWQ checkpoint, the CUDA driver/cuBLAS, and
//! a free GPU selected with `CUDA_VISIBLE_DEVICES`.

#![cfg(feature = "native")]

use decidra::{backend::Backend, config::Config, native_backend::NativeBackend};

#[test]
#[ignore = "requires the AWQ checkpoint and GPU1 (use scripts/native-smoke.sh)"]
fn native_forward_produces_finite_head_logits() {
    let config = Config::from_env().unwrap();
    let backend = NativeBackend::load(&config).unwrap();
    let prompt = decidra::prompt::text_prompt(
        "noul",
        "native smoke test",
        "Is the native engine ready?",
        &["false".to_owned(), "true".to_owned()],
    );
    let ids = [3721u32, 1802];
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let logits = runtime.block_on(backend.logits(&prompt, &ids)).unwrap();
    assert_eq!(logits.len(), ids.len(), "logits: {logits:?}");
    assert!(
        logits.iter().all(|value| value.is_finite()),
        "non-finite logits: {logits:?}"
    );
    println!("head logits for {ids:?}: {logits:?}");
}

/// The INT8 Marlin launcher splits rows into 64-row groups plus a tail. The
/// tail launch must offset the activation pointer by `rows * k` bytes (not
/// `rows` bytes) and the per-row scales by `rows * 4`; otherwise the last
/// group is computed from the wrong activations. Compare the per-row error of
/// the first groups with the tail on a module whose M is not a multiple of 64.
#[test]
#[ignore = "requires the AWQ checkpoint and GPU1 (use scripts/native-smoke.sh)"]
fn int8_tail_rows_match_head_rows() {
    let model = std::env::var("DECIDRA_NATIVE_MODEL")
        .unwrap_or_else(|_| "/srv/models/jev-awq/awq-w4a16-v2".to_owned());
    let engine = decidra_native::Engine::new(std::path::Path::new(&model)).unwrap();
    engine.set_marlin(true);
    engine.set_marlin_int8(true);
    engine.preload_all().unwrap();
    let rows = 130usize; // 2 x 64 + 2-row tail
    let k = 5120usize;
    let input: Vec<f32> = (0..rows * k)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5 + ((i as f32) * 0.071).cos() * 0.2)
        .collect();
    let module = "model.language_model.layers.0.mlp.gate_proj";
    let (got, reference) = engine.matmul_module_int8(module, &input, rows).unwrap();
    let n = got.len() / rows;
    let row_mae = |r: usize| {
        let a = &got[r * n..(r + 1) * n];
        let b = &reference[r * n..(r + 1) * n];
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / n as f32
    };
    let head = (0..64).map(row_mae).sum::<f32>() / 64.0;
    let tail = (128..rows).map(row_mae).sum::<f32>() / (rows - 128) as f32;
    println!("int8 module M={rows}: head MAE {head:.6}, tail MAE {tail:.6}");
    assert!(
        tail < 5.0 * head + 1e-5,
        "tail rows are corrupted: head {head} tail {tail}"
    );
}

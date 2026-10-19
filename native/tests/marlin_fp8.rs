//! Default-off W4A8-FP8 probe for one module.
//!
//! Run on GPU1:
//! `JEV_MODEL=/srv/models/jev-awq/awq-w4a16-v2 cargo test --release \
//!   -p jev-native --test marlin_fp8 -- --ignored --nocapture`
//!
//! This only checks that the a8 repack, FP8 activation quantization and the
//! FP8 Marlin instantiation launch and produce finite output comparable to
//! W4A16. It does not change defaults or exercise the server path.

use std::path::Path;

use jev_native::Engine;

#[test]
#[ignore = "requires the AWQ checkpoint and GPU1"]
fn marlin_fp8_one_module_vs_w4a16() {
    let model = std::env::var("JEV_MODEL")
        .unwrap_or_else(|_| "/srv/models/jev-awq/awq-w4a16-v2".to_string());
    let eng = Engine::new(Path::new(&model)).expect("engine");
    eng.set_marlin(true);
    eng.set_marlin_fp8(true);
    let module = "model.layers.0.mlp.down_proj"; // in=17408, out=5120
    let rows = 4usize;
    let k = 17408usize;
    let input: Vec<f32> = (0..rows * k)
        .map(|i| ((i as f32 * 0.013).sin() * 0.5) + ((i as f32 * 0.031).cos() * 0.1))
        .collect();
    let (got, reference) = eng
        .matmul_module_fp8(module, &input, rows)
        .expect("fp8 matmul");
    assert_eq!(got.len(), reference.len());
    assert!(
        got.iter().all(|v| v.is_finite()),
        "fp8 output has non-finite values"
    );
    let mut max = 0f32;
    let mut sum = 0f64;
    for (a, b) in got.iter().zip(reference.iter()) {
        let d = (a - b).abs();
        max = max.max(d);
        sum += d as f64;
    }
    let scale = reference.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!(
        "w4a8-fp8 vs w4a16: max_abs {max:.6} mean_abs {:.6} ref_scale {scale:.6}",
        sum / got.len() as f64
    );
}

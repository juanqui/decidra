//! Default-off W4A8-FP8 probe for one module.
//!
//! Run on GPU1:
//! `DECIDRA_MODEL=/srv/models/jev-awq/awq-w4a16-v2 cargo test --release \
//!   -p decidra-native --test marlin_fp8 -- --ignored --nocapture`
//!
//! Compares the native FP8 transform/repack/kernel against W4A16 and the
//! canonical AWQ/FP8 reference numbers for gate_proj, and times both kernels
//! with preloaded weights. No defaults or server paths are changed.

use std::path::Path;

use decidra_native::Engine;

fn input_for(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|i| ((i as f32 * 0.013).sin() * 0.5) + ((i as f32 * 0.031).cos() * 0.1))
        .collect()
}

#[test]
#[ignore = "requires the AWQ checkpoint and GPU1"]
fn marlin_fp8_gate_proj_vs_w4a16() {
    let model = std::env::var("DECIDRA_MODEL")
        .unwrap_or_else(|_| "/srv/models/jev-awq/awq-w4a16-v2".to_string());
    let eng = Engine::new(Path::new(&model)).expect("engine");
    eng.set_marlin(true);
    eng.set_marlin_fp8(true);
    let module = "model.language_model.layers.0.mlp.gate_proj"; // in=5120, out=17408
    for rows in [64usize, 1024] {
        let input = input_for(rows, 5120);
        let (got, reference) = eng
            .matmul_module_fp8(module, &input, rows)
            .expect("fp8 matmul");
        assert_eq!(got.len(), rows * 17408);
        assert!(got.iter().all(|v| v.is_finite()), "non-finite FP8 output");
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb = 0f64;
        let mut max = 0f32;
        let mut sum = 0f64;
        let mut sum_ref = 0f64;
        for (a, b) in got.iter().zip(reference.iter()) {
            let (a, b) = (*a as f64, *b as f64);
            dot += a * b;
            na += a * a;
            nb += b * b;
            max = max.max((a - b).abs() as f32);
            sum += (a - b).abs();
            sum_ref += b.abs();
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        println!(
            "M={rows}: cos {cos:.7} mean_rel {:.6} max_abs {max:.6} mean_abs {:.6}",
            sum / sum_ref,
            sum / got.len() as f64
        );
    }
    for rows in [64usize, 1024] {
        let input = input_for(rows, 5120);
        let (t16, t8) = eng
            .bench_matmul_fp8(module, &input, rows, 5)
            .expect("bench");
        let med = |v: &[f64]| {
            let mut x = v.to_vec();
            x.sort_by(|a, b| a.partial_cmp(b).unwrap());
            x[x.len() / 2]
        };
        let m16 = med(&t16);
        let m8 = med(&t8);
        println!(
            "M={rows}: w4a16 {m16:.3} ms, w4a8 {m8:.3} ms, speedup {:.3}x",
            m16 / m8
        );
    }
}

//! GPU A/B check for the experimental chunked GDN kernel.
//!
//! Run explicitly on GPU1:
//! `cargo test -p jev-native --test gdn_chunked -- --ignored --nocapture`
//!
//! Ignored by default: it needs a CUDA device. It uses synthetic data (no
//! checkpoint) and compares the f32 chunked path against `gdn_recurrent_split`.

use cudarc::driver::CudaSlice;
use half::bf16;
use jev_native::cuda::Gpu;
use jev_native::kernels::{self, Kernels};

fn host_data(n: usize, a: f32, b: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32 * 0.017;
            let y = i as f32 * 0.071;
            x.sin() * a + y.cos() * b
        })
        .collect()
}

#[test]
#[ignore = "requires a CUDA device (GPU1)"]
fn chunked_matches_sequential() {
    let seq = 512usize;
    let heads = 48usize;
    let dk = 128usize;
    let dv = 128usize;
    let gpu = Gpu::new(0).expect("cuda context");
    let kernels = Kernels::load(&gpu).expect("kernels");

    let q = host_data(seq * heads * dk, 0.12, 0.03);
    let k = host_data(seq * heads * dk, 0.10, -0.02);
    let v = host_data(seq * heads * dv, 0.5, 0.1);
    let g: Vec<f32> = (0..seq * heads)
        .map(|i| -0.005 - 0.02 * (i as f32 * 0.11).sin().abs())
        .collect();
    let beta: Vec<f32> = (0..seq * heads)
        .map(|i| 0.1 + 0.8 * (i as f32 * 0.07).sin().abs())
        .collect();

    let dq = gpu.stream.clone_htod(&q).unwrap();
    let dk_h: CudaSlice<f32> = gpu.stream.clone_htod(&k).unwrap();
    let dv_h = gpu.stream.clone_htod(&v).unwrap();
    let dg = gpu.stream.clone_htod(&g).unwrap();
    let db = gpu.stream.clone_htod(&beta).unwrap();

    let mut st_chunk = gpu.stream.alloc_zeros::<f32>(heads * dk * dv).unwrap();
    let mut out_chunk = gpu.stream.alloc_zeros::<bf16>(seq * heads * dv).unwrap();
    kernels::gdn_chunked(
        &gpu,
        &kernels.gdn_chunked,
        &dq,
        &dk_h,
        &dv_h,
        &dg,
        &db,
        &mut st_chunk,
        &mut out_chunk,
        seq,
        heads,
        dk,
        dv,
    )
    .expect("chunked launch");

    let mut st_seq = gpu.stream.alloc_zeros::<f32>(heads * dk * dv).unwrap();
    let mut out_seq = gpu.stream.alloc_zeros::<bf16>(seq * heads * dv).unwrap();
    kernels::gdn_recurrent_split(
        &gpu,
        &kernels.gdn_recurrent_split,
        &dq,
        &dk_h,
        &dv_h,
        &dg,
        &db,
        &mut st_seq,
        &mut out_seq,
        seq,
        heads,
        dk,
        dv,
    )
    .expect("sequential launch");
    gpu.synchronize().unwrap();

    let oc = gpu.stream.clone_dtoh(&out_chunk).unwrap();
    let os = gpu.stream.clone_dtoh(&out_seq).unwrap();
    let sc = gpu.stream.clone_dtoh(&st_chunk).unwrap();
    let ss = gpu.stream.clone_dtoh(&st_seq).unwrap();

    let mut out_max = 0f32;
    let mut out_sum = 0f64;
    for (a, b) in oc.iter().zip(os.iter()) {
        let d = (a.to_f32() - b.to_f32()).abs();
        out_max = out_max.max(d);
        out_sum += d as f64;
    }
    let out_mean = out_sum / oc.len() as f64;
    let mut st_max = 0f32;
    for (a, b) in sc.iter().zip(ss.iter()) {
        st_max = st_max.max((a - b).abs());
    }
    println!(
        "chunked vs sequential: out max {out_max:.3e} mean {out_mean:.3e}, state max {st_max:.3e}"
    );
    assert!(out_max < 2.0e-2, "output max {out_max}");
    assert!(out_mean < 2.0e-4, "output mean {out_mean}");
    assert!(st_max < 2.0e-3, "state max {st_max}");
}

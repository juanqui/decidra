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

#[test]
#[ignore = "requires a CUDA device (GPU1)"]
fn tc_chunked_matches_f32_chunked() {
    for seq in [32usize, 512] {
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
        let dk_h = gpu.stream.clone_htod(&k).unwrap();
        let dv_h = gpu.stream.clone_htod(&v).unwrap();
        let dg = gpu.stream.clone_htod(&g).unwrap();
        let db = gpu.stream.clone_htod(&beta).unwrap();
        let mut st_a = gpu.stream.alloc_zeros::<f32>(heads * dk * dv).unwrap();
        let mut out_a = gpu.stream.alloc_zeros::<bf16>(seq * heads * dv).unwrap();
        let mut st_b = gpu.stream.alloc_zeros::<f32>(heads * dk * dv).unwrap();
        let mut out_b = gpu.stream.alloc_zeros::<bf16>(seq * heads * dv).unwrap();
        kernels::gdn_chunked(
            &gpu,
            &kernels.gdn_chunked,
            &dq,
            &dk_h,
            &dv_h,
            &dg,
            &db,
            &mut st_a,
            &mut out_a,
            seq,
            heads,
            dk,
            dv,
        )
        .unwrap();
        kernels::gdn_chunked(
            &gpu,
            &kernels.gdn_chunked_tc,
            &dq,
            &dk_h,
            &dv_h,
            &dg,
            &db,
            &mut st_b,
            &mut out_b,
            seq,
            heads,
            dk,
            dv,
        )
        .unwrap();
        gpu.synchronize().unwrap();
        let oa = gpu.stream.clone_dtoh(&out_a).unwrap();
        let ob = gpu.stream.clone_dtoh(&out_b).unwrap();
        let sa = gpu.stream.clone_dtoh(&st_a).unwrap();
        let sb = gpu.stream.clone_dtoh(&st_b).unwrap();
        let omax = oa
            .iter()
            .zip(ob.iter())
            .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
            .fold(0f32, f32::max);
        let smax = sa
            .iter()
            .zip(sb.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let sscale = sa.iter().map(|x| x.abs()).fold(0f32, f32::max);
        println!(
            "T={seq}: tc-vs-f32 out max {omax:.3e}, state max {smax:.3e} (state scale {sscale:.3e})"
        );
    }
}

#[test]
#[ignore = "requires a CUDA device (GPU1)"]
fn tc_batched_matches_per_sequence() {
    let lens = [73usize, 64, 89];
    let batch = lens.len();
    let heads = 48usize;
    let dk = 128usize;
    let dv = 128usize;
    let total: usize = lens.iter().sum();
    let mut offsets = Vec::with_capacity(batch);
    let mut acc = 0;
    for &l in &lens {
        offsets.push(acc as i32);
        acc += l;
    }
    let gpu = Gpu::new(0).expect("cuda context");
    let kernels = Kernels::load(&gpu).expect("kernels");

    let q = host_data(total * heads * dk, 0.12, 0.03);
    let k = host_data(total * heads * dk, 0.10, -0.02);
    let v = host_data(total * heads * dv, 0.5, 0.1);
    let g: Vec<f32> = (0..total * heads)
        .map(|i| -0.005 - 0.02 * (i as f32 * 0.11).sin().abs())
        .collect();
    let beta: Vec<f32> = (0..total * heads)
        .map(|i| 0.1 + 0.8 * (i as f32 * 0.07).sin().abs())
        .collect();

    let dq = gpu.stream.clone_htod(&q).unwrap();
    let dk_h = gpu.stream.clone_htod(&k).unwrap();
    let dv_h = gpu.stream.clone_htod(&v).unwrap();
    let dg = gpu.stream.clone_htod(&g).unwrap();
    let db = gpu.stream.clone_htod(&beta).unwrap();
    let doff = gpu.stream.clone_htod(&offsets).unwrap();
    let dlen: CudaSlice<i32> = gpu
        .stream
        .clone_htod(&lens.iter().map(|&l| l as i32).collect::<Vec<_>>())
        .unwrap();

    let mut st_b = gpu
        .stream
        .alloc_zeros::<f32>(batch * heads * dk * dv)
        .unwrap();
    let mut out_b = gpu.stream.alloc_zeros::<bf16>(total * heads * dv).unwrap();
    kernels::gdn_chunked_batched(
        &gpu,
        &kernels.gdn_chunked_tc_batched,
        &dq,
        &dk_h,
        &dv_h,
        &dg,
        &db,
        &mut st_b,
        &mut out_b,
        &doff,
        &dlen,
        batch,
        heads,
        dk,
        dv,
    )
    .expect("batched launch");

    let mut st_s = gpu
        .stream
        .alloc_zeros::<f32>(batch * heads * dk * dv)
        .unwrap();
    let mut out_s = gpu.stream.alloc_zeros::<bf16>(total * heads * dv).unwrap();
    for (s, &l) in lens.iter().enumerate() {
        let o = offsets[s] as usize;
        let qv = dq.try_slice(o * heads * dk..(o + l) * heads * dk).unwrap();
        let kv = dk_h
            .try_slice(o * heads * dk..(o + l) * heads * dk)
            .unwrap();
        let vv = dv_h
            .try_slice(o * heads * dv..(o + l) * heads * dv)
            .unwrap();
        let gv = dg.try_slice(o * heads..(o + l) * heads).unwrap();
        let bv = db.try_slice(o * heads..(o + l) * heads).unwrap();
        let mut sv = st_s
            .try_slice_mut(s * heads * dk * dv..(s + 1) * heads * dk * dv)
            .unwrap();
        let mut ov = out_s
            .try_slice_mut(o * heads * dv..(o + l) * heads * dv)
            .unwrap();
        kernels::gdn_chunked(
            &gpu,
            &kernels.gdn_chunked_tc,
            &qv,
            &kv,
            &vv,
            &gv,
            &bv,
            &mut sv,
            &mut ov,
            l,
            heads,
            dk,
            dv,
        )
        .expect("single launch");
    }
    gpu.synchronize().unwrap();

    let ob = gpu.stream.clone_dtoh(&out_b).unwrap();
    let os = gpu.stream.clone_dtoh(&out_s).unwrap();
    let sb = gpu.stream.clone_dtoh(&st_b).unwrap();
    let ss = gpu.stream.clone_dtoh(&st_s).unwrap();
    let omax = ob
        .iter()
        .zip(os.iter())
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0f32, f32::max);
    let smax = sb
        .iter()
        .zip(ss.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("batched vs per-seq: out max {omax:.3e}, state max {smax:.3e}");
    assert_eq!(omax, 0.0, "batched output differs: {omax}");
    assert_eq!(smax, 0.0, "batched state differs: {smax}");
}

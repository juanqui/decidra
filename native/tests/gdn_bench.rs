//! Kernel-level benchmark + correctness for the chunked GDN kernels.
//!
//! Run on GPU1:
//! `CUDA_VISIBLE_DEVICES=1 cargo test -p decidra-native --release --test gdn_bench -- --ignored --nocapture`
//!
//! Synthetic data, no checkpoint. Reports kernel-only wall time per sequence
//! length and compares each chunked variant against `gdn_recurrent_split`.

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use decidra_native::cuda::Gpu;
use decidra_native::kernels::{self, Kernels};
use half::bf16;

const PHASES: [&str; 10] = [
    "load", "KK^T", "A mask", "solve", "Vnew", "Q load", "QS", "QK+mask", "output", "state",
];

/// Launch `gdn_chunked_tc_prof` and return the per-phase microsecond breakdown,
/// normalised so the phase total matches `wall_ms` (mean over the grid).
#[allow(clippy::too_many_arguments)]
fn prof_breakdown(
    gpu: &Gpu,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    g: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    prof: &mut CudaSlice<u64>,
    seq: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    vc: usize,
    wall_ms: f64,
) -> Vec<f64> {
    let vc = vc as i32;
    let smem = (dk * vc as usize + 2 * 32 * dk + 2 * 32 * vc as usize + 32 * 32 + 4 * 32)
        * std::mem::size_of::<f32>();
    let f = gpu.func("gdn_chunked_tc_prof").expect("prof kernel");
    f.set_attribute(
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        90112,
    )
    .unwrap();
    let cfg = LaunchConfig {
        grid_dim: ((dv / vc as usize) as u32, heads as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let (seq_i, heads_i, dk_i, dv_i) = (seq as i32, heads as i32, dk as i32, dv as i32);
    {
        let (qp, _r0) = q.device_ptr(&gpu.stream);
        let (kp, _r1) = k.device_ptr(&gpu.stream);
        let (vp, _r2) = v.device_ptr(&gpu.stream);
        let (gp, _r3) = g.device_ptr(&gpu.stream);
        let (bp, _r4) = beta.device_ptr(&gpu.stream);
        let (sp, _r5) = state.device_ptr_mut(&gpu.stream);
        let (op, _r6) = out.device_ptr_mut(&gpu.stream);
        let (pp, _r7) = prof.device_ptr_mut(&gpu.stream);
        let mut l = gpu.stream.launch_builder(&f);
        l.arg(&qp);
        l.arg(&kp);
        l.arg(&vp);
        l.arg(&gp);
        l.arg(&bp);
        l.arg(&sp);
        l.arg(&op);
        l.arg(&seq_i);
        l.arg(&heads_i);
        l.arg(&dk_i);
        l.arg(&dv_i);
        l.arg(&vc);
        l.arg(&pp);
        unsafe { l.launch(cfg) }.unwrap();
    }
    gpu.synchronize().unwrap();

    let raw = gpu.stream.clone_dtoh(prof).unwrap();
    let nblocks = (dv / vc as usize) * heads;
    let mut mean = vec![0f64; PHASES.len()];
    for b in 0..nblocks {
        for (p, m) in mean.iter_mut().enumerate() {
            *m += raw[b * PHASES.len() + p] as f64 / nblocks as f64;
        }
    }
    let total: f64 = mean.iter().sum();
    let scale = wall_ms * 1e3 / total; // us per cycle
    mean.iter().map(|c| c * scale).collect()
}

fn host_data(n: usize, a: f32, b: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32 * 0.017;
            let y = i as f32 * 0.071;
            x.sin() * a + y.cos() * b
        })
        .collect()
}

fn time_launch(gpu: &Gpu, iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    gpu.synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    gpu.synchronize().unwrap();
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

#[test]
#[ignore = "requires a CUDA device (GPU1)"]
fn bench_and_check() {
    let heads = 48usize;
    let dk = 128usize;
    let dv = 128usize;
    let max_seq = 16384usize;
    let gpu = Gpu::new(0).expect("cuda context");
    let kernels = Kernels::load(&gpu).expect("kernels");

    let q = host_data(max_seq * heads * dk, 0.12, 0.03);
    let k = host_data(max_seq * heads * dk, 0.10, -0.02);
    let v = host_data(max_seq * heads * dv, 0.5, 0.1);
    // Match the reference test's decay range plus a stronger-decay variant.
    let g: Vec<f32> = (0..max_seq * heads)
        .map(|i| -0.005 - 0.02 * (i as f32 * 0.11).sin().abs())
        .collect();
    let beta: Vec<f32> = (0..max_seq * heads)
        .map(|i| 0.1 + 0.8 * (i as f32 * 0.07).sin().abs())
        .collect();

    let dq = gpu.stream.clone_htod(&q).unwrap();
    let dk_h = gpu.stream.clone_htod(&k).unwrap();
    let dv_h = gpu.stream.clone_htod(&v).unwrap();
    let dg = gpu.stream.clone_htod(&g).unwrap();
    let db = gpu.stream.clone_htod(&beta).unwrap();

    for seq in [1024usize, 4096, 16384] {
        println!("--- seq={seq} ---");
        let mut st_tc = gpu.stream.alloc_zeros::<f32>(heads * dk * dv).unwrap();
        let mut out_tc = gpu.stream.alloc_zeros::<bf16>(seq * heads * dv).unwrap();
        let mut st_seq = gpu.stream.alloc_zeros::<f32>(heads * dk * dv).unwrap();
        let mut out_seq = gpu.stream.alloc_zeros::<bf16>(seq * heads * dv).unwrap();

        let mut st_ref = gpu.stream.alloc_zeros::<f32>(heads * dk * dv).unwrap();
        let mut out_ref = gpu.stream.alloc_zeros::<bf16>(seq * heads * dv).unwrap();
        kernels::gdn_recurrent_split(
            &gpu,
            &kernels.gdn_recurrent_split,
            &dq,
            &dk_h,
            &dv_h,
            &dg,
            &db,
            &mut st_ref,
            &mut out_ref,
            seq,
            heads,
            dk,
            dv,
        )
        .expect("sequential launch");
        gpu.synchronize().unwrap();
        let oref = gpu.stream.clone_dtoh(&out_ref).unwrap();
        let sref = gpu.stream.clone_dtoh(&st_ref).unwrap();

        let ms = time_launch(&gpu, 3, || {
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
            .unwrap();
        });
        println!("  seq      {ms:9.3} ms");

        for (name, fun) in [
            ("chunked_f32", &kernels.gdn_chunked),
            ("chunked_tc", &kernels.gdn_chunked_tc),
        ] {
            // correctness
            kernels::gdn_chunked(
                &gpu,
                fun,
                &dq,
                &dk_h,
                &dv_h,
                &dg,
                &db,
                &mut st_tc,
                &mut out_tc,
                seq,
                heads,
                dk,
                dv,
            )
            .unwrap();
            gpu.synchronize().unwrap();
            let oc = gpu.stream.clone_dtoh(&out_tc).unwrap();
            let sc = gpu.stream.clone_dtoh(&st_tc).unwrap();
            let mut omax = 0f32;
            let mut osum = 0f64;
            for (a, b) in oc.iter().zip(oref.iter()) {
                let d = (a.to_f32() - b.to_f32()).abs();
                omax = omax.max(d);
                osum += d as f64;
            }
            let smax = sc
                .iter()
                .zip(sref.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let ms = time_launch(&gpu, 3, || {
                kernels::gdn_chunked(
                    &gpu,
                    fun,
                    &dq,
                    &dk_h,
                    &dv_h,
                    &dg,
                    &db,
                    &mut st_tc,
                    &mut out_tc,
                    seq,
                    heads,
                    dk,
                    dv,
                )
                .unwrap();
            });
            println!(
                "  {name:12} {ms:9.3} ms   out max {omax:.3e} mean {:.3e}  state max {smax:.3e}",
                osum / oc.len() as f64
            );
        }

        // Phase breakdown for chunked_tc (prof variant runs identical code).
        let mut prof = gpu
            .stream
            .alloc_zeros::<u64>((dv / 64) * heads * 10)
            .unwrap();
        let f = gpu.func("gdn_chunked_tc_prof").unwrap();
        f.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            90112,
        )
        .unwrap();
        let ms = time_launch(&gpu, 3, || {
            let (qp, _a) = dq.device_ptr(&gpu.stream);
            let (kp, _b) = dk_h.device_ptr(&gpu.stream);
            let (vp, _c) = dv_h.device_ptr(&gpu.stream);
            let (gp, _d) = dg.device_ptr(&gpu.stream);
            let (bp, _e) = db.device_ptr(&gpu.stream);
            let (sp, _f) = st_tc.device_ptr_mut(&gpu.stream);
            let (op, _g) = out_tc.device_ptr_mut(&gpu.stream);
            let (pp, _h) = prof.device_ptr_mut(&gpu.stream);
            let smem = (dk * 64 + 2 * 32 * dk + 2 * 32 * 64 + 32 * 32 + 4 * 32) * 4;
            let cfg = LaunchConfig {
                grid_dim: (2, heads as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            let (seq_i, heads_i, dk_i, dv_i, vc_i) =
                (seq as i32, heads as i32, dk as i32, dv as i32, 64i32);
            let mut l = gpu.stream.launch_builder(&f);
            l.arg(&qp);
            l.arg(&kp);
            l.arg(&vp);
            l.arg(&gp);
            l.arg(&bp);
            l.arg(&sp);
            l.arg(&op);
            l.arg(&seq_i);
            l.arg(&heads_i);
            l.arg(&dk_i);
            l.arg(&dv_i);
            l.arg(&vc_i);
            l.arg(&pp);
            unsafe { l.launch(cfg) }.unwrap();
        });
        let us = prof_breakdown(
            &gpu,
            &dq,
            &dk_h,
            &dv_h,
            &dg,
            &db,
            &mut st_tc,
            &mut out_tc,
            &mut prof,
            seq,
            heads,
            dk,
            dv,
            64,
            ms,
        );
        println!("  prof kernel {ms:.3} ms; phases (us, % of kernel):");
        for (name, v) in PHASES.iter().zip(us.iter()) {
            println!("    {name:9} {v:8.1} us  {:5.1}%", v / (ms * 1e3) * 100.0);
        }
        println!("    {:9} {:8.1} us", "sum", us.iter().sum::<f64>());
    }
}

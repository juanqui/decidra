//! Safe wrappers around the custom CUDA kernels.
use anyhow::Result;
use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use half::bf16;

use crate::cuda::Gpu;

pub struct Kernels {
    pub rmsnorm: CudaFunction,
    pub rmsnorm_gated: CudaFunction,
    pub add: CudaFunction,
    pub silu_mul: CudaFunction,
    pub sigmoid_gate: CudaFunction,
    pub split_q_gate: CudaFunction,
    pub rope: CudaFunction,
    pub attention: CudaFunction,
    pub attention_warp: CudaFunction,
    /// Tiled FlashAttention-2-style kernel, one entry per supported head_dim.
    pub attention_tc_64: CudaFunction,
    pub attention_tc_128: CudaFunction,
    pub attention_tc_256: CudaFunction,
    pub conv1d_silu: CudaFunction,
    pub gdn_prepare: CudaFunction,
    pub gdn_recurrent: CudaFunction,
    pub dequant_w4a16: CudaFunction,
    pub gemm_w4a16: CudaFunction,
    pub gemm_w4a16_m128: CudaFunction,
    /// Marlin kernels indexed by thread_m_blocks - 1 (1..4).
    pub marlin_cfgs: Vec<CudaFunction>,
    pub marlin_repack: CudaFunction,
    pub marlin_transpose: CudaFunction,
}

impl Kernels {
    pub fn load(gpu: &Gpu) -> Result<Self> {
        Ok(Self {
            rmsnorm: gpu.func("rmsnorm_bf16")?,
            rmsnorm_gated: gpu.func("rmsnorm_gated_bf16")?,
            add: gpu.func("add_bf16")?,
            silu_mul: gpu.func("silu_mul_bf16")?,
            sigmoid_gate: gpu.func("sigmoid_gate_bf16")?,
            split_q_gate: gpu.func("split_q_gate_bf16")?,
            rope: gpu.func("rope_bf16")?,
            attention: gpu.func("attention_bf16")?,
            attention_warp: gpu.func("attention_warp_bf16")?,
            attention_tc_64: gpu.func("attention_tc_bf16_d64")?,
            attention_tc_128: {
                let f = gpu.func("attention_tc_bf16_d128")?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    52224,
                )?;
                f
            },
            attention_tc_256: {
                let f = gpu.func("attention_tc_bf16_d256")?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    101376,
                )?;
                f
            },
            conv1d_silu: gpu.func("conv1d_silu_bf16")?,
            gdn_prepare: gpu.func("gdn_prepare")?,
            gdn_recurrent: gpu.func("gdn_recurrent")?,
            dequant_w4a16: gpu.func("dequant_w4a16_bf16")?,
            marlin_cfgs: {
                const NAMES: [&str; 5] = [
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi1ELi8ELi8ELb1ELi4ELi4ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi1ELi8ELi8ELb0ELi4ELi4ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi2ELi16ELi4ELb0ELi4ELi4ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi3ELi16ELi4ELb0ELi4ELi4ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi4ELi16ELi4ELb0ELi4ELi4ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                ];
                let mut v = Vec::with_capacity(NAMES.len());
                for name in NAMES {
                    let f = gpu.func_marlin(name)?;
                    f.set_attribute(
                        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        MR_SHARED,
                    )?;
                    v.push(f);
                }
                v
            },
            marlin_repack: {
                let f = gpu.func_marlin(
                    "_ZN6marlin29jev_gptq_marlin_repack_kernelILi256ELi4ELb0EEEvPKjPjii",
                )?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    4096,
                )?;
                f
            },
            marlin_transpose: gpu.func_marlin("_Z26transpose_packed_q4_kernelPKjPjii")?,
            gemm_w4a16: gpu.func("gemm_w4a16_bf16")?,
            gemm_w4a16_m128: {
                let f = gpu.func("gemm_w4a16_m128_bf16")?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    90112,
                )?;
                f
            },
        })
    }
}

fn cfg_1d(n: usize, block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u64).div_ceil(block as u64) as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn elem_block(cols: usize) -> u32 {
    cols.next_power_of_two().clamp(32, 256) as u32
}

#[allow(clippy::too_many_arguments)]
pub fn rmsnorm(
    gpu: &Gpu,
    k: &CudaFunction,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    rows: usize,
    cols: usize,
    eps: f32,
    zero_centered: bool,
) -> Result<()> {
    let block = elem_block(cols);
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 4,
    };
    let cols_i = cols as i32;
    let zc = zero_centered as i32;
    let mut l = gpu.stream.launch_builder(k);
    l.arg(x);
    l.arg(w);
    l.arg(out);
    l.arg(&cols_i);
    l.arg(&eps);
    l.arg(&zc);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_gated(
    gpu: &Gpu,
    k: &CudaFunction,
    x: &CudaSlice<bf16>,
    gate: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    let block = elem_block(cols);
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 4,
    };
    let cols_i = cols as i32;
    let mut l = gpu.stream.launch_builder(k);
    l.arg(x);
    l.arg(gate);
    l.arg(w);
    l.arg(out);
    l.arg(&cols_i);
    l.arg(&eps);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

pub fn add(
    gpu: &Gpu,
    k: &CudaFunction,
    a: &CudaSlice<bf16>,
    b: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    n: usize,
) -> Result<()> {
    let cfg = cfg_1d(n, 256);
    let n_i = n as i32;
    let mut l = gpu.stream.launch_builder(k);
    l.arg(a);
    l.arg(b);
    l.arg(out);
    l.arg(&n_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

pub fn silu_mul(
    gpu: &Gpu,
    k: &CudaFunction,
    gate: &CudaSlice<bf16>,
    up: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    n: usize,
) -> Result<()> {
    let cfg = cfg_1d(n, 256);
    let n_i = n as i32;
    let mut l = gpu.stream.launch_builder(k);
    l.arg(gate);
    l.arg(up);
    l.arg(out);
    l.arg(&n_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

pub fn sigmoid_gate(
    gpu: &Gpu,
    k: &CudaFunction,
    x: &CudaSlice<bf16>,
    gate: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    n: usize,
) -> Result<()> {
    let cfg = cfg_1d(n, 256);
    let n_i = n as i32;
    let mut l = gpu.stream.launch_builder(k);
    l.arg(x);
    l.arg(gate);
    l.arg(out);
    l.arg(&n_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn split_q_gate(
    gpu: &Gpu,
    k: &CudaFunction,
    qg: &CudaSlice<bf16>,
    q: &mut CudaSlice<bf16>,
    gate: &mut CudaSlice<bf16>,
    seq: usize,
    heads: usize,
    hd: usize,
) -> Result<()> {
    let n = seq * heads * hd;
    let cfg = cfg_1d(n, 256);
    let seq_i = seq as i32;
    let heads_i = heads as i32;
    let hd_i = hd as i32;
    let mut l = gpu.stream.launch_builder(k);
    l.arg(qg);
    l.arg(q);
    l.arg(gate);
    l.arg(&seq_i);
    l.arg(&heads_i);
    l.arg(&hd_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn rope(
    gpu: &Gpu,
    k: &CudaFunction,
    q: &mut CudaSlice<bf16>,
    kk: &mut CudaSlice<bf16>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    seq: usize,
    n_q_heads: usize,
    n_k_heads: usize,
    hd: usize,
    rhalf: usize,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (seq as u32, (n_q_heads + n_k_heads) as u32, 1),
        block_dim: (rhalf.max(1) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let nq = n_q_heads as i32;
    let nk = n_k_heads as i32;
    let hd_i = hd as i32;
    let rh = rhalf as i32;
    let mut l = gpu.stream.launch_builder(k);
    l.arg(q);
    l.arg(kk);
    l.arg(cos);
    l.arg(sin);
    l.arg(&nq);
    l.arg(&nk);
    l.arg(&hd_i);
    l.arg(&rh);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn attention(
    gpu: &Gpu,
    k: &CudaFunction,
    q: &CudaSlice<bf16>,
    kk: &CudaSlice<bf16>,
    v: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    seq: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    scale: f32,
) -> Result<()> {
    let block = 128u32;
    let cfg = LaunchConfig {
        grid_dim: (seq as u32, n_heads as u32, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: (2 * hd + block as usize + 4) as u32 * 4,
    };
    let seq_i = seq as i32;
    let nh = n_heads as i32;
    let nkv = n_kv as i32;
    let hd_i = hd as i32;
    let mut l = gpu.stream.launch_builder(k);
    l.arg(q);
    l.arg(kk);
    l.arg(v);
    l.arg(out);
    l.arg(&seq_i);
    l.arg(&nh);
    l.arg(&nkv);
    l.arg(&hd_i);
    l.arg(&scale);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dequant_w4a16(
    gpu: &Gpu,
    k: &CudaFunction,
    packed: &CudaSlice<i32>,
    scale: &CudaSlice<bf16>,
    zp: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> Result<()> {
    let total = out_dim * in_dim;
    let cfg = cfg_1d(total, 256);
    let out_i = out_dim as i32;
    let in_i = in_dim as i32;
    let gs = group_size as i32;
    let mut l = gpu.stream.launch_builder(k);
    l.arg(packed);
    l.arg(scale);
    l.arg(zp);
    l.arg(out);
    l.arg(&out_i);
    l.arg(&in_i);
    l.arg(&gs);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_w4a16(
    gpu: &Gpu,
    k: &CudaFunction,
    x: &CudaSlice<bf16>,
    packed: &CudaSlice<i32>,
    scale: &CudaSlice<bf16>,
    zp: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    m: usize,
    n: usize,
    kk: usize,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (m_i, n_i, k_i) = (m as i32, n as i32, kk as i32);
    let mut l = gpu.stream.launch_builder(k);
    l.arg(x);
    l.arg(packed);
    l.arg(scale);
    l.arg(zp);
    l.arg(out);
    l.arg(&m_i);
    l.arg(&n_i);
    l.arg(&k_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_w4a16_m128(
    gpu: &Gpu,
    k: &CudaFunction,
    x: &CudaSlice<bf16>,
    packed: &CudaSlice<i32>,
    scale: &CudaSlice<bf16>,
    zp: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    m: usize,
    n: usize,
    kk: usize,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1),
        block_dim: (512, 1, 1),
        shared_mem_bytes: 90112,
    };
    let (m_i, n_i, k_i) = (m as i32, n as i32, kk as i32);
    let mut l = gpu.stream.launch_builder(k);
    l.arg(x);
    l.arg(packed);
    l.arg(scale);
    l.arg(zp);
    l.arg(out);
    l.arg(&m_i);
    l.arg(&n_i);
    l.arg(&k_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

pub const MR_SHARED: i32 = 101376;

fn ptr_of(slice: &CudaSlice<bf16>, gpu: &Gpu) -> u64 {
    let (p, _rec) = slice.device_ptr(&gpu.stream);
    p
}

pub fn marlin_repack(
    gpu: &Gpu,
    f: &CudaFunction,
    gptq: &CudaSlice<i32>,
    out: &mut CudaSlice<i32>,
    k: usize,
    n: usize,
    sms: usize,
) -> Result<()> {
    let (src_ptr, _r0) = gptq.device_ptr(&gpu.stream);
    let (out_ptr, _r1) = out.device_ptr_mut(&gpu.stream);
    let (src_addr, out_addr) = (src_ptr, out_ptr);
    let (k_i, n_i) = (k as i32, n as i32);
    let _ = sms;
    let cfg = LaunchConfig {
        grid_dim: (128, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 4096,
    };
    let mut l = gpu.stream.launch_builder(f);
    l.arg(&src_addr);
    l.arg(&out_addr);
    l.arg(&k_i);
    l.arg(&n_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

pub fn marlin_transpose(
    gpu: &Gpu,
    f: &CudaFunction,
    src: &CudaSlice<i32>,
    dst: &mut CudaSlice<i32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let (s, _r0) = src.device_ptr(&gpu.stream);
    let (d, _r1) = dst.device_ptr_mut(&gpu.stream);
    let (s_addr, d_addr) = (s, d);
    let (r_i, c_i) = (rows as i32, cols as i32);
    let cfg = LaunchConfig {
        grid_dim: (rows.div_ceil(32) as u32, cols.div_ceil(32) as u32, 1),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };
    let mut l = gpu.stream.launch_builder(f);
    l.arg(&s_addr);
    l.arg(&d_addr);
    l.arg(&r_i);
    l.arg(&c_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// Launch one Marlin tile group. `x` and `out` are the full buffers; the
/// pointers are offset by `row_off` here.
#[allow(clippy::too_many_arguments)]
pub fn marlin_gemm(
    gpu: &Gpu,
    f: &CudaFunction,
    x: &CudaSlice<bf16>,
    qweight: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    scales: &CudaSlice<bf16>,
    zp: &CudaSlice<i32>,
    locks: &CudaSlice<i32>,
    row_off: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let a_addr = ptr_of(x, gpu) + (row_off * k * 2) as u64;
    let (c_base, _r0) = out.device_ptr_mut(&gpu.stream);
    let c_addr = c_base + (row_off * n * 2) as u64;
    let (b, _r1) = qweight.device_ptr(&gpu.stream);
    let (s, _r2) = scales.device_ptr(&gpu.stream);
    let (z, _r3) = zp.device_ptr(&gpu.stream);
    let (l, _r4) = locks.device_ptr(&gpu.stream);
    let (b_addr, s_addr, z_addr, l_addr) = (b, s, z, l);
    let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
    let (lda, zero) = (k as i32, 0u64);
    let (f_bias, f_atomic, f_reduce) = (false, false, false);
    let cfg = LaunchConfig {
        grid_dim: (128, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: MR_SHARED as u32,
    };
    let mut l = gpu.stream.launch_builder(f);
    l.arg(&a_addr);
    l.arg(&b_addr);
    l.arg(&c_addr);
    l.arg(&zero); // C_tmp
    l.arg(&zero); // b_bias
    l.arg(&zero); // a_scales
    l.arg(&s_addr);
    l.arg(&zero); // global scale
    l.arg(&z_addr);
    l.arg(&m_i);
    l.arg(&n_i);
    l.arg(&k_i);
    l.arg(&lda);
    l.arg(&l_addr); // locks
    l.arg(&f_bias);
    l.arg(&f_atomic);
    l.arg(&f_reduce);
    l.arg(&MR_SHARED);
    unsafe { l.launch(cfg) }?;
    if std::env::var("JEV_MARLIN_DEBUG").is_ok() {
        gpu.synchronize()?;
        eprintln!("marlin gemm ok: m={m} n={n} k={k} off={row_off}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn attention_warp(
    gpu: &Gpu,
    k: &CudaFunction,
    q: &CudaSlice<bf16>,
    kk: &CudaSlice<bf16>,
    v: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    seq: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    scale: f32,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, seq.div_ceil(4) as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (seq_i, nh, nkv, hd_i) = (seq as i32, n_heads as i32, n_kv as i32, hd as i32);
    let mut l = gpu.stream.launch_builder(k);
    l.arg(q);
    l.arg(kk);
    l.arg(v);
    l.arg(out);
    l.arg(&seq_i);
    l.arg(&nh);
    l.arg(&nkv);
    l.arg(&hd_i);
    l.arg(&scale);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn attention_tc(
    gpu: &Gpu,
    k: &CudaFunction,
    q: &CudaSlice<bf16>,
    kk: &CudaSlice<bf16>,
    v: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    seq: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    scale: f32,
) -> Result<()> {
    // hd is one of 64/128/256 (validated by the caller / loaded cubin).
    let smem = match hd {
        64 => 27648usize,
        128 => 52224,
        256 => 101376,
        _ => anyhow::bail!("attention_tc: unsupported head_dim {hd}"),
    };
    let cfg = LaunchConfig {
        grid_dim: (seq.div_ceil(64) as u32, n_heads as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let (seq_i, nh, nkv) = (seq as i32, n_heads as i32, n_kv as i32);
    let mut l = gpu.stream.launch_builder(k);
    l.arg(q);
    l.arg(kk);
    l.arg(v);
    l.arg(out);
    l.arg(&seq_i);
    l.arg(&nh);
    l.arg(&nkv);
    l.arg(&scale);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

pub fn conv1d_silu(
    gpu: &Gpu,
    k: &CudaFunction,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    seq: usize,
    ch: usize,
) -> Result<()> {
    let cfg = cfg_1d(ch, 128);
    let seq_i = seq as i32;
    let ch_i = ch as i32;
    let mut l = gpu.stream.launch_builder(k);
    l.arg(x);
    l.arg(w);
    l.arg(out);
    l.arg(&seq_i);
    l.arg(&ch_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_prepare(
    gpu: &Gpu,
    k: &CudaFunction,
    qkv: &CudaSlice<bf16>,
    b: &CudaSlice<bf16>,
    a: &CudaSlice<bf16>,
    a_log: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    qf: &mut CudaSlice<f32>,
    kf: &mut CudaSlice<f32>,
    vf: &mut CudaSlice<f32>,
    g: &mut CudaSlice<f32>,
    beta: &mut CudaSlice<f32>,
    seq: usize,
    num_k: usize,
    num_v: usize,
    dk: usize,
    dv: usize,
) -> Result<()> {
    let block = 128u32;
    let cfg = LaunchConfig {
        grid_dim: (seq as u32, num_v as u32, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 4,
    };
    let nk = num_k as i32;
    let nv = num_v as i32;
    let dk_i = dk as i32;
    let dv_i = dv as i32;
    let mut l = gpu.stream.launch_builder(k);
    l.arg(qkv);
    l.arg(b);
    l.arg(a);
    l.arg(a_log);
    l.arg(dt_bias);
    l.arg(qf);
    l.arg(kf);
    l.arg(vf);
    l.arg(g);
    l.arg(beta);
    l.arg(&nk);
    l.arg(&nv);
    l.arg(&dk_i);
    l.arg(&dv_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent(
    gpu: &Gpu,
    k: &CudaFunction,
    qf: &CudaSlice<f32>,
    kf: &CudaSlice<f32>,
    vf: &CudaSlice<f32>,
    g: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    seq: usize,
    heads: usize,
    dk: usize,
    dv: usize,
) -> Result<()> {
    if dv > 128 {
        anyhow::bail!("gdn_recurrent supports dv <= 128, got {dv}");
    }
    let cfg = LaunchConfig {
        grid_dim: (heads as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let seq_i = seq as i32;
    let heads_i = heads as i32;
    let dk_i = dk as i32;
    let dv_i = dv as i32;
    let mut l = gpu.stream.launch_builder(k);
    l.arg(qf);
    l.arg(kf);
    l.arg(vf);
    l.arg(g);
    l.arg(beta);
    l.arg(state);
    l.arg(out);
    l.arg(&seq_i);
    l.arg(&heads_i);
    l.arg(&dk_i);
    l.arg(&dv_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

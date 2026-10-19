//! Safe wrappers around the custom CUDA kernels.
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
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
    pub conv1d_silu: CudaFunction,
    pub gdn_prepare: CudaFunction,
    pub gdn_recurrent: CudaFunction,
    pub dequant_w4a16: CudaFunction,
    pub gemm_w4a16: CudaFunction,
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
            conv1d_silu: gpu.func("conv1d_silu_bf16")?,
            gdn_prepare: gpu.func("gdn_prepare")?,
            gdn_recurrent: gpu.func("gdn_recurrent")?,
            dequant_w4a16: gpu.func("dequant_w4a16_bf16")?,
            gemm_w4a16: gpu.func("gemm_w4a16_bf16")?,
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

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
    pub add_rmsnorm: CudaFunction,
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
    /// Broadcast one f32 seed state into every row of a batched state buffer.
    pub gdn_seed_broadcast: CudaFunction,
    pub gdn_recurrent_shared: CudaFunction,
    pub gdn_recurrent_split: CudaFunction,
    pub gdn_recurrent_shared_batched: CudaFunction,
    pub gdn_recurrent_split_batched: CudaFunction,
    pub gdn_recurrent_reg_split_batched: CudaFunction,
    pub gdn_chunked: CudaFunction,
    pub gdn_chunked_tc: CudaFunction,
    pub gdn_chunked_tc_prof: CudaFunction,
    pub gdn_chunked_tc_batched: CudaFunction,
    pub conv1d_silu_batched: CudaFunction,
    pub conv1d_silu_tiled: CudaFunction,
    pub conv1d_silu_tiled_batched: CudaFunction,
    pub gather_rows: CudaFunction,
    pub embed_gather: CudaFunction,
    pub quant_fp8: CudaFunction,
    pub quant_int8: CudaFunction,
    pub quant_int8_bf16: CudaFunction,
    pub quant_int8_group_bf16: CudaFunction,
    pub quant_int8_group_silu_bf16: CudaFunction,
    pub quant_fp8_bf16: CudaFunction,
    pub scale_fp8_gemm: CudaFunction,
    pub dequant_w4a16: CudaFunction,
    pub gemm_w4a16: CudaFunction,
    pub gemm_w4a16_m128: CudaFunction,
    /// Marlin kernels indexed by thread_m_blocks - 1 (1..4).
    pub marlin_cfgs: Vec<CudaFunction>,
    pub marlin_fp8_cfgs: Vec<CudaFunction>,
    pub marlin_int8_cfgs: Vec<CudaFunction>,
    pub marlin_int8_group_cfgs: Vec<CudaFunction>,
    /// Group-128 weight configs: W4A16 and W4A8 per-token.
    pub marlin_g128_cfgs: Vec<CudaFunction>,
    pub marlin_int8_g128_cfgs: Vec<CudaFunction>,
    pub marlin_repack: CudaFunction,
    pub marlin_repack_a8: CudaFunction,
    pub awq_fp8_transform: CudaFunction,
    pub marlin_transpose: CudaFunction,
}

impl Kernels {
    pub fn load(gpu: &Gpu) -> Result<Self> {
        Ok(Self {
            rmsnorm: gpu.func("rmsnorm_bf16")?,
            add_rmsnorm: gpu.func("add_rmsnorm_bf16")?,
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
            gdn_seed_broadcast: gpu.func("gdn_seed_broadcast")?,
            gdn_recurrent_shared: {
                let f = gpu.func("gdn_recurrent_shared")?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    65536,
                )?;
                f
            },
            gdn_recurrent_split: {
                let f = gpu.func("gdn_recurrent_split")?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    32768,
                )?;
                f
            },
            gdn_recurrent_shared_batched: {
                let f = gpu.func("gdn_recurrent_shared_batched")?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    65536,
                )?;
                f
            },
            gdn_recurrent_reg_split_batched: gpu.func("gdn_recurrent_reg_split_batched")?,
            gdn_recurrent_split_batched: {
                let f = gpu.func("gdn_recurrent_split_batched")?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    32768,
                )?;
                f
            },
            gdn_chunked: {
                let f = gpu.func("gdn_chunked_f32")?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    90112,
                )?;
                f
            },
            gdn_chunked_tc: {
                let f = gpu.func("gdn_chunked_tc")?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    90112,
                )?;
                f
            },
            gdn_chunked_tc_prof: {
                let f = gpu.func("gdn_chunked_tc_prof")?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    90112,
                )?;
                f
            },
            gdn_chunked_tc_batched: {
                let f = gpu.func("gdn_chunked_tc_batched")?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    90112,
                )?;
                f
            },
            conv1d_silu_batched: gpu.func("conv1d_silu_batched_bf16")?,
            conv1d_silu_tiled: gpu.func("conv1d_silu_tiled_bf16")?,
            conv1d_silu_tiled_batched: gpu.func("conv1d_silu_tiled_batched_bf16")?,
            gather_rows: gpu.func("gather_rows_bf16")?,
            embed_gather: gpu.func("embed_gather_bf16")?,
            quant_fp8: gpu.func("quant_fp8_per_token_f32")?,
            quant_int8: gpu.func("quant_int8_per_token_f32")?,
            quant_int8_bf16: gpu.func("quant_int8_per_token_bf16")?,
            quant_int8_group_bf16: gpu.func("quant_int8_group_bf16")?,
            quant_int8_group_silu_bf16: gpu.func("quant_int8_group_silu_bf16")?,
            quant_fp8_bf16: gpu.func("quant_fp8_per_token_bf16")?,
            scale_fp8_gemm: gpu.func("scale_fp8_gemm_bf16")?,
            dequant_w4a16: gpu.func("dequant_w4a16_bf16")?,
            marlin_cfgs: {
                const NAMES: [&str; 5] = [
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi1ELi8ELi8ELb1ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi1ELi8ELi8ELb0ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi2ELi16ELi4ELb0ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi3ELi16ELi4ELb0ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi4ELi16ELi4ELb0ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
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
            marlin_fp8_cfgs: {
                const NAMES: [&str; 4] = [
                    "_ZN6marlin6MarlinILl2814749767172868ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi1ELi8ELi8ELb0ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl2814749767172868ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi2ELi16ELi4ELb0ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl2814749767172868ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi3ELi16ELi4ELb0ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl2814749767172868ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi4ELi16ELi4ELb0ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
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
            marlin_int8_cfgs: {
                const NAMES: [&str; 4] = [
                    "_ZN6marlin6MarlinILl1125899906909952ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi1ELi8ELi8ELb0ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909952ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi2ELi16ELi4ELb0ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909952ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi3ELi16ELi4ELb0ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909952ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi4ELi16ELi4ELb0ELi4ELi4ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
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
            marlin_int8_group_cfgs: {
                const NAMES: [&str; 4] = [
                    "_ZN6marlin6MarlinILl1125899906909952ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi1ELi8ELi8ELb0ELi4ELi4ELb0ELb1EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909952ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi2ELi16ELi4ELb0ELi4ELi4ELb0ELb1EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909952ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi3ELi16ELi4ELb0ELi4ELi4ELb0ELb1EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909952ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi4ELi16ELi4ELb0ELi4ELi4ELb0ELb1EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
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
            marlin_g128_cfgs: {
                const NAMES: [&str; 5] = [
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi1ELi8ELi8ELb1ELi4ELi8ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi1ELi8ELi8ELb0ELi4ELi8ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi2ELi16ELi4ELb0ELi4ELi8ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi3ELi16ELi4ELb0ELi4ELi8ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909960ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi4ELi16ELi4ELb0ELi4ELi8ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
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
            marlin_int8_g128_cfgs: {
                const NAMES: [&str; 4] = [
                    "_ZN6marlin6MarlinILl1125899906909952ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi1ELi8ELi8ELb0ELi4ELi8ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909952ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi2ELi16ELi4ELb0ELi4ELi8ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909952ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi3ELi16ELi4ELb0ELi4ELi8ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
                    "_ZN6marlin6MarlinILl1125899906909952ELl1125899906843648ELl1125899906909960ELl1125899906909960ELi256ELi4ELi16ELi4ELb0ELi4ELi8ELb0ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
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
                    "_ZN6marlin33decidra_gptq_marlin_repack_kernelILi256ELi4ELb0EEEvPKjPjii",
                )?;
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    4096,
                )?;
                f
            },
            marlin_repack_a8: gpu.func_marlin(
                "_ZN6marlin33decidra_gptq_marlin_repack_kernelILi256ELi4ELb1EEEvPKjPjii",
            )?,
            awq_fp8_transform: gpu.func("awq_fp8_transform_q4")?,
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
#[allow(clippy::too_many_arguments)]
pub fn add_rmsnorm(
    gpu: &Gpu,
    k: &CudaFunction,
    x: &CudaSlice<bf16>,
    mix: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    residual: &mut CudaSlice<bf16>,
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
    l.arg(mix);
    l.arg(w);
    l.arg(residual);
    l.arg(out);
    l.arg(&cols_i);
    l.arg(&eps);
    l.arg(&zc);
    unsafe { l.launch(cfg) }?;
    Ok(())
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

fn marlin_debug() -> bool {
    static DEBUG: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("DECIDRA_MARLIN_DEBUG").is_ok());
    *DEBUG
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
    sms: usize,
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
        grid_dim: (sms as u32, 1, 1),
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
    if marlin_debug() {
        // Log only; a sync here would serialize any profiling run that has the
        // debug variable set. Use DECIDRA_PROFILE / events for timing.
        eprintln!("marlin gemm: m={m} n={n} k={k} off={row_off}");
    }
    Ok(())
}

/// Canonical AWQ -> W4A8 transform: CT packed [N,K/8] + CT zp -> q2 [K/8,N].
#[allow(clippy::too_many_arguments)]
pub fn awq_fp8_transform(
    gpu: &Gpu,
    f: &CudaFunction,
    packed: &CudaSlice<i32>,
    zp: &CudaSlice<i32>,
    out: &mut CudaSlice<i32>,
    k: usize,
    n: usize,
    groups: usize,
) -> Result<()> {
    let total = (k / 8) * n;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (k_i, n_i, g_i) = (k as i32, n as i32, groups as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(packed);
    l.arg(zp);
    l.arg(out);
    l.arg(&k_i);
    l.arg(&n_i);
    l.arg(&g_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// Per-token symmetric INT8 quantization of f32 activations [rows, k].
#[allow(clippy::too_many_arguments)]
pub fn quant_int8(
    gpu: &Gpu,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    rows: usize,
    k: usize,
    global_scale: f32,
    clip: f32,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };
    let (k_i, r_i) = (k as i32, rows as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(x);
    l.arg(out);
    l.arg(scales);
    l.arg(&r_i);
    l.arg(&k_i);
    l.arg(&global_scale);
    l.arg(&clip);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// Per-token FP8 E4M3 quantization of f32 activations [rows, k].
#[allow(clippy::too_many_arguments)]
pub fn quant_fp8(
    gpu: &Gpu,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
    rows: usize,
    k: usize,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };
    let (k_i, r_i) = (k as i32, rows as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(x);
    l.arg(out);
    l.arg(scales);
    l.arg(&r_i);
    l.arg(&k_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// Per-token symmetric INT8 quantization of bf16 activations.
#[allow(clippy::too_many_arguments)]
pub fn quant_int8_bf16(
    gpu: &Gpu,
    f: &CudaFunction,
    x: &CudaSlice<bf16>,
    out: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    rows: usize,
    k: usize,
    global_scale: f32,
    clip: f32,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };
    let (k_i, r_i) = (k as i32, rows as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(x);
    l.arg(out);
    l.arg(scales);
    l.arg(&r_i);
    l.arg(&k_i);
    l.arg(&global_scale);
    l.arg(&clip);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// Per-token, per-K-group symmetric INT8 quantization of bf16 activations.
#[allow(clippy::too_many_arguments)]
pub fn quant_int8_group_bf16(
    gpu: &Gpu,
    f: &CudaFunction,
    x: &CudaSlice<bf16>,
    out: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    rows: usize,
    k: usize,
    group: usize,
    global_scale: f32,
    clip: f32,
    uniform: bool,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (k_i, r_i, g_i) = (k as i32, rows as i32, group as i32);
    let u_i = uniform as i32;
    let mut l = gpu.stream.launch_builder(f);
    l.arg(x);
    l.arg(out);
    l.arg(scales);
    l.arg(&r_i);
    l.arg(&k_i);
    l.arg(&g_i);
    l.arg(&global_scale);
    l.arg(&clip);
    l.arg(&u_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

pub fn quant_fp8_bf16(
    gpu: &Gpu,
    f: &CudaFunction,
    x: &CudaSlice<bf16>,
    out: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
    rows: usize,
    k: usize,
    clip: f32,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };
    let (r_i, k_i) = (rows as i32, k as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(x);
    l.arg(out);
    l.arg(scales);
    l.arg(&r_i);
    l.arg(&k_i);
    l.arg(&clip);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn scale_fp8_gemm(
    gpu: &Gpu,
    f: &CudaFunction,
    c: &CudaSlice<f32>,
    w_scale: &CudaSlice<f32>,
    a_scale: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let cfg = cfg_1d(rows * cols, 256);
    let (r_i, c_i) = (rows as i32, cols as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(c);
    l.arg(w_scale);
    l.arg(a_scale);
    l.arg(out);
    l.arg(&r_i);
    l.arg(&c_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// SiLU(gate)*up quantized per K group (down-projection input).
#[allow(clippy::too_many_arguments)]
pub fn quant_int8_group_silu_bf16(
    gpu: &Gpu,
    f: &CudaFunction,
    gate: &CudaSlice<bf16>,
    up: &CudaSlice<bf16>,
    out: &mut CudaSlice<i8>,
    scales: &mut CudaSlice<f32>,
    rows: usize,
    k: usize,
    group: usize,
    global_scale: f32,
    clip: f32,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (k_i, r_i, g_i) = (k as i32, rows as i32, group as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(gate);
    l.arg(up);
    l.arg(out);
    l.arg(scales);
    l.arg(&r_i);
    l.arg(&k_i);
    l.arg(&g_i);
    l.arg(&global_scale);
    l.arg(&clip);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// Device-side embedding gather: `table[vocab, hidden]`, `ids[seq]` -> `out`.
#[allow(clippy::too_many_arguments)]
pub fn embed_gather(
    gpu: &Gpu,
    f: &CudaFunction,
    table: &CudaSlice<bf16>,
    ids: &CudaSlice<u32>,
    out: &mut CudaSlice<bf16>,
    seq: usize,
    hidden: usize,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (seq as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (s_i, h_i) = (seq as i32, hidden as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(table);
    l.arg(ids);
    l.arg(out);
    l.arg(&s_i);
    l.arg(&h_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// INT8-activation Marlin tile group (signed i8 A, f32 per-row or per-group scales).
#[allow(clippy::too_many_arguments)]
pub fn marlin_gemm_int8(
    gpu: &Gpu,
    f: &CudaFunction,
    x: &CudaSlice<i8>,
    qweight: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    scales: &CudaSlice<bf16>,
    zp: &CudaSlice<i32>,
    a_scales: &CudaSlice<f32>,
    locks: &CudaSlice<i32>,
    row_off: usize,
    m: usize,
    n: usize,
    k: usize,
    a_group_size: usize,
    sms: usize,
) -> Result<()> {
    let (a_base, _ra) = x.device_ptr(&gpu.stream);
    let a_addr = a_base + (row_off * k) as u64;
    let (c_base, _r0) = out.device_ptr_mut(&gpu.stream);
    let c_addr = c_base + (row_off * n * 2) as u64;
    let (b, _r1) = qweight.device_ptr(&gpu.stream);
    let (s, _r2) = scales.device_ptr(&gpu.stream);
    let (z, _r3) = zp.device_ptr(&gpu.stream);
    let (as_, _r4) = a_scales.device_ptr(&gpu.stream);
    let a_row = if a_group_size > 0 {
        k / a_group_size
    } else {
        1
    };
    let as_addr = as_ + (row_off * a_row * 4) as u64;
    let (l, _r5) = locks.device_ptr(&gpu.stream);
    let (b_addr, s_addr, z_addr, l_addr) = (b, s, z, l);
    let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
    let (lda, zero) = (k as i32, 0u64);
    let (f_bias, f_atomic, f_reduce) = (false, false, false);
    let cfg = LaunchConfig {
        grid_dim: (sms as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: MR_SHARED as u32,
    };
    let mut l = gpu.stream.launch_builder(f);
    l.arg(&a_addr);
    l.arg(&b_addr);
    l.arg(&c_addr);
    l.arg(&zero);
    l.arg(&zero);
    l.arg(&as_addr);
    l.arg(&s_addr);
    l.arg(&zero);
    l.arg(&z_addr);
    l.arg(&m_i);
    l.arg(&n_i);
    l.arg(&k_i);
    l.arg(&lda);
    l.arg(&l_addr);
    l.arg(&f_bias);
    l.arg(&f_atomic);
    l.arg(&f_reduce);
    l.arg(&MR_SHARED);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// FP8-activation Marlin tile group. `x` is E4M3 (one byte per element) and
/// `a_scales` is one f32 scale per row; both are offset by `row_off`.
#[allow(clippy::too_many_arguments)]
pub fn marlin_gemm_fp8(
    gpu: &Gpu,
    f: &CudaFunction,
    x: &CudaSlice<u8>,
    qweight: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    scales: &CudaSlice<bf16>,
    zp: &CudaSlice<i32>,
    a_scales: &CudaSlice<f32>,
    locks: &CudaSlice<i32>,
    row_off: usize,
    m: usize,
    n: usize,
    k: usize,
    sms: usize,
) -> Result<()> {
    let (a_base, _ra) = x.device_ptr(&gpu.stream);
    let a_addr = a_base + (row_off * k) as u64;
    let (c_base, _r0) = out.device_ptr_mut(&gpu.stream);
    let c_addr = c_base + (row_off * n * 2) as u64;
    let (b, _r1) = qweight.device_ptr(&gpu.stream);
    let (s, _r2) = scales.device_ptr(&gpu.stream);
    let (z, _r3) = zp.device_ptr(&gpu.stream);
    let (as_, _r4) = a_scales.device_ptr(&gpu.stream);
    let as_addr = as_ + (row_off * 4) as u64;
    let (l, _r5) = locks.device_ptr(&gpu.stream);
    let (b_addr, s_addr, z_addr, l_addr) = (b, s, z, l);
    let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
    let (lda, zero) = (k as i32, 0u64);
    let (f_bias, f_atomic, f_reduce) = (false, false, false);
    let cfg = LaunchConfig {
        grid_dim: (sms as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: MR_SHARED as u32,
    };
    let mut l = gpu.stream.launch_builder(f);
    l.arg(&a_addr);
    l.arg(&b_addr);
    l.arg(&c_addr);
    l.arg(&zero); // C_tmp
    l.arg(&zero); // b_bias
    l.arg(&as_addr); // a_scales
    l.arg(&s_addr);
    l.arg(&zero); // global scale
    l.arg(&z_addr);
    l.arg(&m_i);
    l.arg(&n_i);
    l.arg(&k_i);
    l.arg(&lda);
    l.arg(&l_addr);
    l.arg(&f_bias);
    l.arg(&f_atomic);
    l.arg(&f_reduce);
    l.arg(&MR_SHARED);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn attention_warp<Q, K, V, O>(
    gpu: &Gpu,
    f: &CudaFunction,
    q: &Q,
    kk: &K,
    v: &V,
    out: &mut O,
    seq: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    scale: f32,
) -> Result<()>
where
    Q: DevicePtr<bf16>,
    K: DevicePtr<bf16>,
    V: DevicePtr<bf16>,
    O: DevicePtrMut<bf16>,
{
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, seq.div_ceil(4) as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (seq_i, nh, nkv, hd_i) = (seq as i32, n_heads as i32, n_kv as i32, hd as i32);
    let (qp, _r0) = q.device_ptr(&gpu.stream);
    let (kp, _r1) = kk.device_ptr(&gpu.stream);
    let (vp, _r2) = v.device_ptr(&gpu.stream);
    let (op, _r3) = out.device_ptr_mut(&gpu.stream);
    let (qp, kp, vp, op) = (qp, kp, vp, op);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(&qp);
    l.arg(&kp);
    l.arg(&vp);
    l.arg(&op);
    l.arg(&seq_i);
    l.arg(&nh);
    l.arg(&nkv);
    l.arg(&hd_i);
    l.arg(&scale);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn attention_tc<Q, K, V, O>(
    gpu: &Gpu,
    f: &CudaFunction,
    q: &Q,
    kk: &K,
    v: &V,
    out: &mut O,
    seq: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    scale: f32,
) -> Result<()>
where
    Q: DevicePtr<bf16>,
    K: DevicePtr<bf16>,
    V: DevicePtr<bf16>,
    O: DevicePtrMut<bf16>,
{
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
    let (qp, _r0) = q.device_ptr(&gpu.stream);
    let (kp, _r1) = kk.device_ptr(&gpu.stream);
    let (vp, _r2) = v.device_ptr(&gpu.stream);
    let (op, _r3) = out.device_ptr_mut(&gpu.stream);
    let (qp, kp, vp, op) = (qp, kp, vp, op);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(&qp);
    l.arg(&kp);
    l.arg(&vp);
    l.arg(&op);
    l.arg(&seq_i);
    l.arg(&nh);
    l.arg(&nkv);
    l.arg(&scale);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_shared<QF, KF, VF, G, B, ST, O>(
    gpu: &Gpu,
    f: &CudaFunction,
    qf: &QF,
    kf: &KF,
    vf: &VF,
    g: &G,
    beta: &B,
    state: &mut ST,
    out: &mut O,
    seq: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    vchunks: usize,
) -> Result<()>
where
    QF: DevicePtr<f32>,
    KF: DevicePtr<f32>,
    VF: DevicePtr<f32>,
    G: DevicePtr<f32>,
    B: DevicePtr<f32>,
    ST: DevicePtrMut<f32>,
    O: DevicePtrMut<bf16>,
{
    if dv % vchunks != 0 {
        anyhow::bail!("dv={dv} not divisible by gdn chunks={vchunks}");
    }
    let vc = dv / vchunks;
    let smem = dk * vc * 4;
    let cfg = LaunchConfig {
        grid_dim: ((heads * vchunks) as u32, 1, 1),
        block_dim: (vc as u32, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let (seq_i, heads_i, dk_i, dv_i, vc_i, vch_i) = (
        seq as i32,
        heads as i32,
        dk as i32,
        dv as i32,
        vc as i32,
        vchunks as i32,
    );
    let (qfp, _r0) = qf.device_ptr(&gpu.stream);
    let (kfp, _r1) = kf.device_ptr(&gpu.stream);
    let (vfp, _r2) = vf.device_ptr(&gpu.stream);
    let (gp, _r3) = g.device_ptr(&gpu.stream);
    let (bp, _r4) = beta.device_ptr(&gpu.stream);
    let (sp, _r5) = state.device_ptr_mut(&gpu.stream);
    let (op, _r6) = out.device_ptr_mut(&gpu.stream);
    let (qfp, kfp, vfp, gp, bp, sp, op) = (qfp, kfp, vfp, gp, bp, sp, op);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(&qfp);
    l.arg(&kfp);
    l.arg(&vfp);
    l.arg(&gp);
    l.arg(&bp);
    l.arg(&sp);
    l.arg(&op);
    l.arg(&seq_i);
    l.arg(&heads_i);
    l.arg(&dk_i);
    l.arg(&dv_i);
    l.arg(&vc_i);
    l.arg(&vch_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_split<QF, KF, VF, G, B, ST, O>(
    gpu: &Gpu,
    f: &CudaFunction,
    qf: &QF,
    kf: &KF,
    vf: &VF,
    g: &G,
    beta: &B,
    state: &mut ST,
    out: &mut O,
    seq: usize,
    heads: usize,
    dk: usize,
    dv: usize,
) -> Result<()>
where
    QF: DevicePtr<f32>,
    KF: DevicePtr<f32>,
    VF: DevicePtr<f32>,
    G: DevicePtr<f32>,
    B: DevicePtr<f32>,
    ST: DevicePtrMut<f32>,
    O: DevicePtrMut<bf16>,
{
    let vc = dv.min(32);
    if dv % vc != 0 {
        anyhow::bail!("dv={dv} not divisible by vc={vc}");
    }
    let chunks = dv / vc;
    let threads = vc * 4;
    let smem = dk * (vc + 1) * 4;
    let cfg = LaunchConfig {
        grid_dim: ((heads * chunks) as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let (seq_i, heads_i, dk_i, dv_i, vc_i, ch_i) = (
        seq as i32,
        heads as i32,
        dk as i32,
        dv as i32,
        vc as i32,
        chunks as i32,
    );
    let (qfp, _r0) = qf.device_ptr(&gpu.stream);
    let (kfp, _r1) = kf.device_ptr(&gpu.stream);
    let (vfp, _r2) = vf.device_ptr(&gpu.stream);
    let (gp, _r3) = g.device_ptr(&gpu.stream);
    let (bp, _r4) = beta.device_ptr(&gpu.stream);
    let (sp, _r5) = state.device_ptr_mut(&gpu.stream);
    let (op, _r6) = out.device_ptr_mut(&gpu.stream);
    let (qfp, kfp, vfp, gp, bp, sp, op) = (qfp, kfp, vfp, gp, bp, sp, op);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(&qfp);
    l.arg(&kfp);
    l.arg(&vfp);
    l.arg(&gp);
    l.arg(&bp);
    l.arg(&sp);
    l.arg(&op);
    l.arg(&seq_i);
    l.arg(&heads_i);
    l.arg(&dk_i);
    l.arg(&dv_i);
    l.arg(&vc_i);
    l.arg(&ch_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// Copy one recurrent seed into every row of a batched state buffer. One
/// launch replaces `batch` per-row `memcpy_dtod` calls in the state-once fork.
pub fn gdn_seed_broadcast<S, ST>(
    gpu: &Gpu,
    f: &CudaFunction,
    seed: &S,
    state: &mut ST,
    elems: usize,
    batch: usize,
) -> Result<()>
where
    S: DevicePtr<f32>,
    ST: DevicePtrMut<f32>,
{
    if elems == 0 || batch == 0 {
        anyhow::bail!("gdn_seed_broadcast needs non-empty seed and batch");
    }
    let cfg = LaunchConfig {
        grid_dim: (elems.div_ceil(256) as u32, batch as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let elems_i = elems as i32;
    let (sp, _r0) = seed.device_ptr(&gpu.stream);
    let (dp, _r1) = state.device_ptr_mut(&gpu.stream);
    let (sp, dp) = (sp, dp);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(&sp);
    l.arg(&dp);
    l.arg(&elems_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent<QF, KF, VF, G, B, ST, O>(
    gpu: &Gpu,
    f: &CudaFunction,
    qf: &QF,
    kf: &KF,
    vf: &VF,
    g: &G,
    beta: &B,
    state: &mut ST,
    out: &mut O,
    seq: usize,
    heads: usize,
    dk: usize,
    dv: usize,
) -> Result<()>
where
    QF: DevicePtr<f32>,
    KF: DevicePtr<f32>,
    VF: DevicePtr<f32>,
    G: DevicePtr<f32>,
    B: DevicePtr<f32>,
    ST: DevicePtrMut<f32>,
    O: DevicePtrMut<bf16>,
{
    if dv > 128 {
        anyhow::bail!("gdn_recurrent supports dv <= 128, got {dv}");
    }
    let cfg = LaunchConfig {
        grid_dim: (heads as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (seq_i, heads_i, dk_i, dv_i) = (seq as i32, heads as i32, dk as i32, dv as i32);
    let (qfp, _r0) = qf.device_ptr(&gpu.stream);
    let (kfp, _r1) = kf.device_ptr(&gpu.stream);
    let (vfp, _r2) = vf.device_ptr(&gpu.stream);
    let (gp, _r3) = g.device_ptr(&gpu.stream);
    let (bp, _r4) = beta.device_ptr(&gpu.stream);
    let (sp, _r5) = state.device_ptr_mut(&gpu.stream);
    let (op, _r6) = out.device_ptr_mut(&gpu.stream);
    let (qfp, kfp, vfp, gp, bp, sp, op) = (qfp, kfp, vfp, gp, bp, sp, op);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(&qfp);
    l.arg(&kfp);
    l.arg(&vfp);
    l.arg(&gp);
    l.arg(&bp);
    l.arg(&sp);
    l.arg(&op);
    l.arg(&seq_i);
    l.arg(&heads_i);
    l.arg(&dk_i);
    l.arg(&dv_i);
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
pub fn conv1d_silu_batched(
    gpu: &Gpu,
    f: &CudaFunction,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    offsets: &CudaSlice<i32>,
    lens: &CudaSlice<i32>,
    batch: usize,
    ch: usize,
) -> Result<()> {
    let cfg = cfg_1d(ch, 128);
    let (b_i, ch_i) = (batch as i32, ch as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(x);
    l.arg(w);
    l.arg(out);
    l.arg(offsets);
    l.arg(lens);
    l.arg(&b_i);
    l.arg(&ch_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// Time tile of `conv1d_silu_tiled_bf16`; must match `CONV_TILE` in kernels.cu.
pub const CONV_TILE: usize = 16;

pub fn conv1d_silu_tiled(
    gpu: &Gpu,
    f: &CudaFunction,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    seq: usize,
    ch: usize,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (ch.div_ceil(256) as u32, seq.div_ceil(CONV_TILE) as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (seq_i, ch_i) = (seq as i32, ch as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(x);
    l.arg(w);
    l.arg(out);
    l.arg(&seq_i);
    l.arg(&ch_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_silu_tiled_batched(
    gpu: &Gpu,
    f: &CudaFunction,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    offsets: &CudaSlice<i32>,
    lens: &CudaSlice<i32>,
    batch: usize,
    ch: usize,
    max_len: usize,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (
            ch.div_ceil(256) as u32,
            max_len.div_ceil(CONV_TILE) as u32,
            batch as u32,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (b_i, ch_i) = (batch as i32, ch as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(x);
    l.arg(w);
    l.arg(out);
    l.arg(offsets);
    l.arg(lens);
    l.arg(&b_i);
    l.arg(&ch_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// Chunk length of `gdn_chunked_f32`; must match `GDN_CHUNK_C` in kernels.cu.
pub const GDN_CHUNK_C: usize = 32;
/// Value columns owned by one chunked CTA.
pub const GDN_CHUNK_VC: usize = 64;

#[allow(clippy::too_many_arguments)]
pub fn gdn_chunked<Q, K, V, G, B, ST, O>(
    gpu: &Gpu,
    f: &CudaFunction,
    q: &Q,
    k: &K,
    v: &V,
    g: &G,
    beta: &B,
    state: &mut ST,
    out: &mut O,
    seq: usize,
    heads: usize,
    dk: usize,
    dv: usize,
) -> Result<()>
where
    Q: DevicePtr<f32>,
    K: DevicePtr<f32>,
    V: DevicePtr<f32>,
    G: DevicePtr<f32>,
    B: DevicePtr<f32>,
    ST: DevicePtrMut<f32>,
    O: DevicePtrMut<bf16>,
{
    gdn_chunked_impl(
        gpu, f, q, k, v, g, beta, state, out, seq, heads, dk, dv, None,
    )
}

/// Same launch as [`gdn_chunked`], using `gdn_chunked_tc_prof`: each block
/// appends `clock64()` deltas for the 10 internal phases to `prof`
/// (`[blocks, 10]`).
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunked_prof<Q, K, V, G, B, ST, O>(
    gpu: &Gpu,
    f: &CudaFunction,
    q: &Q,
    k: &K,
    v: &V,
    g: &G,
    beta: &B,
    state: &mut ST,
    out: &mut O,
    prof: &mut CudaSlice<u64>,
    seq: usize,
    heads: usize,
    dk: usize,
    dv: usize,
) -> Result<()>
where
    Q: DevicePtr<f32>,
    K: DevicePtr<f32>,
    V: DevicePtr<f32>,
    G: DevicePtr<f32>,
    B: DevicePtr<f32>,
    ST: DevicePtrMut<f32>,
    O: DevicePtrMut<bf16>,
{
    gdn_chunked_impl(
        gpu,
        f,
        q,
        k,
        v,
        g,
        beta,
        state,
        out,
        seq,
        heads,
        dk,
        dv,
        Some(prof),
    )
}

#[allow(clippy::too_many_arguments)]
fn gdn_chunked_impl<Q, K, V, G, B, ST, O>(
    gpu: &Gpu,
    f: &CudaFunction,
    q: &Q,
    k: &K,
    v: &V,
    g: &G,
    beta: &B,
    state: &mut ST,
    out: &mut O,
    seq: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    prof: Option<&mut CudaSlice<u64>>,
) -> Result<()>
where
    Q: DevicePtr<f32>,
    K: DevicePtr<f32>,
    V: DevicePtr<f32>,
    G: DevicePtr<f32>,
    B: DevicePtr<f32>,
    ST: DevicePtrMut<f32>,
    O: DevicePtrMut<bf16>,
{
    if dk != 128 || dv != 128 {
        anyhow::bail!("chunked GDN kernel requires dk=dv=128, got dk={dk} dv={dv}");
    }
    let smem = (dk * GDN_CHUNK_VC
        + 2 * GDN_CHUNK_C * dk
        + 2 * GDN_CHUNK_C * GDN_CHUNK_VC
        + GDN_CHUNK_C * GDN_CHUNK_C
        + 4 * GDN_CHUNK_C)
        * std::mem::size_of::<f32>();
    let cfg = LaunchConfig {
        grid_dim: ((dv / GDN_CHUNK_VC) as u32, heads as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let (seq_i, heads_i, dk_i, dv_i, vc_i) = (
        seq as i32,
        heads as i32,
        dk as i32,
        dv as i32,
        GDN_CHUNK_VC as i32,
    );
    let (qp, _rq) = q.device_ptr(&gpu.stream);
    let (kp, _rk) = k.device_ptr(&gpu.stream);
    let (vp, _rv) = v.device_ptr(&gpu.stream);
    let (gp, _rg) = g.device_ptr(&gpu.stream);
    let (bp, _rb) = beta.device_ptr(&gpu.stream);
    let (sp, _rs) = state.device_ptr_mut(&gpu.stream);
    let (op, _ro) = out.device_ptr_mut(&gpu.stream);
    let prof_ptr = prof.map(|p| {
        let (pp, _rp) = p.device_ptr_mut(&gpu.stream);
        pp
    });
    let mut l = gpu.stream.launch_builder(f);
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
    if let Some(pp) = prof_ptr.as_ref() {
        l.arg(pp);
    }
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// Batched chunked GDN: one launch over `(dv/VC, heads, batch)`.
#[allow(clippy::too_many_arguments)]
fn gdn_chunked_batched_impl<const C: usize, const VC: usize, Q, K, V, G, B, ST, O, OF, LN>(
    gpu: &Gpu,
    f: &CudaFunction,
    q: &Q,
    k: &K,
    v: &V,
    g: &G,
    beta: &B,
    state: &mut ST,
    out: &mut O,
    offsets: &OF,
    lens: &LN,
    batch: usize,
    heads: usize,
    dk: usize,
    dv: usize,
) -> Result<()>
where
    Q: DevicePtr<f32>,
    K: DevicePtr<f32>,
    V: DevicePtr<f32>,
    G: DevicePtr<f32>,
    B: DevicePtr<f32>,
    ST: DevicePtrMut<f32>,
    O: DevicePtrMut<bf16>,
    OF: DevicePtr<i32>,
    LN: DevicePtr<i32>,
{
    if dk != 128 || dv != 128 {
        anyhow::bail!("chunked GDN kernel requires dk=dv=128, got dk={dk} dv={dv}");
    }
    let smem = (dk * VC + 2 * C * dk + 2 * C * VC + C * C + 4 * C) * std::mem::size_of::<f32>();
    let cfg = LaunchConfig {
        grid_dim: ((dv / VC) as u32, heads as u32, batch as u32),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let (batch_i, heads_i, dk_i, dv_i, vc_i) =
        (batch as i32, heads as i32, dk as i32, dv as i32, VC as i32);
    let (qp, _rq) = q.device_ptr(&gpu.stream);
    let (kp, _rk) = k.device_ptr(&gpu.stream);
    let (vp, _rv) = v.device_ptr(&gpu.stream);
    let (gp, _rg) = g.device_ptr(&gpu.stream);
    let (bp, _rb) = beta.device_ptr(&gpu.stream);
    let (sp, _rs) = state.device_ptr_mut(&gpu.stream);
    let (op, _ro) = out.device_ptr_mut(&gpu.stream);
    let (ofp, _rof) = offsets.device_ptr(&gpu.stream);
    let (lnp, _rln) = lens.device_ptr(&gpu.stream);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(&qp);
    l.arg(&kp);
    l.arg(&vp);
    l.arg(&gp);
    l.arg(&bp);
    l.arg(&sp);
    l.arg(&op);
    l.arg(&ofp);
    l.arg(&lnp);
    l.arg(&batch_i);
    l.arg(&heads_i);
    l.arg(&dk_i);
    l.arg(&dv_i);
    l.arg(&vc_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_chunked_batched<Q, K, V, G, B, ST, O, OF, LN>(
    gpu: &Gpu,
    f: &CudaFunction,
    q: &Q,
    k: &K,
    v: &V,
    g: &G,
    beta: &B,
    state: &mut ST,
    out: &mut O,
    offsets: &OF,
    lens: &LN,
    batch: usize,
    heads: usize,
    dk: usize,
    dv: usize,
) -> Result<()>
where
    Q: DevicePtr<f32>,
    K: DevicePtr<f32>,
    V: DevicePtr<f32>,
    G: DevicePtr<f32>,
    B: DevicePtr<f32>,
    ST: DevicePtrMut<f32>,
    O: DevicePtrMut<bf16>,
    OF: DevicePtr<i32>,
    LN: DevicePtr<i32>,
{
    gdn_chunked_batched_impl::<{ GDN_CHUNK_C }, 64, Q, K, V, G, B, ST, O, OF, LN>(
        gpu, f, q, k, v, g, beta, state, out, offsets, lens, batch, heads, dk, dv,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_reg_split_batched(
    gpu: &Gpu,
    f: &CudaFunction,
    qf: &CudaSlice<f32>,
    kf: &CudaSlice<f32>,
    vf: &CudaSlice<f32>,
    g: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    offsets: &CudaSlice<i32>,
    lens: &CudaSlice<i32>,
    batch: usize,
    heads: usize,
    dk: usize,
    dv: usize,
) -> Result<()> {
    let vc = 32.min(dv);
    if dk != 128 || dv % vc != 0 || vc * 4 > 1024 {
        anyhow::bail!("register GDN requires dk=128 and dv divisible by vc={vc}");
    }
    let chunks = dv / vc;
    let cfg = LaunchConfig {
        grid_dim: ((heads * chunks) as u32, batch as u32, 1),
        block_dim: ((vc * 4) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (heads_i, dk_i, dv_i, vc_i, ch_i) =
        (heads as i32, dk as i32, dv as i32, vc as i32, chunks as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(qf);
    l.arg(kf);
    l.arg(vf);
    l.arg(g);
    l.arg(beta);
    l.arg(state);
    l.arg(out);
    l.arg(offsets);
    l.arg(lens);
    l.arg(&heads_i);
    l.arg(&dk_i);
    l.arg(&dv_i);
    l.arg(&vc_i);
    l.arg(&ch_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_split_batched(
    gpu: &Gpu,
    f: &CudaFunction,
    qf: &CudaSlice<f32>,
    kf: &CudaSlice<f32>,
    vf: &CudaSlice<f32>,
    g: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    offsets: &CudaSlice<i32>,
    lens: &CudaSlice<i32>,
    batch: usize,
    heads: usize,
    dk: usize,
    dv: usize,
) -> Result<()> {
    let vc = dv.min(32);
    if dv % vc != 0 {
        anyhow::bail!("dv={dv} not divisible by vc={vc}");
    }
    let chunks = dv / vc;
    let smem = dk * (vc + 1) * 4;
    let cfg = LaunchConfig {
        grid_dim: ((heads * chunks) as u32, batch as u32, 1),
        block_dim: ((vc * 4) as u32, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let (heads_i, dk_i, dv_i, vc_i, ch_i) =
        (heads as i32, dk as i32, dv as i32, vc as i32, chunks as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(qf);
    l.arg(kf);
    l.arg(vf);
    l.arg(g);
    l.arg(beta);
    l.arg(state);
    l.arg(out);
    l.arg(offsets);
    l.arg(lens);
    l.arg(&heads_i);
    l.arg(&dk_i);
    l.arg(&dv_i);
    l.arg(&vc_i);
    l.arg(&ch_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_shared_batched(
    gpu: &Gpu,
    f: &CudaFunction,
    qf: &CudaSlice<f32>,
    kf: &CudaSlice<f32>,
    vf: &CudaSlice<f32>,
    g: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    offsets: &CudaSlice<i32>,
    lens: &CudaSlice<i32>,
    batch: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    vchunks: usize,
) -> Result<()> {
    if dv % vchunks != 0 {
        anyhow::bail!("dv={dv} not divisible by gdn chunks={vchunks}");
    }
    let vc = dv / vchunks;
    let smem = dk * vc * 4;
    let cfg = LaunchConfig {
        grid_dim: ((heads * vchunks) as u32, batch as u32, 1),
        block_dim: (vc as u32, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let (heads_i, dk_i, dv_i, vc_i, vch_i) = (
        heads as i32,
        dk as i32,
        dv as i32,
        vc as i32,
        vchunks as i32,
    );
    let mut l = gpu.stream.launch_builder(f);
    l.arg(qf);
    l.arg(kf);
    l.arg(vf);
    l.arg(g);
    l.arg(beta);
    l.arg(state);
    l.arg(out);
    l.arg(offsets);
    l.arg(lens);
    l.arg(&heads_i);
    l.arg(&dk_i);
    l.arg(&dv_i);
    l.arg(&vc_i);
    l.arg(&vch_i);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

pub fn gather_rows(
    gpu: &Gpu,
    f: &CudaFunction,
    src: &CudaSlice<bf16>,
    rows: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    n_rows: usize,
    cols: usize,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows as u32, 16, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nr, c) = (n_rows as i32, cols as i32);
    let mut l = gpu.stream.launch_builder(f);
    l.arg(src);
    l.arg(rows);
    l.arg(out);
    l.arg(&nr);
    l.arg(&c);
    unsafe { l.launch(cfg) }?;
    Ok(())
}

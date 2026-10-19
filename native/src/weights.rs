//! Weight sources.
//!
//! The forward graph is written against `LinearWeight` / `WeightProvider`, so a
//! checkpoint can be served either from merged BF16 tensors (graph validation)
//! or from the AWQ W4A16 packed tensors (shipped path) without changing the
//! graph. Norm weights always come from the same checkpoint as the linears:
//! AWQ checkpoints are smoothing-adjusted, so mixing them with merged-bf16
//! norms would be wrong.
use anyhow::{Context, Result, bail};
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;

use crate::tensor::Store;

pub enum LinearWeight {
    /// Merged (unquantized) BF16 weight, row-major [out, in].
    Bf16 {
        out: usize,
        in_: usize,
        w: CudaSlice<bf16>,
    },
    /// Marlin W4A16 layout (repacked), asymmetric group 64.
    Marlin {
        out: usize,
        in_: usize,
        qweight: CudaSlice<i32>,
        scales: CudaSlice<bf16>,
        zeros: CudaSlice<i32>,
    },
    /// compressed-tensors pack-quantized W4A16 (AWQ/GPTQ), asymmetric int4.
    /// `packed` is I32 [out, ceil(in/8)] with 8 signed nibbles per word along
    /// the input dim; `scale` is BF16 [out, in/64]; `zp` is I32
    /// [ceil(out/8), in/64] with 8 zero-point nibbles per word along the output
    /// dim. The paired RMSNorm weights already contain the AWQ smoothing, so
    /// this variant must be used with the same checkpoint's norms.
    PackedW4A16 {
        out: usize,
        in_: usize,
        group_size: usize,
        packed: CudaSlice<i32>,
        scale: CudaSlice<bf16>,
        zp: CudaSlice<i32>,
        #[allow(dead_code)]
        shape: CudaSlice<i64>,
    },
}

impl LinearWeight {
    pub fn out(&self) -> usize {
        match self {
            LinearWeight::Bf16 { out, .. }
            | LinearWeight::PackedW4A16 { out, .. }
            | LinearWeight::Marlin { out, .. } => *out,
        }
    }
    pub fn in_(&self) -> usize {
        match self {
            LinearWeight::Bf16 { in_, .. }
            | LinearWeight::PackedW4A16 { in_, .. }
            | LinearWeight::Marlin { in_, .. } => *in_,
        }
    }
}

pub struct LayerWeights {
    pub input_norm: CudaSlice<bf16>,
    pub post_norm: CudaSlice<bf16>,
    pub gate_proj: LinearWeight,
    pub up_proj: LinearWeight,
    pub down_proj: LinearWeight,
    pub mixer: MixerWeights,
}

pub enum MixerWeights {
    Full(Box<FullAttentionWeights>),
    Gdn(Box<GdnWeights>),
}

pub struct FullAttentionWeights {
    pub q_proj: LinearWeight,
    pub k_proj: LinearWeight,
    pub v_proj: LinearWeight,
    pub o_proj: LinearWeight,
    pub q_norm: CudaSlice<bf16>,
    pub k_norm: CudaSlice<bf16>,
}

pub struct GdnWeights {
    pub in_proj_qkv: LinearWeight,
    pub in_proj_z: LinearWeight,
    pub in_proj_b: LinearWeight,
    pub in_proj_a: LinearWeight,
    pub out_proj: LinearWeight,
    pub conv1d: CudaSlice<bf16>,
    /// Kept in f32: the GDN decay is computed in f32 (`-exp(A_log) * softplus(a + dt_bias)`).
    pub a_log: CudaSlice<f32>,
    pub dt_bias: CudaSlice<f32>,
    pub norm: CudaSlice<bf16>,
}

pub struct WeightProvider<'a> {
    pub store: &'a Store,
    pub prefix: String,
    pub marlin: bool,
    /// FP8-activation Marlin repack (8-bit A layout; default off).
    pub marlin_fp8: bool,
    /// INT8-activation Marlin repack (raw AWQ qweight, 8-bit A layout).
    pub marlin_int8: bool,
    stream: Arc<CudaStream>,
    gpu: &'a crate::cuda::Gpu,
    kernels: &'a crate::kernels::Kernels,
}

impl<'a> WeightProvider<'a> {
    pub fn new(
        store: &'a Store,
        gpu: &'a crate::cuda::Gpu,
        kernels: &'a crate::kernels::Kernels,
    ) -> Result<Self> {
        let prefixes = ["model.language_model.", "model.", ""];
        let mut prefix = None;
        for p in prefixes {
            if store.contains(&format!("{p}embed_tokens.weight")) {
                prefix = Some(p.to_string());
                break;
            }
        }
        let prefix = prefix.context("could not locate embed_tokens.weight in checkpoint")?;
        Ok(Self {
            store,
            prefix,
            marlin: false,
            marlin_fp8: false,
            marlin_int8: false,
            stream: gpu.stream.clone(),
            gpu,
            kernels,
        })
    }

    pub fn name(&self, rest: &str) -> String {
        format!("{}{rest}", self.prefix)
    }

    fn upload(&self, name: &str) -> Result<CudaSlice<bf16>> {
        let host = self.store.bf16(name)?;
        self.stream
            .clone_htod(&host)
            .with_context(|| format!("uploading {name}"))
    }

    /// f32 vector; accepts F32 or BF16 checkpoint storage (both model variants
    /// store A_log/dt_bias as BF16 today, but the graph must not care).
    pub fn tensor_f32(&self, name: &str) -> Result<CudaSlice<f32>> {
        let host = self.store.f32(name)?;
        self.stream
            .clone_htod(&host)
            .with_context(|| format!("uploading {name} as f32"))
    }

    /// Load a `torch.nn.Linear` module by name (without the `.weight` suffix).
    /// Packed compressed-tensors modules are detected by the presence of
    /// `<module>.weight_packed`; everything else is merged BF16.
    pub fn linear(&self, module: &str) -> Result<LinearWeight> {
        let packed_name = format!("{module}.weight_packed");
        if self.store.contains(&packed_name) {
            let shape = self.store.i64(&format!("{module}.weight_shape"))?;
            if shape.len() != 2 {
                bail!("{module}.weight_shape: expected 2 entries");
            }
            let (out, in_) = (shape[0] as usize, shape[1] as usize);
            let scale_info = self
                .store
                .info(&format!("{module}.weight_scale"))
                .with_context(|| format!("{module}.weight_scale missing"))?;
            let groups = *scale_info.shape.last().context("scale shape")?;
            if groups == 0 || in_ % groups != 0 {
                bail!("{module}: in={in_} not divisible by groups={groups}");
            }
            let group_size = in_ / groups;
            if group_size != 64 {
                bail!("{module}: unsupported group size {group_size}");
            }
            let packed_words = in_.div_ceil(8);
            let zp_rows = out.div_ceil(8);
            let packed = self.store.i32(&packed_name)?;
            let scale = self.store.bf16(&format!("{module}.weight_scale"))?;
            let zp = self.store.i32(&format!("{module}.weight_zero_point"))?;
            let shape_dev = self.store.i64(&format!("{module}.weight_shape"))?;
            if packed.len() != out * packed_words {
                bail!(
                    "{module}: packed has {} words, expected {}",
                    packed.len(),
                    out * packed_words
                );
            }
            if scale.len() != out * groups || zp.len() != zp_rows * groups {
                bail!("{module}: scale/zp size mismatch");
            }
            if self.marlin && out % 64 == 0 && in_ % 64 == 0 {
                if (self.marlin_fp8 || self.marlin_int8) && group_size != 64 {
                    bail!("W4A8 transform requires group_size 64, got {group_size}");
                }
                let dbg = std::env::var("JEV_MARLIN_DEBUG").is_ok();
                if dbg {
                    eprintln!("marlin repack {module}: out={out} in={in_}");
                }
                let sms = 128usize;
                let src = self.stream.clone_htod(&packed)?;
                let mut gptq = self.stream.alloc_zeros::<i32>(out * packed_words)?;
                if self.marlin_fp8 {
                    let zp_dev = self.stream.clone_htod(&zp)?;
                    crate::kernels::awq_fp8_transform(
                        self.gpu,
                        &self.kernels.awq_fp8_transform,
                        &src,
                        &zp_dev,
                        &mut gptq,
                        in_,
                        out,
                        groups,
                    )?;
                } else {
                    crate::kernels::marlin_transpose(
                        self.gpu,
                        &self.kernels.marlin_transpose,
                        &src,
                        &mut gptq,
                        out,
                        packed_words,
                    )?;
                }
                if dbg {
                    self.stream.synchronize()?;
                    eprintln!("  transpose ok");
                }
                let mut qweight = self.stream.alloc_zeros::<i32>((in_ / 16) * (out * 2))?;
                let repack_kernel = if self.marlin_fp8 || self.marlin_int8 {
                    &self.kernels.marlin_repack_a8
                } else {
                    &self.kernels.marlin_repack
                };
                crate::kernels::marlin_repack(
                    self.gpu,
                    repack_kernel,
                    &gptq,
                    &mut qweight,
                    in_,
                    out,
                    sms,
                )?;
                let scales = if self.marlin_fp8 {
                    marlin_permute_scales_a8(&scale, out, groups, 512.0)
                } else if self.marlin_int8 {
                    marlin_permute_scales_a8(&scale, out, groups, 1.0)
                } else {
                    marlin_permute_scales(&scale, out, groups, in_ / group_size)
                };
                // The transpose/repack temporaries are freed when this branch
                // returns; synchronize so the async kernels cannot race with
                // memory reuse by the next module.
                self.stream.synchronize()?;
                if dbg {
                    eprintln!("  repack ok");
                }
                let zeros = if self.marlin_fp8 || self.marlin_int8 {
                    marlin_permute_zeros_a8(&zp, out, groups)
                } else {
                    marlin_permute_zeros(&zp, out, groups)
                };
                return Ok(LinearWeight::Marlin {
                    out,
                    in_,
                    qweight,
                    scales: self.stream.clone_htod(&scales)?,
                    zeros: self.stream.clone_htod(&zeros)?,
                });
            }
            return Ok(LinearWeight::PackedW4A16 {
                out,
                in_,
                group_size,
                packed: self.stream.clone_htod(&packed)?,
                scale: self.stream.clone_htod(&scale)?,
                zp: self.stream.clone_htod(&zp)?,
                shape: self.stream.clone_htod(&shape_dev)?,
            });
        }
        let wname = format!("{module}.weight");
        let info = self
            .store
            .info(&wname)
            .with_context(|| format!("linear {wname} not found"))?;
        if info.shape.len() != 2 {
            bail!("{wname}: expected 2-D weight, got {:?}", info.shape);
        }
        let out = info.shape[0];
        let in_ = info.shape[1];
        Ok(LinearWeight::Bf16 {
            out,
            in_,
            w: self.upload(&wname)?,
        })
    }

    pub fn layer(&self, idx: usize, kind: crate::config::LayerType) -> Result<LayerWeights> {
        let l = format!("layers.{idx}.");
        let input_norm = self.upload(&self.name(&format!("{l}input_layernorm.weight")))?;
        let post_norm = self.upload(&self.name(&format!("{l}post_attention_layernorm.weight")))?;
        let gate_proj = self.linear(&self.name(&format!("{l}mlp.gate_proj")))?;
        let up_proj = self.linear(&self.name(&format!("{l}mlp.up_proj")))?;
        let down_proj = self.linear(&self.name(&format!("{l}mlp.down_proj")))?;
        let mixer = match kind {
            crate::config::LayerType::FullAttention => {
                MixerWeights::Full(Box::new(FullAttentionWeights {
                    q_proj: self.linear(&self.name(&format!("{l}self_attn.q_proj")))?,
                    k_proj: self.linear(&self.name(&format!("{l}self_attn.k_proj")))?,
                    v_proj: self.linear(&self.name(&format!("{l}self_attn.v_proj")))?,
                    o_proj: self.linear(&self.name(&format!("{l}self_attn.o_proj")))?,
                    q_norm: self.upload(&self.name(&format!("{l}self_attn.q_norm.weight")))?,
                    k_norm: self.upload(&self.name(&format!("{l}self_attn.k_norm.weight")))?,
                }))
            }
            crate::config::LayerType::LinearAttention => MixerWeights::Gdn(Box::new(GdnWeights {
                in_proj_qkv: self.linear(&self.name(&format!("{l}linear_attn.in_proj_qkv")))?,
                in_proj_z: self.linear(&self.name(&format!("{l}linear_attn.in_proj_z")))?,
                in_proj_b: self.linear(&self.name(&format!("{l}linear_attn.in_proj_b")))?,
                in_proj_a: self.linear(&self.name(&format!("{l}linear_attn.in_proj_a")))?,
                out_proj: self.linear(&self.name(&format!("{l}linear_attn.out_proj")))?,
                conv1d: self.upload(&self.name(&format!("{l}linear_attn.conv1d.weight")))?,
                a_log: self.tensor_f32(&self.name(&format!("{l}linear_attn.A_log")))?,
                dt_bias: self.tensor_f32(&self.name(&format!("{l}linear_attn.dt_bias")))?,
                norm: self.upload(&self.name(&format!("{l}linear_attn.norm.weight")))?,
            })),
        };
        Ok(LayerWeights {
            input_norm,
            post_norm,
            gate_proj,
            up_proj,
            down_proj,
            mixer,
        })
    }

    pub fn final_norm(&self) -> Result<CudaSlice<bf16>> {
        self.upload(&self.name("norm.weight"))
    }

    pub fn embedding(&self) -> String {
        self.name("embed_tokens.weight")
    }
}

/// vLLM `get_scale_perms()` for group-wise scales.
fn scale_perm() -> [usize; 64] {
    let mut p = [0usize; 64];
    let mut k = 0;
    for i in 0..8 {
        for j in 0..8 {
            p[k] = i + 8 * j;
            k += 1;
        }
    }
    p
}

/// [out, groups] CT scales -> vLLM Marlin scale layout [groups, out].
fn marlin_permute_scales(ct: &[bf16], out: usize, groups: usize, _group_size: usize) -> Vec<bf16> {
    let n = out * groups;
    let mut transposed = vec![bf16::ZERO; n];
    for o in 0..out {
        for g in 0..groups {
            transposed[g * out + o] = ct[o * groups + g];
        }
    }
    let perm = scale_perm();
    let mut outv = vec![bf16::ZERO; n];
    let blocks = n / 64;
    for b in 0..blocks {
        for i in 0..64 {
            outv[b * 64 + i] = transposed[b * 64 + perm[i]];
        }
    }
    outv
}

/// vLLM `get_scale_perms()` single permutation (32 entries), used when A is 8-bit.
fn scale_perm_single() -> [usize; 32] {
    let mut p = [0usize; 32];
    let mut k = 0;
    for i in 0..4 {
        for j in [0usize, 1, 8, 9, 16, 17, 24, 25] {
            p[k] = 2 * i + j;
            k += 1;
        }
    }
    p
}

/// [out, groups] CT scales -> Marlin a8 scale layout; vLLM uses the single
/// permutation (32-element blocks) whenever is_a_8bit is true.
fn marlin_permute_scales_a8(ct: &[bf16], out: usize, groups: usize, factor: f32) -> Vec<bf16> {
    let n = out * groups;
    let mut transposed = vec![bf16::ZERO; n];
    for o in 0..out {
        for g in 0..groups {
            transposed[g * out + o] = ct[o * groups + g];
        }
    }
    let perm = scale_perm_single();
    let mut outv = vec![bf16::ZERO; n];
    for b in 0..n / 32 {
        for i in 0..32 {
            outv[b * 32 + i] = bf16::from_f32(transposed[b * 32 + perm[i]].to_f32() * factor);
        }
    }
    outv
}

/// CT zero points [ceil(out/8), groups] -> Marlin packed zeros [groups, out/8].
fn marlin_permute_zeros(ct: &[i32], out: usize, groups: usize) -> Vec<i32> {
    // Unpack to unsigned nibbles [groups, out].
    let mut q = vec![0u8; groups * out];
    for g in 0..groups {
        for o in 0..out {
            let w = ct[(o / 8) * groups + g];
            q[g * out + o] = ((w >> (4 * (o % 8))) & 0xF) as u8;
        }
    }
    // Scale permutation over blocks of 64 along the flattened [groups, out].
    let perm = scale_perm();
    let mut p = vec![0u8; q.len()];
    for b in 0..q.len() / 64 {
        for i in 0..64 {
            p[b * 64 + i] = q[b * 64 + perm[i]];
        }
    }
    // Interleave groups of 8, then pack eight columns per int32.
    const INTER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    let mut r = vec![0u8; p.len()];
    for b in 0..p.len() / 8 {
        for i in 0..8 {
            r[b * 8 + i] = p[b * 8 + INTER[i]];
        }
    }
    let words = out / 8;
    let mut zp = vec![0i32; groups * words];
    for g in 0..groups {
        for w in 0..words {
            let mut v = 0u32;
            for k in 0..8 {
                v |= (r[g * out + w * 8 + k] as u32) << (4 * k);
            }
            zp[g * words + w] = v as i32;
        }
    }
    zp
}

/// a8 variant: vLLM `marlin_zero_points(..., is_a_8bit=True)` keeps the 64-way
/// scale permutation but skips the 8-way column interleave.
fn marlin_permute_zeros_a8(ct: &[i32], out: usize, groups: usize) -> Vec<i32> {
    let mut q = vec![0u8; groups * out];
    for g in 0..groups {
        for o in 0..out {
            let w = ct[(o / 8) * groups + g];
            q[g * out + o] = ((w >> (4 * (o % 8))) & 0xF) as u8;
        }
    }
    let perm = scale_perm();
    let mut p = vec![0u8; q.len()];
    for b in 0..q.len() / 64 {
        for i in 0..64 {
            p[b * 64 + i] = q[b * 64 + perm[i]];
        }
    }
    let words = out / 8;
    let mut zp = vec![0i32; groups * words];
    for g in 0..groups {
        for w in 0..words {
            let mut v = 0u32;
            for k in 0..8 {
                v |= (p[g * out + w * 8 + k] as u32) << (4 * k);
            }
            zp[g * words + w] = v as i32;
        }
    }
    zp
}

#[cfg(test)]
fn awq_transform_host(packed: &[i32], zp: &[i32], k: usize, n: usize, groups: usize) -> Vec<i32> {
    let words_k = k / 8;
    let mut out = vec![0i32; words_k * n];
    for kk in 0..words_k {
        for col in 0..n {
            let word = packed[col * words_k + kk] as u32;
            let zword = zp[(col / 8) * groups + (kk * 8) / 64] as u32;
            let z = (zword >> (4 * (col % 8))) & 0xF;
            let mut outw = 0u32;
            for j in 0..8 {
                let q = (word >> (4 * j)) & 0xF;
                let v = q as i32 - z as i32;
                let t = if v < 0 { 15 - q } else { v as u32 };
                outw |= t << (4 * j);
            }
            out[kk * n + col] = outw as i32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a8_scale_permutation_matches_vllm() {
        // out=64, groups=1, scale[o]=o. vLLM with is_a_8bit uses the 32-entry
        // "single" permutation on flattened [groups, out].
        let ct: Vec<bf16> = (0..64).map(|i| bf16::from_f32(i as f32)).collect();
        let got = marlin_permute_scales_a8(&ct, 64, 1, 512.0);
        let perm = [
            0usize, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21,
            28, 29, 6, 7, 14, 15, 22, 23, 30, 31,
        ];
        for b in 0..2 {
            for i in 0..32 {
                assert_eq!(
                    got[b * 32 + i].to_f32(),
                    ((b * 32 + perm[i]) as f32) * 512.0,
                    "block {b} index {i}"
                );
            }
        }
    }

    #[test]
    fn a8_scale_permutation_no_fp8_factor() {
        let ct: Vec<bf16> = (0..64).map(|i| bf16::from_f32(i as f32)).collect();
        let got = marlin_permute_scales_a8(&ct, 64, 1, 1.0);
        let perm = [
            0usize, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21,
            28, 29, 6, 7, 14, 15, 22, 23, 30, 31,
        ];
        for b in 0..2 {
            for i in 0..32 {
                assert_eq!(got[b * 32 + i].to_f32(), (b * 32 + perm[i]) as f32);
            }
        }
    }

    #[test]
    fn a8_zero_packing_matches_vllm() {
        // out=64, groups=1, unpacked zero point u[o] = o % 16.
        let mut ct = vec![0i32; 8];
        for b in 0..8 {
            let mut v = 0u32;
            for k in 0..8 {
                v |= (((b * 8 + k) % 16) as u32) << (4 * k);
            }
            ct[b] = v as i32;
        }
        let got = marlin_permute_zeros_a8(&ct, 64, 1);
        let want = [
            -2139062144i32,
            -1852730991,
            -1566399838,
            -1280068685,
            -993737532,
            -707406379,
            -421075226,
            -134744073,
        ];
        assert_eq!(got, want.to_vec());
    }

    #[test]
    fn awq_fp8_transform_matches_canonical_rule() {
        let k = 128usize; // groups = 2
        let n = 64usize;
        let groups = k / 64;
        let words_k = k / 8;
        // q[k,n] = (9*k + 5*n + 3) % 16; z[g,n] = (7*g + 11*n + 2) % 16.
        let mut q = vec![0u8; k * n];
        let mut z = vec![0u8; groups * n];
        for kk in 0..k {
            for col in 0..n {
                q[kk * n + col] = ((9 * kk + 5 * col + 3) % 16) as u8;
            }
        }
        for g in 0..groups {
            for col in 0..n {
                z[g * n + col] = ((7 * g + 11 * col + 2) % 16) as u8;
            }
        }
        // CT packed [N, K/8] and zp [N/8, groups].
        let mut packed = vec![0i32; n * words_k];
        for col in 0..n {
            for kk in 0..words_k {
                let mut w = 0u32;
                for j in 0..8 {
                    w |= (q[(kk * 8 + j) * n + col] as u32) << (4 * j);
                }
                packed[col * words_k + kk] = w as i32;
            }
        }
        let zp_words = n / 8;
        let mut zp = vec![0i32; zp_words * groups];
        for g in 0..groups {
            for col in 0..n {
                let w = &mut zp[(col / 8) * groups + g];
                *w |= (z[g * n + col] as i32) << (4 * (col % 8));
            }
        }
        let got = awq_transform_host(&packed, &zp, k, n, groups);
        for kk in 0..k {
            for col in 0..n {
                let word = got[(kk / 8) * n + col] as u32;
                let t = (word >> (4 * (kk % 8))) & 0xF;
                let orig = q[kk * n + col] as i32;
                let zz = z[(kk / 64) * n + col] as i32;
                let v = orig - zz;
                let want = if v < 0 { 15 - orig } else { v } as u32;
                assert_eq!(t, want, "k={kk} n={col}");
            }
        }
    }
}

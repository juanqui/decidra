//! Weight sources.
//!
//! The forward graph is written against `LinearWeight` / `WeightProvider`, so a
//! checkpoint can be served either from merged BF16 tensors (reference and
//! SM90 path) or from compressed-tensors W4A16 packed tensors (repacked to
//! Marlin W4A8 for the INT8 path; excluded modules stay W4A16) without changing
//! the graph. Norm weights always come from the same checkpoint as the linears:
//! packed checkpoints can be smoothing-adjusted, so mixing them with
//! merged-BF16 norms would be wrong.
use anyhow::{Context, Result, bail};
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;

use crate::tensor::Store;

pub struct Lora {
    /// A: [rank, in], BF16 row-major.
    pub a: CudaSlice<bf16>,
    /// B: [out, rank], BF16 row-major, pre-scaled by alpha/rank.
    pub b: CudaSlice<bf16>,
    pub rank: usize,
}

/// FP8 E4M3 view of a BF16 linear for SM120 large-M MLP GEMMs.
pub struct Fp8Weight {
    /// [out, in] row-major E4M3 bytes.
    pub w: CudaSlice<u8>,
    /// [out] dequantization scale.
    pub scale: CudaSlice<f32>,
}

/// Compressed-tensors NVFP4 weight: E2M1 pairs + per-16 UE4M3 scales in the
/// cuBLASLt 128x4 tiled layout.
pub struct NvFp4Weight {
    /// [out, in/2] packed E2M1 pairs.
    pub packed: CudaSlice<u8>,
    /// cuBLASLt VEC16 UE4M3 scale layout for A.
    pub scales: CudaSlice<u8>,
    pub out: usize,
    pub in_: usize,
    pub weight_global: f32,
    pub input_global: f32,
}

pub enum LinearWeight {
    /// Merged (unquantized) BF16 weight, row-major [out, in].
    Bf16 {
        out: usize,
        in_: usize,
        w: CudaSlice<bf16>,
        /// Optional SM120 FP8 image for large-M MLP GEMMs.
        fp8: Option<Fp8Weight>,
        lora: Option<Arc<Lora>>,
    },
    /// Marlin W4A16 layout (repacked), asymmetric or symmetric group 64/128.
    Marlin {
        out: usize,
        in_: usize,
        qweight: CudaSlice<i32>,
        scales: CudaSlice<bf16>,
        zeros: CudaSlice<i32>,
        /// INT8-A global scale folded into activation scales; 0 for W4A16/FP8.
        a_global: f32,
        /// Repacked for 8-bit activations (INT8 per-token quantized A).
        int8_a: bool,
        /// Use per-K-group (64) INT8 activation scales instead of one scale per
        /// row. Requires `int8_a` and 64-element weight groups.
        group_a: bool,
        /// Weight quantization group size (64 or 128).
        group_size: usize,
        lora: Option<Arc<Lora>>,
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
        lora: Option<Arc<Lora>>,
    },
    /// NVFP4 W4A4 checkpoint, consumed directly by the SM120 FP4 GEMM.
    NvFp4(NvFp4Weight),
}

impl LinearWeight {
    pub fn a_global(&self) -> f32 {
        match self {
            LinearWeight::Marlin { a_global, .. } => *a_global,
            _ => 0.0,
        }
    }

    /// True when this module was repacked for INT8 activation quantization.
    pub fn int8_a(&self) -> bool {
        match self {
            LinearWeight::Marlin { int8_a, .. } => *int8_a,
            _ => false,
        }
    }

    /// True when this module uses per-K-group INT8 activation scales.
    pub fn group_a(&self) -> bool {
        match self {
            LinearWeight::Marlin { group_a, .. } => *group_a,
            _ => false,
        }
    }

    pub fn lora(&self) -> Option<&Arc<Lora>> {
        match self {
            LinearWeight::Bf16 { lora, .. }
            | LinearWeight::Marlin { lora, .. }
            | LinearWeight::PackedW4A16 { lora, .. } => lora.as_ref(),
            LinearWeight::NvFp4(_) => None,
        }
    }

    /// Weight quantization group size (64 or 128); 0 for BF16.
    pub fn fp8(&self) -> Option<&Fp8Weight> {
        match self {
            LinearWeight::Bf16 { fp8, .. } => fp8.as_ref(),
            _ => None,
        }
    }

    pub fn group_size(&self) -> usize {
        match self {
            LinearWeight::Marlin { group_size, .. }
            | LinearWeight::PackedW4A16 { group_size, .. } => *group_size,
            LinearWeight::Bf16 { .. } | LinearWeight::NvFp4(_) => 0,
        }
    }

    pub fn out(&self) -> usize {
        match self {
            LinearWeight::Bf16 { out, .. }
            | LinearWeight::PackedW4A16 { out, .. }
            | LinearWeight::Marlin { out, .. } => *out,
            LinearWeight::NvFp4(w) => w.out,
        }
    }
    pub fn in_(&self) -> usize {
        match self {
            LinearWeight::Bf16 { in_, .. }
            | LinearWeight::PackedW4A16 { in_, .. }
            | LinearWeight::Marlin { in_, .. } => *in_,
            LinearWeight::NvFp4(w) => w.in_,
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
    /// Module-name substrings kept on W4A16 activations when `marlin_int8`.
    pub int8_exclude: Vec<String>,
    /// Use per-K-group (64) activation scales on INT8 modules by default.
    pub int8_group_scale: bool,
    /// Module-name substrings that keep per-token scales under
    /// `int8_group_scale` (they use the faster int32 epilogue).
    pub int8_group_exclude: Vec<String>,
    /// Layer indices that use all-linear INT8 (sorted), overriding the
    /// per-module exclusion list inside those layers.
    pub int8_all_layers: Vec<usize>,
    /// Build FP8 E4M3 images for MLP linears on SM120.
    pub sm120_fp8_mlp: bool,
    /// Optional LoRA adapter store (PEFT `base_model.model.<module>` keys).
    pub lora_store: Option<&'a Store>,
    /// LoRA scale, `lora_alpha / r`.
    pub lora_scale: f32,
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
        let prefixes = ["model.language_model.", "language_model.", "model.", ""];
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
            int8_exclude: Vec::new(),
            int8_group_scale: false,
            int8_group_exclude: Vec::new(),
            int8_all_layers: Vec::new(),
            sm120_fp8_mlp: false,
            lora_store: None,
            lora_scale: 1.0,
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
        let lora = self.lora_for(module)?;
        let packed_name = format!("{module}.weight_packed");
        if self
            .store
            .contains(&format!("{module}.weight_global_scale"))
        {
            return self.nvfp4_linear(module);
        }
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
            if group_size != 64 && group_size != 128 {
                bail!("{module}: unsupported group size {group_size}");
            }
            let packed_words = in_.div_ceil(8);
            let zp_rows = out.div_ceil(8);
            let packed = self.store.i32(&packed_name)?;
            let scale = self.store.bf16(&format!("{module}.weight_scale"))?;
            // Symmetric int4 checkpoints omit the zero-point tensor; the
            // compressed-tensors packing is offset by 8.
            let zp = if self.store.contains(&format!("{module}.weight_zero_point")) {
                self.store.i32(&format!("{module}.weight_zero_point"))?
            } else {
                vec![0x8888_8888u32 as i32; zp_rows * groups]
            };
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
            let layer = module
                .split_once("layers.")
                .and_then(|(_, rest)| rest.split('.').next())
                .and_then(|n| n.parse::<usize>().ok());
            let all_int8 = layer
                .map(|i| self.int8_all_layers.binary_search(&i).is_ok())
                .unwrap_or(false);
            let use_int8 = self.marlin_int8
                && (all_int8
                    || !self
                        .int8_exclude
                        .iter()
                        .any(|p| !p.is_empty() && module.contains(p)));
            let group_a = use_int8
                && group_size == 64
                && self.int8_group_scale
                && !self
                    .int8_group_exclude
                    .iter()
                    .any(|p| !p.is_empty() && module.contains(p));
            if self.marlin && out % 64 == 0 && in_ % 64 == 0 {
                if self.marlin_fp8 && group_size != 64 {
                    bail!("FP8 transform requires group_size 64, got {group_size}");
                }
                if use_int8 && group_size != 64 && group_size != 128 {
                    bail!("W4A8 transform requires group_size 64/128, got {group_size}");
                }
                let dbg = std::env::var("DECIDRA_MARLIN_DEBUG").is_ok();
                if dbg {
                    eprintln!("marlin repack {module}: out={out} in={in_}");
                }
                let sms = self.gpu.sms;
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
                let repack_kernel = if self.marlin_fp8 || use_int8 {
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
                let (scales, a_global) = if self.marlin_fp8 {
                    (marlin_permute_scales_a8(&scale, out, groups, 512.0), 0.0)
                } else if use_int8 {
                    marlin_process_int8_scales(&scale, out, groups)
                } else {
                    (
                        marlin_permute_scales(&scale, out, groups, in_ / group_size),
                        0.0,
                    )
                };
                // The transpose/repack temporaries are freed when this branch
                // returns; synchronize so the async kernels cannot race with
                // memory reuse by the next module.
                self.stream.synchronize()?;
                if dbg {
                    eprintln!("  repack ok");
                }
                let zeros = if self.marlin_fp8 || use_int8 {
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
                    a_global,
                    int8_a: use_int8,
                    group_a,
                    group_size,
                    lora: lora.clone(),
                });
            }
            return Ok(LinearWeight::PackedW4A16 {
                out,
                in_,
                group_size,
                packed: self.stream.clone_htod(&packed)?,
                scale: self.stream.clone_htod(&scale)?,
                zp: self.stream.clone_htod(&zp)?,
                lora: lora.clone(),
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
        let w = self.upload(&wname)?;
        let fp8_mode = std::env::var("DECIDRA_FP8_MODE").unwrap_or_else(|_| "mlp".to_owned());
        let fp8_selected = if fp8_mode == "all_linear" {
            true
        } else if fp8_mode == "gate_up" {
            module.contains("mlp.gate_proj") || module.contains("mlp.up_proj")
        } else {
            module.contains("mlp.gate_proj")
                || module.contains("mlp.up_proj")
                || module.contains("mlp.down_proj")
        };
        let fp8 = if self.sm120_fp8_mlp && lora.is_none() && fp8_selected {
            let mut q = self.stream.alloc_zeros::<u8>(out * in_)?;
            let mut scales = self.stream.alloc_zeros::<f32>(out)?;
            crate::kernels::quant_fp8_bf16(
                self.gpu,
                &self.kernels.quant_fp8_bf16,
                &w,
                &mut q,
                &mut scales,
                out,
                in_,
                1.0,
            )?;
            self.stream.synchronize()?;
            Some(Fp8Weight {
                w: q,
                scale: scales,
            })
        } else {
            None
        };
        Ok(LinearWeight::Bf16 {
            out,
            in_,
            w,
            fp8,
            lora,
        })
    }

    fn nvfp4_linear(&self, module: &str) -> Result<LinearWeight> {
        let packed_info = self
            .store
            .info(&format!("{module}.weight_packed"))
            .with_context(|| format!("{module}.weight_packed missing"))?;
        if packed_info.shape.len() != 2 {
            bail!("{module}.weight_packed must be 2-D");
        }
        let out = packed_info.shape[0];
        let in_ = packed_info.shape[1] * 2;
        if in_ % 64 != 0 {
            bail!("{module}: K={in_} is not a multiple of 64");
        }
        let scale_info = self
            .store
            .info(&format!("{module}.weight_scale"))
            .with_context(|| format!("{module}.weight_scale missing"))?;
        if scale_info.shape != vec![out, in_ / 16] {
            bail!(
                "{module}: weight_scale shape {:?}, expected [{out}, {}]",
                scale_info.shape,
                in_ / 16
            );
        }
        let packed = self.store.u8(&format!("{module}.weight_packed"))?;
        let scales_raw = self.store.u8(&format!("{module}.weight_scale"))?;
        let scales = repack_nvfp4_scales(&scales_raw, out, in_);
        let weight_global = *self
            .store
            .f32(&format!("{module}.weight_global_scale"))?
            .first()
            .context("weight_global_scale empty")?;
        let input_global = *self
            .store
            .f32(&format!("{module}.input_global_scale"))?
            .first()
            .context("input_global_scale empty")?;
        Ok(LinearWeight::NvFp4(NvFp4Weight {
            packed: self.stream.clone_htod(&packed)?,
            scales: self.stream.clone_htod(&scales)?,
            out,
            in_,
            weight_global,
            input_global,
        }))
    }

    /// Load a PEFT LoRA pair for `module` and upload it (B pre-scaled).
    fn lora_for(&self, module: &str) -> Result<Option<Arc<Lora>>> {
        let Some(store) = self.lora_store else {
            return Ok(None);
        };
        let a_name = format!("base_model.model.{module}.lora_A.weight");
        if !store.contains(&a_name) {
            return Ok(None);
        }
        let b_name = format!("base_model.model.{module}.lora_B.weight");
        let a_info = store
            .info(&a_name)
            .context("lora_A info")
            .map(|i| i.shape.clone())?;
        let b_info = store
            .info(&b_name)
            .context("lora_B info")
            .map(|i| i.shape.clone())?;
        if a_info.len() != 2 || b_info.len() != 2 {
            bail!("{module}: LoRA tensors must be 2-D");
        }
        let rank = a_info[0];
        let mut b = store.bf16(&b_name)?;
        if self.lora_scale != 1.0 {
            for v in &mut b {
                *v = bf16::from_f32(v.to_f32() * self.lora_scale);
            }
        }
        Ok(Some(Arc::new(Lora {
            a: self.stream.clone_htod(&store.bf16(&a_name)?)?,
            b: self.stream.clone_htod(&b)?,
            rank,
        })))
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

/// Compressed-tensors `[outer, inner/16]` row-major scales -> cuBLASLt
/// VEC16 UE4M3 128x4 tiled layout.
fn repack_nvfp4_scales(src: &[u8], outer: usize, inner: usize) -> Vec<u8> {
    let inner_blocks = inner / 64;
    let outer_blocks = outer.div_ceil(128);
    let mut dst = vec![0u8; inner_blocks * outer_blocks * 512];
    for o in 0..outer {
        for j in 0..inner / 16 {
            let base = ((j / 4) + (o / 128) * inner_blocks) * 512;
            let off = base + (o % 32) * 16 + ((o % 128) / 32) * 4 + (j % 4);
            dst[off] = src[o * (inner / 16) + j];
        }
    }
    dst
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

/// `marlin_act_int8_process_scales`: normalize bf16 weight scales to int16
/// bit patterns in [0,4096] and return `max(scale)/4096`; INT8 activation
/// scales must be multiplied by that global scale.
fn marlin_process_int8_scales(ct: &[bf16], out: usize, groups: usize) -> (Vec<bf16>, f32) {
    let raw = marlin_permute_scales_a8(ct, out, groups, 1.0);
    let max = raw.iter().map(|v| v.to_f32()).fold(0.0f32, f32::max);
    if max <= 0.0 {
        return (raw, 0.0);
    }
    let factor = max / 4096.0;
    let proc = raw
        .iter()
        .map(|v| {
            let q = (v.to_f32() / max * 4096.0).round() as i32;
            bf16::from_bits(q.clamp(0, 65535) as u16)
        })
        .collect();
    (proc, factor)
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
    fn nvfp4_scale_repack_uses_cublaslt_tiles() {
        // outer=3, inner=128: two 64-wide inner tiles, one 128-wide outer tile.
        let outer = 3usize;
        let inner = 128usize;
        let src: Vec<u8> = (0..outer * inner / 16).map(|i| i as u8).collect();
        let got = repack_nvfp4_scales(&src, outer, inner);
        assert_eq!(got.len(), (inner / 64) * 1 * 512);
        for o in 0..outer {
            for j in 0..inner / 16 {
                let base = ((j / 4) + (o / 128) * (inner / 64)) * 512;
                let off = base + (o % 32) * 16 + ((o % 128) / 32) * 4 + (j % 4);
                assert_eq!(got[off], src[o * (inner / 16) + j], "o={o} j={j}");
            }
        }
    }

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

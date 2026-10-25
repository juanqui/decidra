//! Qwen3.5 vision tower for d3-turbo image states.
//!
//! Weights are the checkpoint's BF16 `visual.*` tensors. The tower runs the
//! patch projection, learned 48x48 position interpolation, 27 transformer
//! blocks with axial rotary vision attention, and the 2x2 patch merger that
//! emits 5120-wide image token features for the language model.

use anyhow::{Context, Result, bail};
use cudarc::driver::CudaSlice;
use half::bf16;

use crate::engine::{Buf, Engine};
use crate::kernels;
use crate::tensor::Store;

pub const HIDDEN: usize = 1152;
pub const DEPTH: usize = 27;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 72;
pub const HEAD_PAD: usize = 128;
pub const INTERMEDIATE: usize = 4304;
pub const OUT_HIDDEN: usize = 5120;
pub const PATCH_ELEMS: usize = 3 * 2 * 16 * 16;
pub const POS_SIDE: usize = 48;

pub struct VisionBlock {
    norm1_w: CudaSlice<bf16>,
    norm1_b: CudaSlice<bf16>,
    qkv_w: CudaSlice<bf16>,
    qkv_b: CudaSlice<bf16>,
    proj_w: CudaSlice<bf16>,
    proj_b: CudaSlice<bf16>,
    norm2_w: CudaSlice<bf16>,
    norm2_b: CudaSlice<bf16>,
    fc1_w: CudaSlice<bf16>,
    fc1_b: CudaSlice<bf16>,
    fc2_w: CudaSlice<bf16>,
    fc2_b: CudaSlice<bf16>,
}

pub struct VisionModel {
    patch_w: CudaSlice<bf16>,
    patch_b: CudaSlice<bf16>,
    pos: Vec<bf16>,
    blocks: Vec<VisionBlock>,
    merger_norm_w: CudaSlice<bf16>,
    merger_norm_b: CudaSlice<bf16>,
    merger_fc1: CudaSlice<bf16>,
    merger_fc1_b: CudaSlice<bf16>,
    merger_fc2: CudaSlice<bf16>,
    merger_fc2_b: CudaSlice<bf16>,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionGrid {
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

#[derive(Clone, Debug)]
pub struct VisionInput<'a> {
    pub patches: &'a [bf16],
    pub grid: VisionGrid,
}

pub struct VisionOutput {
    pub features: Buf,
    pub tokens: usize,
    pub grids: Vec<VisionGrid>,
}

impl VisionModel {
    pub fn load(store: &Store, gpu: &crate::cuda::Gpu) -> Result<Self> {
        let w = |name: &str, shape: &[usize]| -> Result<CudaSlice<bf16>> {
            let info = store
                .info(name)
                .with_context(|| format!("vision tensor {name} missing"))?;
            if info.shape != shape {
                bail!("{name}: shape {:?}, expected {shape:?}", info.shape);
            }
            let host = store
                .bf16(name)
                .with_context(|| format!("loading {name}"))?;
            Ok(gpu.stream.clone_htod(&host)?)
        };
        let mut blocks = Vec::with_capacity(DEPTH);
        for i in 0..DEPTH {
            let p = format!("visual.blocks.{i}");
            blocks.push(VisionBlock {
                norm1_w: w(&format!("{p}.norm1.weight"), &[HIDDEN])?,
                norm1_b: w(&format!("{p}.norm1.bias"), &[HIDDEN])?,
                qkv_w: w(&format!("{p}.attn.qkv.weight"), &[3 * HIDDEN, HIDDEN])?,
                qkv_b: w(&format!("{p}.attn.qkv.bias"), &[3 * HIDDEN])?,
                proj_w: w(&format!("{p}.attn.proj.weight"), &[HIDDEN, HIDDEN])?,
                proj_b: w(&format!("{p}.attn.proj.bias"), &[HIDDEN])?,
                norm2_w: w(&format!("{p}.norm2.weight"), &[HIDDEN])?,
                norm2_b: w(&format!("{p}.norm2.bias"), &[HIDDEN])?,
                fc1_w: w(
                    &format!("{p}.mlp.linear_fc1.weight"),
                    &[INTERMEDIATE, HIDDEN],
                )?,
                fc1_b: w(&format!("{p}.mlp.linear_fc1.bias"), &[INTERMEDIATE])?,
                fc2_w: w(
                    &format!("{p}.mlp.linear_fc2.weight"),
                    &[HIDDEN, INTERMEDIATE],
                )?,
                fc2_b: w(&format!("{p}.mlp.linear_fc2.bias"), &[HIDDEN])?,
            });
        }
        let pos = store.bf16("visual.pos_embed.weight")?;
        if pos.len() != POS_SIDE * POS_SIDE * HIDDEN {
            bail!("visual.pos_embed.weight has {} values", pos.len());
        }
        Ok(Self {
            patch_w: w("visual.patch_embed.proj.weight", &[HIDDEN, 3, 2, 16, 16])?,
            patch_b: w("visual.patch_embed.proj.bias", &[HIDDEN])?,
            pos,
            blocks,
            merger_norm_w: w("visual.merger.norm.weight", &[HIDDEN])?,
            merger_norm_b: w("visual.merger.norm.bias", &[HIDDEN])?,
            merger_fc1: w("visual.merger.linear_fc1.weight", &[4 * HIDDEN, 4 * HIDDEN])?,
            merger_fc1_b: w("visual.merger.linear_fc1.bias", &[4 * HIDDEN])?,
            merger_fc2: w("visual.merger.linear_fc2.weight", &[OUT_HIDDEN, 4 * HIDDEN])?,
            merger_fc2_b: w("visual.merger.linear_fc2.bias", &[OUT_HIDDEN])?,
        })
    }
}

impl Engine {
    pub fn vision_encode(&self, inputs: &[VisionInput<'_>]) -> Result<VisionOutput> {
        let vision = self
            .vision
            .as_ref()
            .context("checkpoint has no vision tower")?;
        if inputs.is_empty() {
            bail!("no images");
        }
        let rows: usize = inputs.iter().map(|i| i.grid.grid_h * i.grid.grid_w).sum();
        if rows == 0 {
            bail!("empty image grids");
        }
        if rows % 4 != 0 {
            bail!("vision patch count {rows} is not divisible by four");
        }
        let mut xhost = Vec::with_capacity(rows * PATCH_ELEMS);
        for input in inputs {
            if input.patches.len() != input.grid.grid_h * input.grid.grid_w * PATCH_ELEMS {
                bail!(
                    "image patch buffer has {} values, expected {}",
                    input.patches.len(),
                    input.grid.grid_h * input.grid.grid_w * PATCH_ELEMS
                );
            }
            xhost.extend_from_slice(input.patches);
        }
        let mut x = self
            .alloc_bf16(rows * PATCH_ELEMS)
            .context("vision patch alloc")?;
        self.gpu
            .stream
            .memcpy_htod(&xhost, &mut *x)
            .context("vision patch upload")?;
        let mut x = self
            .blas_gemm(&x, rows, PATCH_ELEMS, &vision.patch_w, HIDDEN)
            .context("vision patch gemm")?;
        kernels::add_bias(
            &self.gpu,
            &self.kernels.add_bias_bf16,
            &mut x,
            &vision.patch_b,
            rows,
            HIDDEN,
        )?;
        let pos = self
            .vision_positions(inputs)
            .context("vision position embeddings")?;
        kernels::add_rows(
            &self.gpu,
            &self.kernels.add_rows_bf16,
            &mut x,
            &pos,
            rows * HIDDEN,
        )?;
        drop(pos);

        let mut cos_vec = Vec::with_capacity(rows * HEAD_DIM);
        let mut sin_vec = Vec::with_capacity(rows * HEAD_DIM);
        for input in inputs {
            append_vision_rope(input.grid, &mut cos_vec, &mut sin_vec);
        }
        let cos = self
            .gpu
            .stream
            .clone_htod(&cos_vec)
            .context("vision cos upload")?;
        let sin = self
            .gpu
            .stream
            .clone_htod(&sin_vec)
            .context("vision sin upload")?;
        for block in &vision.blocks {
            let mut norm = self
                .alloc_bf16(rows * HIDDEN)
                .context("vision norm alloc")?;
            kernels::layernorm(
                &self.gpu,
                &self.kernels.layernorm_bf16,
                &x,
                &block.norm1_w,
                &block.norm1_b,
                &mut norm,
                rows,
                HIDDEN,
                1e-6,
            )?;
            let mut qkv = self.blas_gemm(&norm, rows, HIDDEN, &block.qkv_w, 3 * HIDDEN)?;
            drop(norm);
            kernels::add_bias(
                &self.gpu,
                &self.kernels.add_bias_bf16,
                &mut qkv,
                &block.qkv_b,
                rows,
                3 * HIDDEN,
            )?;
            let mut q = self
                .gpu
                .stream
                .alloc_zeros::<bf16>(rows * HEADS * HEAD_PAD)?;
            let mut k = self
                .gpu
                .stream
                .alloc_zeros::<bf16>(rows * HEADS * HEAD_PAD)?;
            let mut v = self
                .gpu
                .stream
                .alloc_zeros::<bf16>(rows * HEADS * HEAD_PAD)?;
            kernels::vision_split_qkv(
                &self.gpu,
                &self.kernels.vision_split_qkv_bf16,
                &qkv,
                &mut q,
                &mut k,
                &mut v,
                rows,
                HEADS,
                HEAD_DIM,
                HEAD_PAD,
            )?;
            drop(qkv);
            kernels::vision_rope(
                &self.gpu,
                &self.kernels.vision_rope_qk_bf16,
                &mut q,
                &mut k,
                &cos,
                &sin,
                rows,
                HEADS,
                HEAD_DIM,
                HEAD_PAD,
                HEAD_DIM,
            )?;
            let mut attended = self
                .gpu
                .stream
                .alloc_zeros::<bf16>(rows * HEADS * HEAD_PAD)
                .context("vision attention alloc")?;
            let mut start = 0usize;
            for input in inputs {
                let n = input.grid.grid_h * input.grid.grid_w;
                let range = start * HEADS * HEAD_PAD..(start + n) * HEADS * HEAD_PAD;
                let qv = q.try_slice(range.clone()).context("vision q view")?;
                let kv = k.try_slice(range.clone()).context("vision k view")?;
                let vv = v.try_slice(range.clone()).context("vision v view")?;
                let mut ov = attended
                    .try_slice_mut(range)
                    .context("vision output view")?;
                kernels::attention_noncausal_128(
                    &self.gpu,
                    &self.kernels.attention_tc_nc_128,
                    &qv,
                    &kv,
                    &vv,
                    &mut ov,
                    n,
                    HEADS,
                    HEADS,
                    1.0 / (HEAD_DIM as f32).sqrt(),
                )?;
                start += n;
            }
            drop(q);
            drop(k);
            drop(v);
            let mut flat = self.alloc_bf16(rows * HIDDEN)?;
            kernels::vision_extract(
                &self.gpu,
                &self.kernels.vision_extract_bf16,
                &attended,
                &mut flat,
                rows,
                HEADS,
                HEAD_DIM,
                HEAD_PAD,
            )?;
            drop(attended);
            let mut attn = self.blas_gemm(&flat, rows, HIDDEN, &block.proj_w, HIDDEN)?;
            drop(flat);
            kernels::add_bias(
                &self.gpu,
                &self.kernels.add_bias_bf16,
                &mut attn,
                &block.proj_b,
                rows,
                HIDDEN,
            )?;
            let mut xnext = self.alloc_bf16(rows * HIDDEN)?;
            kernels::add(
                &self.gpu,
                &self.kernels.add,
                &x,
                &attn,
                &mut xnext,
                rows * HIDDEN,
            )?;
            drop(x);
            drop(attn);
            let mut norm = self
                .alloc_bf16(rows * HIDDEN)
                .context("vision norm alloc")?;
            kernels::layernorm(
                &self.gpu,
                &self.kernels.layernorm_bf16,
                &xnext,
                &block.norm2_w,
                &block.norm2_b,
                &mut norm,
                rows,
                HIDDEN,
                1e-6,
            )?;
            let mut hidden = self.blas_gemm(&norm, rows, HIDDEN, &block.fc1_w, INTERMEDIATE)?;
            drop(norm);
            kernels::add_bias(
                &self.gpu,
                &self.kernels.add_bias_bf16,
                &mut hidden,
                &block.fc1_b,
                rows,
                INTERMEDIATE,
            )?;
            kernels::gelu(
                &self.gpu,
                &self.kernels.gelu_tanh_bf16,
                &mut hidden,
                rows * INTERMEDIATE,
            )?;
            let mut mlp = self.blas_gemm(&hidden, rows, INTERMEDIATE, &block.fc2_w, HIDDEN)?;
            drop(hidden);
            kernels::add_bias(
                &self.gpu,
                &self.kernels.add_bias_bf16,
                &mut mlp,
                &block.fc2_b,
                rows,
                HIDDEN,
            )?;
            let mut next = self.alloc_bf16(rows * HIDDEN)?;
            kernels::add(
                &self.gpu,
                &self.kernels.add,
                &xnext,
                &mlp,
                &mut next,
                rows * HIDDEN,
            )?;
            drop(xnext);
            drop(mlp);
            x = next;
        }
        drop(cos);
        drop(sin);
        let mut normed = self
            .alloc_bf16(rows * HIDDEN)
            .context("vision merger norm alloc")?;
        kernels::layernorm(
            &self.gpu,
            &self.kernels.layernorm_bf16,
            &x,
            &vision.merger_norm_w,
            &vision.merger_norm_b,
            &mut normed,
            rows,
            HIDDEN,
            1e-6,
        )?;
        drop(x);
        let merged_rows = rows / 4;
        let mut merged = self
            .alloc_bf16(merged_rows * 4 * HIDDEN)
            .context("vision merger alloc")?;
        kernels::vision_merge(
            &self.gpu,
            &self.kernels.vision_merge_bf16,
            &normed,
            &mut merged,
            rows,
            HIDDEN,
        )?;
        drop(normed);
        let mut fc1 = self.blas_gemm(
            &merged,
            merged_rows,
            4 * HIDDEN,
            &vision.merger_fc1,
            4 * HIDDEN,
        )?;
        drop(merged);
        kernels::add_bias(
            &self.gpu,
            &self.kernels.add_bias_bf16,
            &mut fc1,
            &vision.merger_fc1_b,
            merged_rows,
            4 * HIDDEN,
        )?;
        kernels::gelu(
            &self.gpu,
            &self.kernels.gelu_exact_bf16,
            &mut fc1,
            merged_rows * 4 * HIDDEN,
        )?;
        let mut features = self.blas_gemm(
            &fc1,
            merged_rows,
            4 * HIDDEN,
            &vision.merger_fc2,
            OUT_HIDDEN,
        )?;
        kernels::add_bias(
            &self.gpu,
            &self.kernels.add_bias_bf16,
            &mut features,
            &vision.merger_fc2_b,
            merged_rows,
            OUT_HIDDEN,
        )?;
        drop(fc1);
        Ok(VisionOutput {
            tokens: merged_rows,
            features,
            grids: inputs.iter().map(|i| i.grid).collect(),
        })
    }

    fn vision_positions(&self, inputs: &[VisionInput<'_>]) -> Result<CudaSlice<bf16>> {
        let vision = self
            .vision
            .as_ref()
            .context("checkpoint has no vision tower")?;
        let total: usize = inputs.iter().map(|i| i.grid.grid_h * i.grid.grid_w).sum();
        let mut out = vec![bf16::ZERO; total * HIDDEN];
        let mut base = 0usize;
        for input in inputs {
            let g = input.grid;
            let bh = g.grid_h / 2;
            let bw = g.grid_w / 2;
            let hs = if g.grid_h > 1 {
                (POS_SIDE - 1) as f32 / (g.grid_h - 1) as f32
            } else {
                0.0
            };
            let ws = if g.grid_w > 1 {
                (POS_SIDE - 1) as f32 / (g.grid_w - 1) as f32
            } else {
                0.0
            };
            for by in 0..bh {
                for bx in 0..bw {
                    for my in 0..2 {
                        for mx in 0..2 {
                            let row = by * 2 + my;
                            let col = bx * 2 + mx;
                            let hsrc = row as f32 * hs;
                            let wsrc = col as f32 * ws;
                            let h0 = hsrc.floor() as usize;
                            let h1 = (h0 + 1).min(POS_SIDE - 1);
                            let w0 = wsrc.floor() as usize;
                            let w1 = (w0 + 1).min(POS_SIDE - 1);
                            let fh = hsrc - h0 as f32;
                            let fw = wsrc - w0 as f32;
                            let dst = (base + (by * bw + bx) * 4 + my * 2 + mx) * HIDDEN;
                            for c in 0..HIDDEN {
                                let a = bf2f(vision.pos[(h0 * POS_SIDE + w0) * HIDDEN + c])
                                    * (1.0 - fh)
                                    * (1.0 - fw)
                                    + bf2f(vision.pos[(h1 * POS_SIDE + w0) * HIDDEN + c])
                                        * fh
                                        * (1.0 - fw)
                                    + bf2f(vision.pos[(h0 * POS_SIDE + w1) * HIDDEN + c])
                                        * (1.0 - fh)
                                        * fw
                                    + bf2f(vision.pos[(h1 * POS_SIDE + w1) * HIDDEN + c]) * fh * fw;
                                out[dst + c] = bf16::from_f32(a);
                            }
                        }
                    }
                }
            }
            base += g.grid_h * g.grid_w;
        }
        Ok(self.gpu.stream.clone_htod(&out)?)
    }
}

fn bf2f(v: bf16) -> f32 {
    v.to_f32()
}

fn append_vision_rope(grid: VisionGrid, cos: &mut Vec<bf16>, sin: &mut Vec<bf16>) {
    let bh = grid.grid_h / 2;
    let bw = grid.grid_w / 2;
    let mut inv = [0f32; 18];
    for (i, v) in inv.iter_mut().enumerate() {
        *v = 1.0 / 10000f32.powf((2 * i) as f32 / 36.0);
    }
    for by in 0..bh {
        for bx in 0..bw {
            for my in 0..2 {
                for mx in 0..2 {
                    let h = by * 2 + my;
                    let w = bx * 2 + mx;
                    let mut ch = [0f32; 18];
                    let mut sh = [0f32; 18];
                    let mut cw = [0f32; 18];
                    let mut sw = [0f32; 18];
                    for i in 0..18 {
                        let ah = h as f32 * inv[i];
                        let aw = w as f32 * inv[i];
                        ch[i] = ah.cos();
                        sh[i] = ah.sin();
                        cw[i] = aw.cos();
                        sw[i] = aw.sin();
                    }
                    for i in 0..18 {
                        cos.push(bf16::from_f32(ch[i]));
                    }
                    for i in 0..18 {
                        cos.push(bf16::from_f32(cw[i]));
                    }
                    for i in 0..18 {
                        cos.push(bf16::from_f32(ch[i]));
                    }
                    for i in 0..18 {
                        cos.push(bf16::from_f32(cw[i]));
                    }
                    for i in 0..18 {
                        sin.push(bf16::from_f32(sh[i]));
                    }
                    for i in 0..18 {
                        sin.push(bf16::from_f32(sw[i]));
                    }
                    for i in 0..18 {
                        sin.push(bf16::from_f32(sh[i]));
                    }
                    for i in 0..18 {
                        sin.push(bf16::from_f32(sw[i]));
                    }
                }
            }
        }
    }
}

impl Engine {
    fn blas_gemm(
        &self,
        x: &CudaSlice<bf16>,
        m: usize,
        k: usize,
        w: &CudaSlice<bf16>,
        n: usize,
    ) -> Result<Buf> {
        let mut out = self.alloc_bf16(m * n)?;
        self.blas_gemm_bf16(x, m, k, w, n, &mut out)?;
        Ok(out)
    }
}

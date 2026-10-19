//! Decoder forward.
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::CudaSlice;
use half::bf16;

use crate::config::Config;
use crate::cuda::Gpu;
use crate::kernels::{self, Kernels};
use crate::tensor::Store;
use crate::weights::{
    FullAttentionWeights, GdnWeights, LayerWeights, LinearWeight, MixerWeights, WeightProvider,
};

type BufferPool = Arc<Mutex<HashMap<usize, Vec<CudaSlice<bf16>>>>>;

/// Recycled device buffer. Returning outputs to the pool avoids per-GEMM
/// cuMemAlloc: on this driver a fresh allocation of some sizes costs ~0.6 ms,
/// which dwarfed the kernel for large-N projections. Slices are reused on the
/// same stream, so ordering guarantees correctness.
pub struct Buf {
    slice: Option<CudaSlice<bf16>>,
    pool: BufferPool,
}

impl Buf {
    pub fn len(&self) -> usize {
        self.slice.as_ref().map(|s| s.len()).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::ops::Deref for Buf {
    type Target = CudaSlice<bf16>;
    fn deref(&self) -> &Self::Target {
        self.slice.as_ref().expect("buffer taken")
    }
}

impl std::ops::DerefMut for Buf {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice.as_mut().expect("buffer taken")
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if let Some(s) = self.slice.take() {
            let mut pool = self.pool.lock().unwrap();
            // Keep at most two buffers per size so long-running servers with
            // many prompt lengths do not accumulate device memory.
            let keep = pool.get(&s.len()).map(|v| v.len() < 2).unwrap_or(true);
            if keep {
                pool.entry(s.len()).or_default().push(s);
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LogitsOpts {
    pub head_ids: Option<Vec<u32>>,
    pub full: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Embed,
    Attn,
    Mlp,
    Layer,
    FinalNorm,
    Logits,
}

impl Stage {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "embed" => Stage::Embed,
            "attn" => Stage::Attn,
            "mlp" => Stage::Mlp,
            "layer" => Stage::Layer,
            "final_norm" | "final" => Stage::FinalNorm,
            "logits" => Stage::Logits,
            other => bail!("unknown stage {other:?} (embed|attn|mlp|layer|final_norm|logits)"),
        })
    }
}

/// `Engine` is `Send`: cuBLAS/cuBLASLt handles and slices are tied to one
/// context/stream, so it may be moved to a blocking worker thread (as the
/// server does with `spawn_blocking`) but it is **not** intended for concurrent
/// forwards. `decision_logits` stores per-call head ids; serialize calls.
pub struct Engine {
    pub gpu: Gpu,
    pub cfg: Config,
    pub store: Store,
    pub kernels: Kernels,
    /// Resident weights. Populated lazily by the CLI layer tests (merged BF16 is
    /// far too large to hold on a 24 GB card) and eagerly via `preload_all` for
    /// the shipped packed W4A16 path.
    weight_cache: Mutex<Vec<Option<Arc<LayerWeights>>>>,
    dump_dir: Mutex<Option<std::path::PathBuf>>,
    /// BF16 scratch for packed W4A16 weights (largest module: 17408x5120).
    /// Reused across GEMMs on the same stream.
    dequant_ws: Mutex<Option<CudaSlice<bf16>>>,
    logits_opts: Mutex<LogitsOpts>,
    pool: BufferPool,
    /// Stream layers instead of keeping them resident. Needed to run the merged
    /// BF16 checkpoint (51 GiB) on a 24 GiB card; the shipped AWQ path keeps
    /// every packed layer resident instead.
    cache_weights: AtomicBool,
    /// Use the fused in-shared-memory dequant + tensor-core GEMM. The
    /// dequant-to-BF16 + cuBLAS path remains available as the reference.
    fused_w4a16: AtomicBool,
    /// Minimum N for the fused kernel; smaller modules use dequant+cuBLAS.
    fused_min_n: std::sync::atomic::AtomicUsize,
    /// Above this M, use dequant+cuBLAS: the fused kernel re-dequantizes W per
    /// M tile, so large prefills are better served by the reference path.
    fused_max_m: std::sync::atomic::AtomicUsize,
}

const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<Engine>();
};

impl Engine {
    pub fn new(model_dir: &std::path::Path) -> Result<Self> {
        let cfg = Config::from_dir(model_dir)?;
        let store = Store::open(model_dir)?;
        let gpu = Gpu::new(0)?;
        let kernels = Kernels::load(&gpu)?;
        let weight_cache = Mutex::new(vec![None; cfg.num_hidden_layers]);
        Ok(Self {
            gpu,
            cfg,
            store,
            kernels,
            weight_cache,
            dump_dir: Mutex::new(None),
            dequant_ws: Mutex::new(None),
            logits_opts: Mutex::new(LogitsOpts::default()),
            pool: Arc::new(Mutex::new(HashMap::new())),
            cache_weights: AtomicBool::new(true),
            fused_w4a16: AtomicBool::new(false),
            fused_min_n: std::sync::atomic::AtomicUsize::new(5120),
            fused_max_m: std::sync::atomic::AtomicUsize::new(256),
        })
    }

    /// Load every decoder layer into persistent GPU buffers. This is the
    /// residency model of the shipped engine: one upload at startup, no
    /// per-request PCIe traffic.
    /// Write f32 copies of selected intermediates for golden comparison.
    pub fn set_logits_opts(&self, head_ids: Option<Vec<u32>>, full: bool) {
        *self.logits_opts.lock().unwrap() = LogitsOpts { head_ids, full };
    }

    pub fn set_fused_w4a16(&self, enabled: bool) {
        self.fused_w4a16.store(enabled, Ordering::Relaxed);
    }

    pub fn set_fused_min_n(&self, n: usize) {
        self.fused_min_n.store(n, Ordering::Relaxed);
    }

    pub fn set_fused_max_m(&self, m: usize) {
        self.fused_max_m.store(m, Ordering::Relaxed);
    }

    pub fn set_cache_weights(&self, enabled: bool) {
        self.cache_weights.store(enabled, Ordering::Relaxed);
    }

    pub fn set_dump(&self, dir: &std::path::Path) {
        *self.dump_dir.lock().unwrap() = Some(dir.to_path_buf());
    }

    fn dump_enabled(&self) -> bool {
        self.dump_dir.lock().unwrap().is_some()
    }

    fn dump_f32(&self, name: &str, vals: &[f32], rows: usize, cols: usize) -> Result<()> {
        let dir = match self.dump_dir.lock().unwrap().clone() {
            Some(d) => d,
            None => return Ok(()),
        };
        std::fs::create_dir_all(&dir)?;
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join(format!("{name}.f32")), bytes)?;
        let shape = serde_json::json!({"shape": [rows, cols], "dtype": "f32"});
        std::fs::write(
            dir.join(format!("{name}.json")),
            serde_json::to_vec(&shape)?,
        )?;
        Ok(())
    }

    fn dump_dev(&self, name: &str, x: &CudaSlice<bf16>, rows: usize, cols: usize) -> Result<()> {
        if !self.dump_enabled() {
            return Ok(());
        }
        let vals = self.to_f32(x, rows * cols)?;
        self.dump_f32(name, &vals, rows, cols)
    }

    pub fn preload_all(&self) -> Result<()> {
        // One upload at startup; no per-forward PCIe traffic for decoder weights.
        for i in 0..self.cfg.num_hidden_layers {
            self.layer_weights(i)?;
        }
        if let Ok((free, _)) = self.gpu.mem_info() {
            eprintln!(
                "preloaded {} layers, {:.2} GiB device free",
                self.cfg.num_hidden_layers,
                free as f64 / (1u64 << 30) as f64
            );
        }
        Ok(())
    }

    fn layer_weights(&self, idx: usize) -> Result<Arc<LayerWeights>> {
        let mut cache = self.weight_cache.lock().expect("weight cache poisoned");
        if let Some(w) = &cache[idx] {
            return Ok(w.clone());
        }
        let prov = WeightProvider::new(&self.store, &self.gpu.stream)?;
        let w = Arc::new(prov.layer(idx, self.cfg.layer_types[idx])?);
        if self.cache_weights.load(Ordering::Relaxed) {
            cache[idx] = Some(w.clone());
        }
        Ok(w)
    }

    fn alloc_bf16(&self, n: usize) -> Result<Buf> {
        {
            let mut pool = self.pool.lock().unwrap();
            if pool.len() > 128 {
                pool.clear();
            }
        }
        if let Some(v) = self.pool.lock().unwrap().get_mut(&n) {
            if let Some(s) = v.pop() {
                return Ok(Buf {
                    slice: Some(s),
                    pool: self.pool.clone(),
                });
            }
        }
        let s = self.gpu.stream.alloc_zeros::<bf16>(n)?;
        Ok(Buf {
            slice: Some(s),
            pool: self.pool.clone(),
        })
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Buf> {
        let h = self.cfg.hidden_size;
        let name = WeightProvider::new(&self.store, &self.gpu.stream)?.embedding();
        let mut host = vec![bf16::ZERO; tokens.len() * h];
        for (t, &tok) in tokens.iter().enumerate() {
            if tok as usize >= self.cfg.vocab_size {
                bail!(
                    "token id {tok} out of range (vocab {})",
                    self.cfg.vocab_size
                );
            }
            self.store
                .bf16_row(&name, tok as usize, &mut host[t * h..(t + 1) * h])
                .with_context(|| format!("embedding row {tok}"))?;
        }
        let mut buf = self.alloc_bf16(tokens.len() * h)?;
        self.gpu.stream.memcpy_htod(&host, &mut *buf)?;
        Ok(buf)
    }

    /// causal LM over a prompt, returning the hidden state at the requested
    /// layer/stage as f32 `[seq, hidden]`.
    pub fn forward(
        &self,
        tokens: &[u32],
        layer: isize,
        stage: Stage,
        last_only: bool,
    ) -> Result<Vec<f32>> {
        let x = self.embed(tokens)?;
        self.forward_dev(x, tokens.len(), 0, layer, stage, last_only)
    }

    /// Continue from an externally supplied hidden state (f32, `[seq, hidden]`).
    /// Used to validate one layer in isolation against golden data.
    pub fn forward_hidden(
        &self,
        hidden: &[f32],
        seq: usize,
        layer: isize,
        stage: Stage,
        last_only: bool,
    ) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        if hidden.len() != seq * h {
            bail!(
                "input hidden has {} values, expected {}",
                hidden.len(),
                seq * h
            );
        }
        let host: Vec<bf16> = hidden.iter().map(|v| bf16::from_f32(*v)).collect();
        let mut x = self.alloc_bf16(seq * h)?;
        self.gpu.stream.memcpy_htod(&host, &mut *x)?;
        if layer < 0 {
            bail!("--input requires --layer >= 0");
        }
        self.forward_dev(x, seq, layer as usize, layer, stage, last_only)
    }

    fn forward_dev(
        &self,
        x: Buf,
        seq: usize,
        start: usize,
        layer: isize,
        stage: Stage,
        last_only: bool,
    ) -> Result<Vec<f32>> {
        if stage == Stage::Embed || layer < 0 {
            return self.to_f32(&x, seq * self.cfg.hidden_size);
        }
        let last = layer as usize;
        if last >= self.cfg.num_hidden_layers {
            bail!(
                "layer {last} out of range (model has {})",
                self.cfg.num_hidden_layers
            );
        }
        let mut x = x;
        for i in start..=last {
            let w = self.layer_weights(i)?;
            let mut captured = None;
            self.dump_dev("00_layer_input", &x, seq, self.cfg.hidden_size)?;
            let next = self.run_layer(&x, seq, &w, i == last, stage, &mut captured)?;
            if i == last {
                self.dump_dev("06_layer_output", &next, seq, self.cfg.hidden_size)?;
            }
            if let Some(v) = captured {
                return self.to_f32(&v, seq * self.cfg.hidden_size);
            }
            x = next;
        }
        match stage {
            Stage::FinalNorm | Stage::Logits => {
                let norm = WeightProvider::new(&self.store, &self.gpu.stream)?.final_norm()?;
                let mut h = self.alloc_bf16(seq * self.cfg.hidden_size)?;
                kernels::rmsnorm(
                    &self.gpu,
                    &self.kernels.rmsnorm,
                    &x,
                    &norm,
                    &mut h,
                    seq,
                    self.cfg.hidden_size,
                    self.cfg.rms_norm_eps,
                    true,
                )?;
                if stage == Stage::FinalNorm {
                    return self.slice_out(&h, seq, self.cfg.hidden_size, last_only);
                }
                let opts = self.logits_opts.lock().unwrap().clone();
                if let Some(ids) = opts.head_ids {
                    self.logits_rows(&h, seq, last_only, &ids)
                } else if opts.full {
                    self.logits_chunked(&h, seq, last_only)
                } else {
                    bail!("--stage logits requires --head-ids <ids> or --full-logits")
                }
            }
            _ => self.slice_out(&x, seq, self.cfg.hidden_size, last_only),
        }
    }

    fn slice_out(
        &self,
        x: &CudaSlice<bf16>,
        seq: usize,
        cols: usize,
        last_only: bool,
    ) -> Result<Vec<f32>> {
        let all = self.to_f32(x, seq * cols)?;
        if last_only {
            Ok(all[(seq - 1) * cols..].to_vec())
        } else {
            Ok(all)
        }
    }

    /// Runs one decoder layer. If `capture` is set and `stage` is Attn/Mlp, the
    /// requested intermediate is stored in `captured` (the returned tensor is
    /// then the intermediate residual and not used by the caller).
    fn run_layer(
        &self,
        x: &CudaSlice<bf16>,
        seq: usize,
        w: &LayerWeights,
        capture: bool,
        stage: Stage,
        captured: &mut Option<Buf>,
    ) -> Result<Buf> {
        let hidden = self.cfg.hidden_size;
        let h = self.alloc_bf16(seq * hidden)?;
        let mut h = h;
        kernels::rmsnorm(
            &self.gpu,
            &self.kernels.rmsnorm,
            x,
            &w.input_norm,
            &mut h,
            seq,
            hidden,
            self.cfg.rms_norm_eps,
            true,
        )?;
        self.dump_dev("01_input_norm", &h, seq, hidden)?;

        let mix = match &w.mixer {
            MixerWeights::Full(a) => self.full_attention(&h, seq, a, capture)?,
            MixerWeights::Gdn(a) => self.gdn(&h, seq, a, capture)?,
        };
        if capture {
            self.dump_dev("02_mixer_out", &mix, seq, hidden)?;
        }
        if capture && stage == Stage::Attn {
            *captured = Some(mix);
            return Ok(self.alloc_bf16(0)?);
        }

        let mut x1 = self.alloc_bf16(seq * hidden)?;
        kernels::add(&self.gpu, &self.kernels.add, x, &mix, &mut x1, seq * hidden)?;
        if capture {
            self.dump_dev("03_after_attn_residual", &x1, seq, hidden)?;
        }

        let mut h2 = self.alloc_bf16(seq * hidden)?;
        kernels::rmsnorm(
            &self.gpu,
            &self.kernels.rmsnorm,
            &x1,
            &w.post_norm,
            &mut h2,
            seq,
            hidden,
            self.cfg.rms_norm_eps,
            true,
        )?;
        if capture {
            self.dump_dev("04_post_attn_norm", &h2, seq, hidden)?;
        }

        let mlp = self.mlp(&h2, seq, w)?;
        if capture {
            self.dump_dev("05_mlp_out", &mlp, seq, hidden)?;
        }
        if capture && stage == Stage::Mlp {
            *captured = Some(mlp);
            return Ok(x1);
        }
        let mut out = self.alloc_bf16(seq * hidden)?;
        kernels::add(
            &self.gpu,
            &self.kernels.add,
            &x1,
            &mlp,
            &mut out,
            seq * hidden,
        )?;
        Ok(out)
    }

    fn mlp(&self, h: &CudaSlice<bf16>, seq: usize, w: &LayerWeights) -> Result<Buf> {
        let inter = self.cfg.intermediate_size;
        let gate = self.gemm(h, seq, &w.gate_proj)?;
        let up = self.gemm(h, seq, &w.up_proj)?;
        let mut act = self.alloc_bf16(seq * inter)?;
        kernels::silu_mul(
            &self.gpu,
            &self.kernels.silu_mul,
            &gate,
            &up,
            &mut act,
            seq * inter,
        )?;
        self.gemm(&act, seq, &w.down_proj)
    }

    fn full_attention(
        &self,
        h: &CudaSlice<bf16>,
        seq: usize,
        w: &FullAttentionWeights,
        capture: bool,
    ) -> Result<Buf> {
        let cfg = &self.cfg;
        let (heads, kv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let qg = self.gemm(h, seq, &w.q_proj)?;
        let kk = self.gemm(h, seq, &w.k_proj)?;
        let vv = self.gemm(h, seq, &w.v_proj)?;
        if capture {
            self.dump_dev("attn_q_proj", &qg, seq, w.q_proj.out())?;
            self.dump_dev("attn_k_proj", &kk, seq, kv * hd)?;
            self.dump_dev("attn_v_proj", &vv, seq, kv * hd)?;
        }

        let (q, gate) = if cfg.attn_output_gate {
            let mut q = self.alloc_bf16(seq * heads * hd)?;
            let mut gate = self.alloc_bf16(seq * heads * hd)?;
            kernels::split_q_gate(
                &self.gpu,
                &self.kernels.split_q_gate,
                &qg,
                &mut q,
                &mut gate,
                seq,
                heads,
                hd,
            )?;
            if capture {
                self.dump_dev("attn_q", &q, seq, heads * hd)?;
                self.dump_dev("attn_gate", &gate, seq, heads * hd)?;
            }
            (q, Some(gate))
        } else {
            (qg, None)
        };

        let q = q;
        let mut qn = self.alloc_bf16(seq * heads * hd)?;
        kernels::rmsnorm(
            &self.gpu,
            &self.kernels.rmsnorm,
            &q,
            &w.q_norm,
            &mut qn,
            seq * heads,
            hd,
            cfg.rms_norm_eps,
            true,
        )?;
        if capture {
            self.dump_dev("attn_q_norm", &qn, seq, heads * hd)?;
        }
        let mut kn = self.alloc_bf16(seq * kv * hd)?;
        kernels::rmsnorm(
            &self.gpu,
            &self.kernels.rmsnorm,
            &kk,
            &w.k_norm,
            &mut kn,
            seq * kv,
            hd,
            cfg.rms_norm_eps,
            true,
        )?;
        if capture {
            self.dump_dev("attn_k_norm", &kn, seq, kv * hd)?;
        }

        let rd = cfg.rotary_dim();
        let (cos, sin) = self.rope_tables(seq)?;
        kernels::rope(
            &self.gpu,
            &self.kernels.rope,
            &mut qn,
            &mut kn,
            &cos,
            &sin,
            seq,
            heads,
            kv,
            hd,
            rd / 2,
        )?;
        if capture {
            self.dump_dev("attn_q_rope", &qn, seq, heads * hd)?;
            self.dump_dev("attn_k_rope", &kn, seq, kv * hd)?;
        }

        let mut attn = self.alloc_bf16(seq * heads * hd)?;
        kernels::attention(
            &self.gpu,
            &self.kernels.attention,
            &qn,
            &kn,
            &vv,
            &mut attn,
            seq,
            heads,
            kv,
            hd,
            1.0 / (hd as f32).sqrt(),
        )?;
        if capture {
            self.dump_dev("attn_weights_out", &attn, seq, heads * hd)?;
        }

        let gated = match gate {
            Some(g) => {
                let mut o = self.alloc_bf16(seq * heads * hd)?;
                kernels::sigmoid_gate(
                    &self.gpu,
                    &self.kernels.sigmoid_gate,
                    &attn,
                    &g,
                    &mut o,
                    seq * heads * hd,
                )?;
                o
            }
            None => attn,
        };
        if capture {
            self.dump_dev("attn_gated", &gated, seq, heads * hd)?;
        }
        self.gemm(&gated, seq, &w.o_proj)
    }

    fn gdn(&self, h: &CudaSlice<bf16>, seq: usize, w: &GdnWeights, capture: bool) -> Result<Buf> {
        let cfg = &self.cfg;
        let (nk, nv) = (cfg.linear_num_key_heads, cfg.linear_num_value_heads);
        let (dk, dv) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
        let conv_dim = 2 * cfg.gdn_key_dim() + cfg.gdn_value_dim();

        let qkv = self.gemm(h, seq, &w.in_proj_qkv)?;
        if capture {
            self.dump_dev("gdn_in_proj_qkv", &qkv, seq, conv_dim)?;
        }
        if qkv.len() != seq * conv_dim {
            bail!(
                "in_proj_qkv returned {} values, expected {}",
                qkv.len(),
                seq * conv_dim
            );
        }
        let mut conv = self.alloc_bf16(seq * conv_dim)?;
        kernels::conv1d_silu(
            &self.gpu,
            &self.kernels.conv1d_silu,
            &qkv,
            &w.conv1d,
            &mut conv,
            seq,
            conv_dim,
        )?;
        if capture {
            self.dump_dev("gdn_mixed_qkv_conv", &conv, seq, conv_dim)?;
        }

        let z = self.gemm(h, seq, &w.in_proj_z)?;
        let b = self.gemm(h, seq, &w.in_proj_b)?;
        let a = self.gemm(h, seq, &w.in_proj_a)?;
        if capture {
            self.dump_dev("gdn_z", &z, seq, cfg.gdn_value_dim())?;
            self.dump_dev("gdn_b", &b, seq, nv)?;
            self.dump_dev("gdn_a", &a, seq, nv)?;
        }

        let qf = self.gpu.stream.alloc_zeros::<f32>(seq * nv * dk)?;
        let kf = self.gpu.stream.alloc_zeros::<f32>(seq * nv * dk)?;
        let vf = self.gpu.stream.alloc_zeros::<f32>(seq * nv * dv)?;
        let g = self.gpu.stream.alloc_zeros::<f32>(seq * nv)?;
        let beta = self.gpu.stream.alloc_zeros::<f32>(seq * nv)?;
        let mut qf = qf;
        let mut kf = kf;
        let mut vf = vf;
        let mut g = g;
        let mut beta = beta;
        kernels::gdn_prepare(
            &self.gpu,
            &self.kernels.gdn_prepare,
            &conv,
            &b,
            &a,
            &w.a_log,
            &w.dt_bias,
            &mut qf,
            &mut kf,
            &mut vf,
            &mut g,
            &mut beta,
            seq,
            nk,
            nv,
            dk,
            dv,
        )?;
        if capture {
            let qfv = self.gpu.stream.clone_dtoh(&qf)?;
            let kfv = self.gpu.stream.clone_dtoh(&kf)?;
            let vfv = self.gpu.stream.clone_dtoh(&vf)?;
            let gv = self.gpu.stream.clone_dtoh(&g)?;
            let bv = self.gpu.stream.clone_dtoh(&beta)?;
            self.dump_f32("gdn_qf", &qfv, seq * nv, dk)?;
            self.dump_f32("gdn_kf", &kfv, seq * nv, dk)?;
            self.dump_f32("gdn_vf", &vfv, seq * nv, dv)?;
            self.dump_f32("gdn_g", &gv, seq, nv)?;
            self.dump_f32("gdn_beta", &bv, seq, nv)?;
        }

        let mut state = self.gpu.stream.alloc_zeros::<f32>(nv * dk * dv)?;
        let mut core = self.alloc_bf16(seq * nv * dv)?;
        kernels::gdn_recurrent(
            &self.gpu,
            &self.kernels.gdn_recurrent,
            &qf,
            &kf,
            &vf,
            &g,
            &beta,
            &mut state,
            &mut core,
            seq,
            nv,
            dk,
            dv,
        )?;
        if capture {
            self.dump_dev("gdn_recurrent_out", &core, seq, nv * dv)?;
            let sv = self.gpu.stream.clone_dtoh(&state)?;
            self.dump_f32("gdn_state", &sv, nv * dk, dv)?;
        }

        let mut normed = self.alloc_bf16(seq * nv * dv)?;
        kernels::rmsnorm_gated(
            &self.gpu,
            &self.kernels.rmsnorm_gated,
            &core,
            &z,
            &w.norm,
            &mut normed,
            seq * nv,
            dv,
            cfg.rms_norm_eps,
        )?;
        if capture {
            self.dump_dev("gdn_norm_out", &normed, seq * nv, dv)?;
        }
        let out = self.gemm(&normed, seq, &w.out_proj)?;
        if capture {
            self.dump_dev("gdn_out_proj", &out, seq, self.cfg.hidden_size)?;
        }
        Ok(out)
    }

    fn rope_tables(&self, seq: usize) -> Result<(CudaSlice<bf16>, CudaSlice<bf16>)> {
        let dim = self.cfg.rotary_dim();
        let half = dim / 2;
        let theta = self.cfg.rope_theta;
        let mut inv = vec![0f32; half];
        for (i, v) in inv.iter_mut().enumerate() {
            *v = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
        }
        let mut cos = vec![bf16::ZERO; seq * half];
        let mut sin = vec![bf16::ZERO; seq * half];
        for t in 0..seq {
            for i in 0..half {
                let ang = t as f32 * inv[i];
                cos[t * half + i] = bf16::from_f32(ang.cos());
                sin[t * half + i] = bf16::from_f32(ang.sin());
            }
        }
        Ok((
            self.gpu.stream.clone_htod(&cos)?,
            self.gpu.stream.clone_htod(&sin)?,
        ))
    }

    /// C = x @ W^T for row-major bf16 buffers (cuBLAS column-major trick).
    pub fn gemm(&self, x: &CudaSlice<bf16>, rows: usize, w: &LinearWeight) -> Result<Buf> {
        let (n, k) = (w.out(), w.in_());
        if x.len() != rows * k {
            bail!("gemm input has {} values, expected {}", x.len(), rows * k);
        }
        let dbg = std::env::var("JEV_GEMM_DEBUG").is_ok();
        let mark = |label: &str, t: std::time::Instant| {
            if dbg {
                let _ = self.gpu.synchronize();
                eprintln!("{label}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
            }
        };
        if dbg {
            let _ = self.gpu.synchronize();
        }
        let t0 = std::time::Instant::now();
        match w {
            LinearWeight::Bf16 { w: wd, .. } => {
                let mut c = self.alloc_bf16(rows * n)?;
                mark("  alloc", t0);
                let cfg = GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,
                    n: rows as i32,
                    k: k as i32,
                    alpha: bf16::from_f32(1.0),
                    lda: k as i32,
                    ldb: k as i32,
                    beta: bf16::from_f32(0.0),
                    ldc: n as i32,
                };
                let t1 = std::time::Instant::now();
                unsafe {
                    self.gpu
                        .blas
                        .gemm(cfg, wd, x, &mut *c)
                        .context("cublas bf16 gemm")?;
                }
                mark("  cublas", t1);
                Ok(c)
            }
            LinearWeight::PackedW4A16 {
                group_size,
                packed,
                scale,
                zp,
                ..
            } => {
                if self.fused_w4a16.load(Ordering::Relaxed)
                    && n >= self.fused_min_n.load(Ordering::Relaxed)
                    && rows <= self.fused_max_m.load(Ordering::Relaxed)
                {
                    let mut c = self.alloc_bf16(rows * n)?;
                    mark("  alloc", t0);
                    let t1 = std::time::Instant::now();
                    kernels::gemm_w4a16(
                        &self.gpu,
                        &self.kernels.gemm_w4a16,
                        x,
                        packed,
                        scale,
                        zp,
                        &mut c,
                        rows,
                        n,
                        k,
                    )?;
                    mark("  fused", t1);
                    return Ok(c);
                }
                // Reference path: dequantize into a reusable BF16 workspace,
                // then run the same cuBLAS GEMM as the merged-BF16 path.
                let need = n * k;
                let mut guard = self.dequant_ws.lock().unwrap();
                if guard.as_ref().is_none_or(|w| w.len() < need) {
                    *guard = Some(self.gpu.stream.alloc_zeros::<bf16>(need)?);
                }
                let ws = guard.as_mut().unwrap();
                kernels::dequant_w4a16(
                    &self.gpu,
                    &self.kernels.dequant_w4a16,
                    packed,
                    scale,
                    zp,
                    ws,
                    n,
                    k,
                    *group_size,
                )?;
                let view = ws.try_slice(..need).context("dequant workspace view")?;
                let mut c = self.alloc_bf16(rows * n)?;
                let cfg = GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,
                    n: rows as i32,
                    k: k as i32,
                    alpha: bf16::from_f32(1.0),
                    lda: k as i32,
                    ldb: k as i32,
                    beta: bf16::from_f32(0.0),
                    ldc: n as i32,
                };
                unsafe {
                    self.gpu
                        .blas
                        .gemm(cfg, &view, x, &mut *c)
                        .context("cublas packed W4A16 gemm")?;
                }
                Ok(c)
            }
        }
    }

    /// End-to-end text path for the decision server: full forward, final norm,
    /// last-token hidden state (`[hidden]`). The engine keeps all decoder
    /// weights resident after `preload_all`, so this performs no weight I/O.
    pub fn forward_last_hidden(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let last = self.cfg.num_hidden_layers.saturating_sub(1) as isize;
        self.forward(tokens, last, Stage::FinalNorm, true)
    }

    /// Full forward + final norm + lm_head rows for `head_ids`, at the last
    /// position. `head_ids` are the decision head's verbalizer token ids.
    pub fn decision_logits(&self, tokens: &[u32], head_ids: &[u32]) -> Result<Vec<f32>> {
        self.set_logits_opts(Some(head_ids.to_vec()), false);
        let last = self.cfg.num_hidden_layers.saturating_sub(1) as isize;
        self.forward(tokens, last, Stage::Logits, true)
    }

    /// Diagnostic: run one packed linear on host f32 input, via the reference
    /// dequant+cuBLAS path or the fused kernel.
    pub fn matmul_module(
        &self,
        module: &str,
        input: &[f32],
        rows: usize,
        fused: bool,
    ) -> Result<Vec<f32>> {
        let prov = WeightProvider::new(&self.store, &self.gpu.stream)?;
        let w = prov.linear(module)?;
        if input.len() != rows * w.in_() {
            bail!(
                "input has {} values, expected {}",
                input.len(),
                rows * w.in_()
            );
        }
        let host: Vec<bf16> = input.iter().map(|v| bf16::from_f32(*v)).collect();
        let x = self.gpu.stream.clone_htod(&host)?;
        self.set_fused_w4a16(fused);
        let y = self.gemm(&x, rows, &w)?;
        self.set_fused_w4a16(false);
        self.to_f32(&y, rows * w.out())
    }

    /// Benchmark one module with preloaded device weights (no upload inside the
    /// timed section). Returns (reference_ms, fused_ms).
    pub fn bench_matmul(
        &self,
        module: &str,
        input: &[f32],
        rows: usize,
        repeat: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let prov = WeightProvider::new(&self.store, &self.gpu.stream)?;
        let w = prov.linear(module)?;
        let host: Vec<bf16> = input.iter().map(|v| bf16::from_f32(*v)).collect();
        let x = self.gpu.stream.clone_htod(&host)?;
        let mut refs = Vec::new();
        let mut fused = Vec::new();
        for _ in 0..repeat {
            self.set_fused_w4a16(false);
            self.gpu.synchronize()?;
            let t = std::time::Instant::now();
            let _ = self.gemm(&x, rows, &w)?;
            self.gpu.synchronize()?;
            refs.push(t.elapsed().as_secs_f64() * 1e3);
        }
        for _ in 0..repeat {
            self.set_fused_w4a16(true);
            self.set_fused_min_n(0);
            self.gpu.synchronize()?;
            let t = std::time::Instant::now();
            let _ = self.gemm(&x, rows, &w)?;
            self.gpu.synchronize()?;
            fused.push(t.elapsed().as_secs_f64() * 1e3);
        }
        self.set_fused_w4a16(false);
        Ok((refs, fused))
    }

    /// Dequantize one packed module to f32 (validation hook for AWQ).
    pub fn dequant_module(&self, module: &str) -> Result<Vec<f32>> {
        let prov = WeightProvider::new(&self.store, &self.gpu.stream)?;
        let w = prov.linear(module)?;
        match &w {
            LinearWeight::Bf16 { .. } => bail!("{module} is not packed"),
            LinearWeight::PackedW4A16 {
                out,
                in_,
                group_size,
                packed,
                scale,
                zp,
                ..
            } => {
                let need = out * in_;
                let mut ws = self.gpu.stream.alloc_zeros::<bf16>(need)?;
                kernels::dequant_w4a16(
                    &self.gpu,
                    &self.kernels.dequant_w4a16,
                    packed,
                    scale,
                    zp,
                    &mut ws,
                    *out,
                    *in_,
                    *group_size,
                )?;
                self.to_f32(&ws, need)
            }
        }
    }

    /// Head logits for selected token ids, at every position (or the last).
    fn logits_rows(
        &self,
        h: &CudaSlice<bf16>,
        seq: usize,
        last_only: bool,
        ids: &[u32],
    ) -> Result<Vec<f32>> {
        let hidden = self.cfg.hidden_size;
        let mut host = vec![bf16::ZERO; ids.len() * hidden];
        for (i, &id) in ids.iter().enumerate() {
            if id as usize >= self.cfg.vocab_size {
                bail!("lm_head id {id} out of range");
            }
            self.store.bf16_row(
                "lm_head.weight",
                id as usize,
                &mut host[i * hidden..(i + 1) * hidden],
            )?;
        }
        let w = self.gpu.stream.clone_htod(&host)?;
        let lw = LinearWeight::Bf16 {
            out: ids.len(),
            in_: hidden,
            w,
        };
        let logits = self.gemm(h, seq, &lw)?;
        self.slice_out(&logits, seq, ids.len(), last_only)
    }

    /// Full-vocabulary logits, streaming lm_head rows in chunks.
    fn logits_chunked(&self, h: &CudaSlice<bf16>, seq: usize, last_only: bool) -> Result<Vec<f32>> {
        const CHUNK: usize = 8192;
        let hidden = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let all = self.gpu.stream.clone_dtoh(h)?;
        let (rows, input) = if last_only {
            (1usize, all[(seq - 1) * hidden..].to_vec())
        } else {
            (seq, all)
        };
        let x = self.gpu.stream.clone_htod(&input)?;
        let mut out: Vec<f32> = Vec::with_capacity(rows * vocab);
        for start in (0..vocab).step_by(CHUNK) {
            let n = CHUNK.min(vocab - start);
            let mut host = vec![bf16::ZERO; n * hidden];
            for r in 0..n {
                self.store.bf16_row(
                    "lm_head.weight",
                    start + r,
                    &mut host[r * hidden..(r + 1) * hidden],
                )?;
            }
            let w = self.gpu.stream.clone_htod(&host)?;
            let lw = LinearWeight::Bf16 {
                out: n,
                in_: hidden,
                w,
            };
            let c = self.gemm(&x, rows, &lw)?;
            out.extend(self.to_f32(&c, rows * n)?);
        }
        Ok(out)
    }

    pub fn to_f32(&self, x: &CudaSlice<bf16>, n: usize) -> Result<Vec<f32>> {
        let host: Vec<bf16> = self.gpu.stream.clone_dtoh(x).context("d2h copy")?;
        if host.len() != n {
            bail!("expected {n} values, got {}", host.len());
        }
        Ok(host.iter().map(|v| v.to_f32()).collect())
    }
}

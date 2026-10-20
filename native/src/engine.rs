//! Decoder forward.
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaEvent, CudaSlice};
use half::bf16;

use crate::config::Config;
use crate::cuda::Gpu;
use crate::kernels::{self, Kernels};
use crate::tensor::Store;
use crate::weights::{
    FullAttentionWeights, GdnWeights, LayerWeights, LinearWeight, Lora, MixerWeights,
    WeightProvider,
};

struct PoolState {
    free: Mutex<HashMap<usize, Vec<CudaSlice<bf16>>>>,
    /// When set (very long prefills), dropped buffers are released to the
    /// driver instead of retained, to keep peak memory inside 24 GiB.
    low_mem: AtomicBool,
}
type BufferPool = Arc<PoolState>;

struct F32PoolState {
    free: Mutex<HashMap<usize, Vec<CudaSlice<f32>>>>,
    /// Mirrors `PoolState::low_mem`: at very long prefills, dropped f32 scratch
    /// is released instead of retained so the 24 GiB budget holds.
    low_mem: AtomicBool,
}
type F32Pool = Arc<F32PoolState>;

/// Recycled f32 scratch (GDN q/k/v/g/beta and recurrent state). These are
/// fully overwritten every layer (except `state`, which is explicitly zeroed),
/// so reuse avoids the per-layer cuMemAlloc/memset churn of the prefill path.
pub struct F32Buf {
    slice: Option<CudaSlice<f32>>,
    pool: F32Pool,
}

impl F32Buf {
    pub fn len(&self) -> usize {
        self.slice.as_ref().map(|s| s.len()).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::ops::Deref for F32Buf {
    type Target = CudaSlice<f32>;
    fn deref(&self) -> &Self::Target {
        self.slice.as_ref().expect("buffer taken")
    }
}

impl std::ops::DerefMut for F32Buf {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice.as_mut().expect("buffer taken")
    }
}

impl Drop for F32Buf {
    fn drop(&mut self) {
        if let Some(s) = self.slice.take() {
            if self.pool.low_mem.load(Ordering::Relaxed) {
                return;
            }
            let mut pool = self.pool.free.lock().unwrap();
            let keep = pool.get(&s.len()).map(|v| v.len() < 2).unwrap_or(true);
            if keep {
                pool.entry(s.len()).or_default().push(s);
            }
        }
    }
}

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
            if self.pool.low_mem.load(Ordering::Relaxed) {
                return; // s drops here -> cuMemFree
            }
            let mut pool = self.pool.free.lock().unwrap();
            // Keep at most two buffers per size so long-running servers with
            // many prompt lengths do not accumulate device memory.
            let keep = pool.get(&s.len()).map(|v| v.len() < 2).unwrap_or(true);
            if keep {
                pool.entry(s.len()).or_default().push(s);
            }
        }
    }
}

struct ProfEntry {
    name: &'static str,
    start: CudaEvent,
    end: CudaEvent,
}

struct Prof {
    enabled: bool,
    entries: Mutex<Vec<ProfEntry>>,
}

impl Prof {
    fn new() -> Self {
        Self {
            enabled: std::env::var("DECIDRA_PROFILE").is_ok(),
            entries: Mutex::new(Vec::new()),
        }
    }
}

/// Host-side time spent inside each `timed!` region. CUDA event timing also
/// includes host enqueue gaps, so this separates launch/alloc overhead from
/// GPU execution. Enabled by `DECIDRA_CPU_PROFILE=1`; reported next to the GPU
/// table at the end of a forward.
struct CpuProf {
    enabled: bool,
    entries: Mutex<std::collections::HashMap<&'static str, (f64, usize)>>,
}

impl CpuProf {
    fn new() -> Self {
        Self {
            enabled: std::env::var("DECIDRA_CPU_PROFILE").is_ok(),
            entries: Mutex::new(std::collections::HashMap::new()),
        }
    }

    fn begin(&self) -> Option<std::time::Instant> {
        if self.enabled {
            Some(std::time::Instant::now())
        } else {
            None
        }
    }

    fn end(&self, name: &'static str, start: Option<std::time::Instant>) {
        let Some(start) = start else { return };
        let us = start.elapsed().as_secs_f64() * 1e6;
        let mut m = self.entries.lock().unwrap();
        let e = m.entry(name).or_insert((0.0, 0));
        e.0 += us;
        e.1 += 1;
    }

    fn report(&self) {
        if !self.enabled {
            return;
        }
        let mut m = std::mem::take(&mut *self.entries.lock().unwrap());
        let total: f64 = m.values().map(|v| v.0).sum();
        if total <= 0.0 {
            return;
        }
        let mut rows: Vec<_> = m.drain().collect();
        rows.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
        eprintln!("--- cpu profile (sum of regions {:.2} ms) ---", total / 1e3);
        for (name, (us, n)) in rows {
            eprintln!(
                "{name:16} {:9.3} ms  {:6.2}%  n={n}  {:.1} us/op",
                us / 1e3,
                us / total * 100.0,
                us / n as f64
            );
        }
    }
}

/// Accumulated `gdn_chunked_tc_prof` phase counters (`DECIDRA_GDN_PROF=1`).
#[derive(Default)]
struct GdnProf {
    cycles: [u64; 10],
    blocks: u64,
    chunks: u64,
}

const GDN_PHASES: [&str; 10] = [
    "load", "kk^t", "mask", "solve", "vnew", "qload", "qs", "qk+mask", "out", "state",
];

fn gemm_debug() -> bool {
    static DEBUG: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("DECIDRA_GEMM_DEBUG").is_ok());
    *DEBUG
}

fn gemm_debug_sync() -> bool {
    static DEBUG: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("DECIDRA_GEMM_DEBUG_SYNC").is_ok());
    *DEBUG
}

/// `DECIDRA_MARLIN_CHUNK`: 0 = one launch for the whole M.
fn marlin_chunk() -> usize {
    static CHUNK: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("DECIDRA_MARLIN_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    });
    *CHUNK
}

/// Diagnostic: replicate one row-wide scale to every K group.
fn int8_group_uniform() -> bool {
    static UNIFORM: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("DECIDRA_INT8_GROUP_UNIFORM").is_some());
    *UNIFORM
}

/// Reused per-token INT8 activation scratch. The worker is serialized, so one
/// buffer pair serves every INT8 GEMM in a forward; the stream orders the next
/// quantization after the previous Marlin kernel has consumed it.
struct Int8Scratch {
    x: CudaSlice<i8>,
    scales: CudaSlice<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct LogitsOpts {
    pub head_ids: Option<Vec<u32>>,
    pub full: bool,
}

macro_rules! timed {
    ($self:expr, $name:expr, $e:expr) => {{
        let __t = $self.prof_begin($name);
        let __c = $self.cpu_prof.begin();
        let __r = $e;
        $self.cpu_prof.end($name, __c);
        $self.prof_end($name, __t);
        __r
    }};
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
struct LoraHead {
    /// A: [rank, hidden].
    a: CudaSlice<bf16>,
    rank: usize,
}

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
    f32_pool: F32Pool,
    /// Stream layers instead of keeping them resident. Needed to run the merged
    /// BF16 checkpoint (51 GiB) on a 24 GiB card; the shipped AWQ path keeps
    /// every packed layer resident instead.
    cache_weights: AtomicBool,
    /// Keep the embedding table resident on the GPU and gather rows on-device.
    gpu_embed: AtomicBool,
    embedding_table: Mutex<Option<CudaSlice<bf16>>>,
    /// Optional PEFT adapter (mmap) and its alpha/rank scale.
    lora_store: Option<Store>,
    lora_scale: f32,
    /// lm_head LoRA A [rank, hidden]; B rows are read from `lora_store`.
    lora_head: Option<LoraHead>,
    /// Use the fused in-shared-memory dequant + tensor-core GEMM. The
    /// dequant-to-BF16 + cuBLAS path remains available as the reference.
    fused_w4a16: AtomicBool,
    /// Minimum N for the fused kernel; smaller modules use dequant+cuBLAS.
    fused_min_n: std::sync::atomic::AtomicUsize,
    /// Above this M, use dequant+cuBLAS: the fused kernel re-dequantizes W per
    /// M tile, so large prefills are better served by the reference path.
    fused_max_m: std::sync::atomic::AtomicUsize,
    /// Experimental 128x128 tile (slower on this GPU at the tested shapes).
    fused_tile128: AtomicBool,
    /// Use the vendored Marlin W4A16 kernel/layout instead of dequant+cuBLAS.
    marlin: AtomicBool,
    /// Experimental W4A8-FP8 repack/dispatch (default off).
    marlin_fp8: AtomicBool,
    /// Experimental W4A8-INT8 repack/dispatch (default off).
    marlin_int8: AtomicBool,
    /// Abs-max clip factor for per-token INT8 activation scales
    /// (`--int8-clip`, default 1.0 = no clipping). F32 bits in an AtomicU32.
    int8_clip: std::sync::atomic::AtomicU32,
    /// Use per-K-group INT8 activation scales in the Marlin W4A8 path.
    int8_group_scale: AtomicBool,
    /// Modules that keep per-token scales when group scales are enabled.
    int8_group_exclude: Mutex<Vec<String>>,
    /// Module-name substrings kept on W4A16 activations under `marlin_int8`.
    int8_exclude: Mutex<Vec<String>>,
    /// Layers that use all-linear INT8 (overrides `int8_exclude` inside them).
    int8_all_layers: Mutex<Vec<usize>>,
    /// Marlin split-K barrier workspace (one int per SM), zeroed once.
    marlin_locks: Mutex<Option<CudaSlice<i32>>>,
    int8_scratch: Mutex<Option<Int8Scratch>>,
    /// Warp-per-query attention (fast for long contexts); the original block
    /// kernel remains selectable.
    warp_attention: AtomicBool,
    /// Vendored FlashInfer FA2 prefill (single-request, hd256, 24Q/4KV).
    flashinfer: Option<Arc<crate::flashinfer::FlashInfer>>,
    flashinfer_enabled: AtomicBool,
    flashinfer_min_seq: std::sync::atomic::AtomicUsize,
    prof: Prof,
    cpu_prof: CpuProf,
    gdn_prof: Mutex<GdnProf>,
    /// Use the shared-state GDN recurrence (default) and its value-column split.
    gdn_shared: AtomicBool,
    gdn_chunks: std::sync::atomic::AtomicUsize,
    gdn_split: AtomicBool,
    /// Experimental FLA-style chunked prefill (default off until gates pass).
    gdn_chunked: AtomicBool,
    /// Tensor-core (TF32 WMMA) variant of the chunked prefill.
    gdn_chunked_tc: AtomicBool,
    /// Minimum sequence length for the chunked GDN path (default 64; shorter
    /// prompts keep the accurate shared recurrence).
    gdn_chunked_min_seq: std::sync::atomic::AtomicUsize,
    /// Time-tiled causal conv (default); `--legacy-conv` selects the original
    /// channel-only kernel.
    conv_tiled: AtomicBool,
    /// Below this sequence length the accurate single-column shared GDN
    /// recurrence is used; the split reduction is only enabled for long
    /// prefills where its f32 reordering cannot affect decision gates.
    gdn_split_min_seq: std::sync::atomic::AtomicUsize,
}

const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<Engine>();
};

/// Flattened multi-sequence batch: sequences are row ranges
/// `[offsets[s], offsets[s] + lens[s])` of one `[total, ...]` buffer.
struct BatchInfo {
    offsets: Vec<usize>,
    lens: Vec<usize>,
    total: usize,
    offsets_dev: CudaSlice<i32>,
    lens_dev: CudaSlice<i32>,
}

impl BatchInfo {
    fn new(gpu: &Gpu, tokens: &[&[u32]]) -> Result<Self> {
        if tokens.is_empty() {
            bail!("empty batch");
        }
        let mut offsets = Vec::with_capacity(tokens.len());
        let mut lens = Vec::with_capacity(tokens.len());
        let mut total = 0usize;
        for (i, t) in tokens.iter().enumerate() {
            if t.is_empty() {
                bail!("batch item {i} has no tokens");
            }
            offsets.push(total);
            lens.push(t.len());
            total += t.len();
        }
        let off_i: Vec<i32> = offsets.iter().map(|&v| v as i32).collect();
        let len_i: Vec<i32> = lens.iter().map(|&v| v as i32).collect();
        Ok(Self {
            offsets,
            lens,
            total,
            offsets_dev: gpu.stream.clone_htod(&off_i)?,
            lens_dev: gpu.stream.clone_htod(&len_i)?,
        })
    }

    fn batch(&self) -> usize {
        self.lens.len()
    }

    /// Per-row position inside its own sequence (RoPE resets per sequence).
    fn positions(&self) -> Vec<usize> {
        let mut pos = Vec::with_capacity(self.total);
        for s in 0..self.batch() {
            for t in 0..self.lens[s] {
                pos.push(t);
            }
        }
        pos
    }
}

impl Engine {
    pub fn new(model_dir: &std::path::Path) -> Result<Self> {
        let cfg = Config::from_dir(model_dir)?;
        let store = Store::open(model_dir)?;
        let gpu = Gpu::new(0)?;
        let kernels = Kernels::load(&gpu)?;
        let flashinfer = crate::flashinfer::FlashInfer::load();
        let flashinfer_enabled = flashinfer.is_some()
            && std::env::var("DECIDRA_FLASHINFER_ATTENTION")
                .map(|v| v != "0")
                .unwrap_or(true);
        let flashinfer_min_seq = std::env::var("DECIDRA_FLASHINFER_MIN_SEQ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024usize);
        let weight_cache = Mutex::new(vec![None; cfg.num_hidden_layers]);
        let mut engine = Self {
            gpu,
            cfg,
            store,
            kernels,
            weight_cache,
            dump_dir: Mutex::new(None),
            dequant_ws: Mutex::new(None),
            logits_opts: Mutex::new(LogitsOpts::default()),
            pool: Arc::new(PoolState {
                free: Mutex::new(HashMap::new()),
                low_mem: AtomicBool::new(false),
            }),
            f32_pool: Arc::new(F32PoolState {
                free: Mutex::new(HashMap::new()),
                low_mem: AtomicBool::new(false),
            }),
            cache_weights: AtomicBool::new(true),
            gpu_embed: AtomicBool::new(false),
            embedding_table: Mutex::new(None),
            lora_store: None,
            lora_scale: 1.0,
            lora_head: None,
            fused_w4a16: AtomicBool::new(false),
            fused_min_n: std::sync::atomic::AtomicUsize::new(5120),
            fused_max_m: std::sync::atomic::AtomicUsize::new(128),
            fused_tile128: AtomicBool::new(false),
            marlin: AtomicBool::new(false),
            marlin_fp8: AtomicBool::new(false),
            marlin_int8: AtomicBool::new(false),
            int8_clip: std::sync::atomic::AtomicU32::new(1.0f32.to_bits()),
            int8_group_scale: AtomicBool::new(false),
            int8_group_exclude: Mutex::new(Vec::new()),
            int8_exclude: Mutex::new(Vec::new()),
            int8_all_layers: Mutex::new(Vec::new()),
            marlin_locks: Mutex::new(None),
            int8_scratch: Mutex::new(None),
            warp_attention: AtomicBool::new(true),
            flashinfer,
            flashinfer_enabled: AtomicBool::new(flashinfer_enabled),
            flashinfer_min_seq: std::sync::atomic::AtomicUsize::new(flashinfer_min_seq),
            prof: Prof::new(),
            cpu_prof: CpuProf::new(),
            gdn_prof: Mutex::new(GdnProf::default()),
            gdn_shared: AtomicBool::new(true),
            gdn_chunks: std::sync::atomic::AtomicUsize::new(2),
            gdn_split: AtomicBool::new(true),
            gdn_chunked: AtomicBool::new(false),
            gdn_chunked_tc: AtomicBool::new(false),
            gdn_chunked_min_seq: std::sync::atomic::AtomicUsize::new(64),
            conv_tiled: AtomicBool::new(true),
            gdn_split_min_seq: std::sync::atomic::AtomicUsize::new(512),
        };
        if let Ok(path) = std::env::var("DECIDRA_LORA_ADAPTER") {
            if !path.is_empty() {
                engine.set_lora(std::path::Path::new(&path))?;
            }
        }
        Ok(engine)
    }

    /// Attach a PEFT LoRA adapter directory (or `adapter_model.safetensors`).
    pub fn set_lora(&mut self, path: &std::path::Path) -> Result<()> {
        let (model, dir) = if path.is_dir() {
            (path.join("adapter_model.safetensors"), path.to_path_buf())
        } else {
            (
                path.to_path_buf(),
                path.parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .to_path_buf(),
            )
        };
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("adapter_config.json"))
                .with_context(|| format!("reading adapter_config.json in {}", dir.display()))?,
        )
        .context("parsing adapter_config.json")?;
        let rank = cfg
            .get("r")
            .and_then(|v| v.as_u64())
            .context("adapter_config.json has no r")? as usize;
        let alpha = cfg
            .get("lora_alpha")
            .and_then(|v| v.as_f64())
            .unwrap_or(rank as f64) as f32;
        let store = Store::open_file(&model)?;
        let a_name = "base_model.model.lm_head.lora_A.weight";
        if !store.contains(a_name) {
            bail!("LoRA adapter has no lm_head A tensor");
        }
        self.lora_scale = alpha / rank as f32;
        self.lora_head = Some(LoraHead {
            a: self.gpu.stream.clone_htod(&store.bf16(a_name)?)?,
            rank,
        });
        self.lora_store = Some(store);
        Ok(())
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

    pub fn set_fused_tile128(&self, enabled: bool) {
        self.fused_tile128.store(enabled, Ordering::Relaxed);
    }

    pub fn set_warp_attention(&self, enabled: bool) {
        self.warp_attention.store(enabled, Ordering::Relaxed);
    }

    pub fn set_flashinfer_attention(&self, enabled: bool) {
        self.flashinfer_enabled.store(enabled, Ordering::Relaxed);
    }

    pub fn set_flashinfer_min_seq(&self, seq: usize) {
        self.flashinfer_min_seq.store(seq, Ordering::Relaxed);
    }

    fn flashinfer_for(
        &self,
        seq: usize,
        hd: usize,
        heads: usize,
        kv: usize,
    ) -> Option<&crate::flashinfer::FlashInfer> {
        if !self.flashinfer_enabled.load(Ordering::Relaxed)
            || seq < self.flashinfer_min_seq.load(Ordering::Relaxed)
            || hd != 256
            || heads != 24
            || kv != 4
        {
            return None;
        }
        self.flashinfer.as_deref()
    }

    pub fn set_gdn_shared(&self, enabled: bool) {
        self.gdn_shared.store(enabled, Ordering::Relaxed);
    }

    pub fn set_gdn_split(&self, enabled: bool) {
        self.gdn_split.store(enabled, Ordering::Relaxed);
    }

    pub fn set_gdn_split_min_seq(&self, seq: usize) {
        self.gdn_split_min_seq.store(seq, Ordering::Relaxed);
    }

    pub fn set_gdn_chunks(&self, chunks: usize) {
        self.gdn_chunks.store(chunks, Ordering::Relaxed);
    }

    pub fn set_gdn_chunked(&self, enabled: bool) {
        self.gdn_chunked.store(enabled, Ordering::Relaxed);
    }

    pub fn set_gdn_chunked_tc(&self, enabled: bool) {
        self.gdn_chunked_tc.store(enabled, Ordering::Relaxed);
    }

    pub fn set_gdn_chunked_min_seq(&self, seq: usize) {
        self.gdn_chunked_min_seq.store(seq, Ordering::Relaxed);
    }

    pub fn set_conv_tiled(&self, enabled: bool) {
        self.conv_tiled.store(enabled, Ordering::Relaxed);
    }

    pub fn set_marlin(&self, enabled: bool) {
        self.marlin.store(enabled, Ordering::Relaxed);
    }

    pub fn set_marlin_fp8(&self, enabled: bool) {
        self.marlin_fp8.store(enabled, Ordering::Relaxed);
    }

    pub fn set_marlin_int8(&self, enabled: bool) {
        self.marlin_int8.store(enabled, Ordering::Relaxed);
    }

    /// Clip per-token INT8 activation scales at `clip * absmax` (1.0 = off).
    pub fn set_int8_clip(&self, clip: f32) {
        self.int8_clip
            .store(clip.clamp(0.01, 1.0).to_bits(), Ordering::Relaxed);
    }

    /// Use per-K-group (64) INT8 activation scales in the Marlin W4A8 path.
    pub fn set_int8_group_scale(&self, enabled: bool) {
        self.int8_group_scale.store(enabled, Ordering::Relaxed);
    }

    /// Modules that keep per-token INT8 scales when group scales are enabled.
    /// Must be set before `preload_all`, because the choice is fixed at repack.
    pub fn set_int8_group_exclude(&self, exclude: Vec<String>) {
        *self.int8_group_exclude.lock().unwrap() = exclude;
    }

    fn int8_clip(&self) -> f32 {
        f32::from_bits(self.int8_clip.load(Ordering::Relaxed))
    }

    /// Keep modules whose names contain any of these substrings on W4A16
    /// activations when INT8 is enabled (must be set before preload_all).
    pub fn set_int8_exclude(&self, exclude: Vec<String>) {
        *self.int8_exclude.lock().unwrap() = exclude;
    }

    /// Select layers that run all-linear INT8. Spec: `first:N`, `last:N`,
    /// `alt:N` (N evenly spaced layers), or a comma-separated list.
    pub fn set_int8_all_layers(&self, spec: &str) {
        let l = self.cfg.num_hidden_layers;
        let spec = spec.trim();
        let mut v: Vec<usize> = if let Some(n) = spec.strip_prefix("first:") {
            let n = n.trim().parse::<usize>().unwrap_or(0).min(l);
            (0..n).collect()
        } else if let Some(n) = spec.strip_prefix("last:") {
            let n = n.trim().parse::<usize>().unwrap_or(0).min(l);
            (l - n..l).collect()
        } else if let Some(n) = spec.strip_prefix("alt:") {
            let n = n.trim().parse::<usize>().unwrap_or(0).min(l);
            if n == 0 {
                Vec::new()
            } else {
                (0..n).map(|i| i * l / n).collect()
            }
        } else {
            spec.split(',')
                .filter_map(|x| x.trim().parse::<usize>().ok())
                .filter(|&i| i < l)
                .collect()
        };
        v.sort_unstable();
        v.dedup();
        *self.int8_all_layers.lock().unwrap() = v;
    }

    fn prof_begin(&self, _name: &'static str) -> Option<CudaEvent> {
        if !self.prof.enabled {
            return None;
        }
        let ev = self
            .gpu
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .ok()?;
        ev.record(&self.gpu.stream).ok()?;
        Some(ev)
    }

    fn prof_end(&self, name: &'static str, start: Option<CudaEvent>) {
        if let Some(start) = start {
            if let Ok(end) = self
                .gpu
                .ctx
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            {
                if end.record(&self.gpu.stream).is_ok() {
                    self.prof
                        .entries
                        .lock()
                        .unwrap()
                        .push(ProfEntry { name, start, end });
                }
            }
        }
    }

    fn prof_report(&self) {
        self.cpu_prof.report();
        {
            let g = std::mem::take(&mut *self.gdn_prof.lock().unwrap());
            if g.blocks > 0 && g.chunks > 0 {
                let denom = (g.blocks * g.chunks) as f64;
                eprintln!("--- gdn phase profile (avg cycles/chunk/block) ---");
                for (p, name) in GDN_PHASES.iter().enumerate() {
                    let c = g.cycles[p] as f64 / denom;
                    eprintln!("{name:10} {c:9.0} cycles  {:6.1} us", c / 2.91e3);
                }
            }
        }
        if !self.prof.enabled {
            return;
        }
        let entries = std::mem::take(&mut *self.prof.entries.lock().unwrap());
        let mut sums: std::collections::HashMap<&'static str, (f64, usize)> =
            std::collections::HashMap::new();
        for e in &entries {
            if let Ok(ms) = e.start.elapsed_ms(&e.end) {
                let v = sums.entry(e.name).or_insert((0.0, 0));
                v.0 += ms as f64;
                v.1 += 1;
            }
        }
        let total: f64 = sums.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = sums.into_iter().collect();
        rows.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
        eprintln!("--- profile (sum of regions {total:.2} ms) ---");
        for (name, (ms, n)) in rows {
            eprintln!(
                "{name:16} {ms:9.3} ms  {:6.2}%  n={n}  {:.3} ms/op",
                ms / total * 100.0,
                ms / n as f64
            );
        }
    }

    fn provider(&self) -> Result<WeightProvider<'_>> {
        let mut p = WeightProvider::new(&self.store, &self.gpu, &self.kernels)?;
        p.marlin = self.marlin.load(Ordering::Relaxed);
        p.marlin_fp8 = self.marlin_fp8.load(Ordering::Relaxed);
        p.marlin_int8 = self.marlin_int8.load(Ordering::Relaxed);
        p.int8_exclude = self.int8_exclude.lock().unwrap().clone();
        p.int8_group_scale = self.int8_group_scale.load(Ordering::Relaxed);
        p.int8_group_exclude = self.int8_group_exclude.lock().unwrap().clone();
        p.int8_all_layers = self.int8_all_layers.lock().unwrap().clone();
        p.lora_store = self.lora_store.as_ref();
        p.lora_scale = self.lora_scale;
        Ok(p)
    }

    pub fn set_cache_weights(&self, enabled: bool) {
        self.cache_weights.store(enabled, Ordering::Relaxed);
    }

    /// Keep the embedding table resident on the GPU (set before `preload_all`).
    pub fn set_gpu_embed(&self, enabled: bool) {
        self.gpu_embed.store(enabled, Ordering::Relaxed);
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
        // Upload the embedding table first: a 2.4 GiB contiguous block is easy
        // to place before the packed decoder weights fragment the pool.
        if self.gpu_embed.load(Ordering::Relaxed) {
            let name = self.provider()?.embedding();
            match self.store.bf16_slice(&name) {
                Ok(host) => {
                    eprintln!(
                        "gpu embedding: {:.2} GiB resident",
                        (host.len() * 2) as f64 / (1u64 << 30) as f64
                    );
                    let dev = self.gpu.stream.clone_htod(host)?;
                    *self.embedding_table.lock().unwrap() = Some(dev);
                }
                Err(error) => eprintln!("gpu embedding disabled: {error}"),
            }
        } else {
            eprintln!("gpu embedding: flag off");
        }
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
        let prov = self.provider()?;
        let w = Arc::new(prov.layer(idx, self.cfg.layer_types[idx])?);
        if self.cache_weights.load(Ordering::Relaxed) {
            cache[idx] = Some(w.clone());
        }
        Ok(w)
    }

    fn alloc_bf16(&self, n: usize) -> Result<Buf> {
        {
            let mut pool = self.pool.free.lock().unwrap();
            if pool.len() > 128 {
                pool.clear();
            }
        }
        if let Some(v) = self.pool.free.lock().unwrap().get_mut(&n) {
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

    /// Reusable f32 scratch; contents are undefined. GDN q/k/v/g/beta are
    /// completely overwritten by `gdn_prepare` before use.
    fn alloc_f32(&self, n: usize) -> Result<F32Buf> {
        {
            let mut pool = self.f32_pool.free.lock().unwrap();
            if pool.len() > 128 {
                pool.clear();
            }
        }
        if let Some(v) = self.f32_pool.free.lock().unwrap().get_mut(&n) {
            if let Some(s) = v.pop() {
                return Ok(F32Buf {
                    slice: Some(s),
                    pool: self.f32_pool.clone(),
                });
            }
        }
        let s = unsafe { self.gpu.stream.alloc::<f32>(n)? };
        Ok(F32Buf {
            slice: Some(s),
            pool: self.f32_pool.clone(),
        })
    }

    /// Reusable f32 scratch, explicitly zeroed (used for the GDN state).
    fn alloc_f32_zeros(&self, n: usize) -> Result<F32Buf> {
        let mut b = self.alloc_f32(n)?;
        self.gpu.stream.memset_zeros(&mut *b)?;
        Ok(b)
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Buf> {
        let h = self.cfg.hidden_size;
        for &tok in tokens {
            if tok as usize >= self.cfg.vocab_size {
                bail!(
                    "token id {tok} out of range (vocab {})",
                    self.cfg.vocab_size
                );
            }
        }
        if self.gpu_embed.load(Ordering::Relaxed) {
            let table = self.embedding_table.lock().unwrap();
            if let Some(table) = table.as_ref() {
                let ids = self.gpu.stream.clone_htod(tokens)?;
                let mut buf = self.alloc_bf16(tokens.len() * h)?;
                kernels::embed_gather(
                    &self.gpu,
                    &self.kernels.embed_gather,
                    table,
                    &ids,
                    &mut buf,
                    tokens.len(),
                    h,
                )?;
                return Ok(buf);
            }
        }
        let name = self.provider()?.embedding();
        let mut host = vec![bf16::ZERO; tokens.len() * h];
        // The gather is host-side mmap traffic and is the only serial CPU step
        // in a long prefill (23 ms at 4K, ~95 ms at 16K). Rows are independent,
        // so split them across cores; errors are recorded once and re-raised
        // after the scope.
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(8)
            .min(tokens.len().max(1));
        if workers > 1 && tokens.len() >= 64 {
            let rows = tokens.len().div_ceil(workers);
            let err = Mutex::new(None);
            std::thread::scope(|s| {
                for (ci, out) in host.chunks_mut(rows * h).enumerate() {
                    let start = ci * rows;
                    let n = out.len() / h;
                    let store = &self.store;
                    let name = name.as_str();
                    let err = &err;
                    s.spawn(move || {
                        for t in 0..n {
                            if let Err(e) = store.bf16_row(
                                name,
                                tokens[start + t] as usize,
                                &mut out[t * h..(t + 1) * h],
                            ) {
                                *err.lock().unwrap() = Some(e);
                                return;
                            }
                        }
                    });
                }
            });
            if let Some(e) = err.into_inner().unwrap() {
                return Err(e);
            }
        } else {
            for (t, &tok) in tokens.iter().enumerate() {
                self.store
                    .bf16_row(&name, tok as usize, &mut host[t * h..(t + 1) * h])
                    .with_context(|| format!("embedding row {tok}"))?;
            }
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
        self.prof.entries.lock().unwrap().clear();
        let x = timed!(self, "embed", self.embed(tokens)?);
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
        self.prof.entries.lock().unwrap().clear();
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
        let low_mem = seq >= 12_288;
        let was_low = self.pool.low_mem.swap(low_mem, Ordering::Relaxed);
        let was_low_f32 = self.f32_pool.low_mem.swap(low_mem, Ordering::Relaxed);
        if low_mem && !was_low {
            self.pool.free.lock().unwrap().clear();
        }
        if low_mem && !was_low_f32 {
            self.f32_pool.free.lock().unwrap().clear();
        }
        let mut x = x;
        for i in start..=last {
            let w = timed!(self, "weights", self.layer_weights(i)?);
            let mut captured = None;
            self.dump_dev("00_layer_input", &x, seq, self.cfg.hidden_size)?;
            let next = self.run_layer(&x, seq, &w, i == last, stage, &mut captured)?;
            if i == last {
                self.dump_dev("06_layer_output", &next, seq, self.cfg.hidden_size)?;
            }
            if let Some(v) = captured {
                self.prof_report();
                return self.to_f32(&v, seq * self.cfg.hidden_size);
            }
            x = next;
        }
        let result = match stage {
            Stage::FinalNorm | Stage::Logits => {
                let norm = self.provider()?.final_norm()?;
                let mut h = self.alloc_bf16(seq * self.cfg.hidden_size)?;
                timed!(
                    self,
                    "norm.final",
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
                    )?
                );
                if stage == Stage::FinalNorm {
                    self.slice_out(&h, seq, self.cfg.hidden_size, last_only)
                } else {
                    let opts = self.logits_opts.lock().unwrap().clone();
                    if let Some(ids) = opts.head_ids {
                        self.logits_rows(&h, seq, last_only, &ids)
                    } else if opts.full {
                        self.logits_chunked(&h, seq, last_only)
                    } else {
                        bail!("--stage logits requires --head-ids <ids> or --full-logits")
                    }
                }
            }
            _ => self.slice_out(&x, seq, self.cfg.hidden_size, last_only),
        };
        self.prof_report();
        result
    }

    // -----------------------------------------------------------------
    // Batched inference
    // -----------------------------------------------------------------

    /// Validate a batch request and flatten token ids.
    fn batch_tokens(&self, tokens: &[&[u32]]) -> Result<Vec<u32>> {
        if tokens.is_empty() {
            bail!("empty batch");
        }
        let mut flat = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            if t.is_empty() {
                bail!("batch item {i} has no tokens");
            }
            flat.extend_from_slice(t);
        }
        Ok(flat)
    }

    fn rope_tables_positions(
        &self,
        positions: &[usize],
    ) -> Result<(CudaSlice<bf16>, CudaSlice<bf16>)> {
        let dim = self.cfg.rotary_dim();
        let half = dim / 2;
        let theta = self.cfg.rope_theta;
        let mut inv = vec![0f32; half];
        for (i, v) in inv.iter_mut().enumerate() {
            *v = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
        }
        let mut cos = vec![bf16::ZERO; positions.len() * half];
        let mut sin = vec![bf16::ZERO; positions.len() * half];
        for (row, &pos) in positions.iter().enumerate() {
            for i in 0..half {
                let ang = pos as f32 * inv[i];
                cos[row * half + i] = bf16::from_f32(ang.cos());
                sin[row * half + i] = bf16::from_f32(ang.sin());
            }
        }
        Ok((
            self.gpu.stream.clone_htod(&cos)?,
            self.gpu.stream.clone_htod(&sin)?,
        ))
    }

    /// Final RMSNorm over `rows` flattened rows.
    fn final_norm_rows(&self, x: &CudaSlice<bf16>, rows: usize) -> Result<Buf> {
        let hidden = self.cfg.hidden_size;
        let norm = self.provider()?.final_norm()?;
        let mut h = self.alloc_bf16(rows * hidden)?;
        timed!(
            self,
            "norm.final",
            kernels::rmsnorm(
                &self.gpu,
                &self.kernels.rmsnorm,
                x,
                &norm,
                &mut h,
                rows,
                hidden,
                self.cfg.rms_norm_eps,
                true,
            )?
        );
        Ok(h)
    }

    /// Device-side indices of each sequence's last row.
    fn last_row_indices(&self, b: &BatchInfo) -> Result<CudaSlice<i32>> {
        let idx: Vec<i32> = (0..b.batch())
            .map(|s| (b.offsets[s] + b.lens[s] - 1) as i32)
            .collect();
        Ok(self.gpu.stream.clone_htod(&idx)?)
    }

    fn gather_last_rows(&self, h: &CudaSlice<bf16>, b: &BatchInfo, cols: usize) -> Result<Buf> {
        let idx = self.last_row_indices(b)?;
        let mut out = self.alloc_bf16(b.batch() * cols)?;
        kernels::gather_rows(
            &self.gpu,
            &self.kernels.gather_rows,
            h,
            &idx,
            &mut out,
            b.batch(),
            cols,
        )?;
        Ok(out)
    }

    fn vecs_from_flat(flat: Vec<f32>, n: usize, cols: usize) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(flat[i * cols..(i + 1) * cols].to_vec());
        }
        out
    }

    /// Full batched text forward, returning post-final-norm hidden states over
    /// all flattened rows plus the batch layout.
    fn batch_forward(&self, tokens: &[&[u32]]) -> Result<(Buf, BatchInfo)> {
        let flat = self.batch_tokens(tokens)?;
        let b = BatchInfo::new(&self.gpu, tokens)?;
        self.prof.entries.lock().unwrap().clear();
        let total = b.total;
        let low_mem = total >= 12_288;
        let was_low = self.pool.low_mem.swap(low_mem, Ordering::Relaxed);
        let was_low_f32 = self.f32_pool.low_mem.swap(low_mem, Ordering::Relaxed);
        if low_mem && !was_low {
            self.pool.free.lock().unwrap().clear();
        }
        if low_mem && !was_low_f32 {
            self.f32_pool.free.lock().unwrap().clear();
        }
        let mut x = timed!(self, "embed", self.embed(&flat)?);
        for i in 0..self.cfg.num_hidden_layers {
            let w = self.layer_weights(i)?;
            x = self.run_layer_batch(&x, &b, &w)?;
        }
        let h = self.final_norm_rows(&x, total)?;
        Ok((h, b))
    }

    fn run_layer_batch(&self, x: &CudaSlice<bf16>, b: &BatchInfo, w: &LayerWeights) -> Result<Buf> {
        let hidden = self.cfg.hidden_size;
        let total = b.total;
        let mut h = self.alloc_bf16(total * hidden)?;
        timed!(
            self,
            "norm.input",
            kernels::rmsnorm(
                &self.gpu,
                &self.kernels.rmsnorm,
                x,
                &w.input_norm,
                &mut h,
                total,
                hidden,
                self.cfg.rms_norm_eps,
                true,
            )?
        );
        let mix = match &w.mixer {
            MixerWeights::Full(a) => self.full_attention_batch(&h, b, a)?,
            MixerWeights::Gdn(a) => self.gdn_batch(&h, b, a)?,
        };
        let mut x1 = self.alloc_bf16(total * hidden)?;
        timed!(
            self,
            "add.residual",
            kernels::add(
                &self.gpu,
                &self.kernels.add,
                x,
                &mix,
                &mut x1,
                total * hidden
            )?
        );
        let mut h2 = self.alloc_bf16(total * hidden)?;
        timed!(
            self,
            "norm.post",
            kernels::rmsnorm(
                &self.gpu,
                &self.kernels.rmsnorm,
                &x1,
                &w.post_norm,
                &mut h2,
                total,
                hidden,
                self.cfg.rms_norm_eps,
                true,
            )?
        );
        let mlp = self.mlp(&h2, total, w)?;
        let mut out = self.alloc_bf16(total * hidden)?;
        timed!(
            self,
            "add.residual",
            kernels::add(
                &self.gpu,
                &self.kernels.add,
                &x1,
                &mlp,
                &mut out,
                total * hidden
            )?
        );
        Ok(out)
    }

    fn full_attention_batch(
        &self,
        h: &CudaSlice<bf16>,
        b: &BatchInfo,
        w: &FullAttentionWeights,
    ) -> Result<Buf> {
        let cfg = &self.cfg;
        let (heads, kv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let total = b.total;
        let qg = timed!(self, "gemm.q", self.gemm(h, total, &w.q_proj)?);
        let kk = timed!(self, "gemm.k", self.gemm(h, total, &w.k_proj)?);
        let vv = timed!(self, "gemm.v", self.gemm(h, total, &w.v_proj)?);

        let (q, gate) = if cfg.attn_output_gate {
            let mut q = self.alloc_bf16(total * heads * hd)?;
            let mut gate = self.alloc_bf16(total * heads * hd)?;
            timed!(
                self,
                "attn.split",
                kernels::split_q_gate(
                    &self.gpu,
                    &self.kernels.split_q_gate,
                    &qg,
                    &mut q,
                    &mut gate,
                    total,
                    heads,
                    hd,
                )?
            );
            drop(qg);
            (q, Some(gate))
        } else {
            (qg, None)
        };

        let mut qn = self.alloc_bf16(total * heads * hd)?;
        timed!(
            self,
            "norm.q",
            kernels::rmsnorm(
                &self.gpu,
                &self.kernels.rmsnorm,
                &q,
                &w.q_norm,
                &mut qn,
                total * heads,
                hd,
                cfg.rms_norm_eps,
                true,
            )?
        );
        let mut kn = self.alloc_bf16(total * kv * hd)?;
        timed!(
            self,
            "norm.k",
            kernels::rmsnorm(
                &self.gpu,
                &self.kernels.rmsnorm,
                &kk,
                &w.k_norm,
                &mut kn,
                total * kv,
                hd,
                cfg.rms_norm_eps,
                true,
            )?
        );

        let rd = cfg.rotary_dim();
        let positions = b.positions();
        let (cos, sin) = self.rope_tables_positions(&positions)?;
        timed!(
            self,
            "rope",
            kernels::rope(
                &self.gpu,
                &self.kernels.rope,
                &mut qn,
                &mut kn,
                &cos,
                &sin,
                total,
                heads,
                kv,
                hd,
                rd / 2,
            )?
        );
        drop(cos);
        drop(sin);

        // Attention resets at every sequence boundary; one launch per sequence
        // (attention is a small fraction of the forward).
        let mut attn = self.alloc_bf16(total * heads * hd)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let tc = match hd {
            64 => Some(&self.kernels.attention_tc_64),
            128 => Some(&self.kernels.attention_tc_128),
            256 => Some(&self.kernels.attention_tc_256),
            _ => None,
        };
        let hv = heads * hd;
        let kvv = kv * hd;
        let use_warp = self.warp_attention.load(Ordering::Relaxed);
        let __attn_t = self.prof_begin("attn");
        for s in 0..b.batch() {
            let o = b.offsets[s];
            let l = b.lens[s];
            let qv = qn
                .try_slice(o * hv..(o + l) * hv)
                .context("attention q view")?;
            let kvw = kn
                .try_slice(o * kvv..(o + l) * kvv)
                .context("attention k view")?;
            let vvw = vv
                .try_slice(o * kvv..(o + l) * kvv)
                .context("attention v view")?;
            let mut ov = attn
                .try_slice_mut(o * hv..(o + l) * hv)
                .context("attention out view")?;
            let fi = if b.batch() == 1 {
                self.flashinfer_for(l, hd, heads, kv)
            } else {
                None
            };
            if let Some(fi) = fi {
                fi.prefill(&self.gpu, &qv, &kvw, &vvw, &mut ov, l)?;
            } else {
                match (use_warp, tc) {
                    (true, Some(tc)) => kernels::attention_tc(
                        &self.gpu, tc, &qv, &kvw, &vvw, &mut ov, l, heads, kv, hd, scale,
                    )?,
                    _ => kernels::attention_warp(
                        &self.gpu,
                        &self.kernels.attention_warp,
                        &qv,
                        &kvw,
                        &vvw,
                        &mut ov,
                        l,
                        heads,
                        kv,
                        hd,
                        scale,
                    )?,
                }
            }
        }
        self.prof_end("attn", __attn_t);
        drop(qn);
        drop(kn);
        drop(vv);

        let gated = match gate {
            Some(g) => {
                let mut o = self.alloc_bf16(total * heads * hd)?;
                timed!(
                    self,
                    "attn.gate",
                    kernels::sigmoid_gate(
                        &self.gpu,
                        &self.kernels.sigmoid_gate,
                        &attn,
                        &g,
                        &mut o,
                        total * heads * hd,
                    )?
                );
                drop(attn);
                drop(g);
                o
            }
            None => attn,
        };
        Ok(timed!(self, "gemm.o", self.gemm(&gated, total, &w.o_proj)?))
    }

    fn gdn_batch(&self, h: &CudaSlice<bf16>, b: &BatchInfo, w: &GdnWeights) -> Result<Buf> {
        let cfg = &self.cfg;
        let (nk, nv) = (cfg.linear_num_key_heads, cfg.linear_num_value_heads);
        let (dk, dv) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
        let conv_dim = 2 * cfg.gdn_key_dim() + cfg.gdn_value_dim();
        let total = b.total;

        let qkv = timed!(self, "gemm.in_qkv", self.gemm(h, total, &w.in_proj_qkv)?);
        let mut conv = self.alloc_bf16(total * conv_dim)?;
        let max_len = b.lens.iter().copied().max().unwrap_or(0);
        timed!(
            self,
            "gdn.conv",
            if self.conv_tiled.load(Ordering::Relaxed) {
                kernels::conv1d_silu_tiled_batched(
                    &self.gpu,
                    &self.kernels.conv1d_silu_tiled_batched,
                    &qkv,
                    &w.conv1d,
                    &mut conv,
                    &b.offsets_dev,
                    &b.lens_dev,
                    b.batch(),
                    conv_dim,
                    max_len,
                )?
            } else {
                kernels::conv1d_silu_batched(
                    &self.gpu,
                    &self.kernels.conv1d_silu_batched,
                    &qkv,
                    &w.conv1d,
                    &mut conv,
                    &b.offsets_dev,
                    &b.lens_dev,
                    b.batch(),
                    conv_dim,
                )?
            }
        );
        let z = timed!(self, "gemm.z", self.gemm(h, total, &w.in_proj_z)?);
        let bb = timed!(self, "gemm.b", self.gemm(h, total, &w.in_proj_b)?);
        let aa = timed!(self, "gemm.a", self.gemm(h, total, &w.in_proj_a)?);

        let mut qf = self.alloc_f32(total * nv * dk)?;
        let mut kf = self.alloc_f32(total * nv * dk)?;
        let mut vf = self.alloc_f32(total * nv * dv)?;
        let mut gg = self.alloc_f32(total * nv)?;
        let mut beta = self.alloc_f32(total * nv)?;
        timed!(
            self,
            "gdn.prepare",
            kernels::gdn_prepare(
                &self.gpu,
                &self.kernels.gdn_prepare,
                &conv,
                &bb,
                &aa,
                &w.a_log,
                &w.dt_bias,
                &mut qf,
                &mut kf,
                &mut vf,
                &mut gg,
                &mut beta,
                total,
                nk,
                nv,
                dk,
                dv,
            )?
        );
        drop(qkv);
        drop(conv);
        drop(bb);
        drop(aa);

        let mut state = self.alloc_f32_zeros(b.batch() * nv * dk * dv)?;
        let mut core = self.alloc_bf16(total * nv * dv)?;
        let min_seq = self.gdn_split_min_seq.load(Ordering::Relaxed);
        let all_long = b.lens.iter().all(|&l| l >= min_seq);
        let all_short = b.lens.iter().all(|&l| l < min_seq);
        let split = self.gdn_split.load(Ordering::Relaxed);
        let shared = self.gdn_shared.load(Ordering::Relaxed);
        let chunked_tc = self.gdn_chunked_tc.load(Ordering::Relaxed);
        let chunked = self.gdn_chunked.load(Ordering::Relaxed);
        let chunk_min = self.gdn_chunked_min_seq.load(Ordering::Relaxed);
        let chunked_any = (chunked || chunked_tc)
            && dk == 128
            && dv == 128
            && b.lens.iter().any(|&l| l >= chunk_min);
        // One batched TF32 launch when every sequence is long enough; mixed
        // batches keep the per-sequence loop so short items stay on the
        // accurate shared recurrence.
        let all_chunked =
            chunked_tc && dk == 128 && dv == 128 && b.lens.iter().all(|&l| l >= chunk_min);
        let dv_chunks = if dv >= 64 {
            self.gdn_chunks.load(Ordering::Relaxed).max(1).min(dv / 16)
        } else {
            1
        };
        timed!(
            self,
            "gdn.recurrent",
            if all_chunked {
                kernels::gdn_chunked_batched(
                    &self.gpu,
                    &self.kernels.gdn_chunked_tc_batched,
                    &*qf,
                    &*kf,
                    &*vf,
                    &*gg,
                    &*beta,
                    &mut *state,
                    &mut *core,
                    &b.offsets_dev,
                    &b.lens_dev,
                    b.batch(),
                    nv,
                    dk,
                    dv,
                )?
            } else if !chunked_any && split && all_long {
                kernels::gdn_recurrent_split_batched(
                    &self.gpu,
                    &self.kernels.gdn_recurrent_split_batched,
                    &qf,
                    &kf,
                    &vf,
                    &gg,
                    &beta,
                    &mut state,
                    &mut core,
                    &b.offsets_dev,
                    &b.lens_dev,
                    b.batch(),
                    nv,
                    dk,
                    dv,
                )?
            } else if !chunked_any && shared && all_short {
                kernels::gdn_recurrent_shared_batched(
                    &self.gpu,
                    &self.kernels.gdn_recurrent_shared_batched,
                    &qf,
                    &kf,
                    &vf,
                    &gg,
                    &beta,
                    &mut state,
                    &mut core,
                    &b.offsets_dev,
                    &b.lens_dev,
                    b.batch(),
                    nv,
                    dk,
                    dv,
                    dv_chunks,
                )?
            } else {
                // Mixed lengths: one launch per sequence so each uses the same
                // kernel the single-sequence path selects.
                for s in 0..b.batch() {
                    let o = b.offsets[s];
                    let l = b.lens[s];
                    let qv = qf
                        .try_slice(o * nv * dk..(o + l) * nv * dk)
                        .context("gdn q view")?;
                    let kvw = kf
                        .try_slice(o * nv * dk..(o + l) * nv * dk)
                        .context("gdn k view")?;
                    let vvw = vf
                        .try_slice(o * nv * dv..(o + l) * nv * dv)
                        .context("gdn v view")?;
                    let gw = gg.try_slice(o * nv..(o + l) * nv).context("gdn g view")?;
                    let bw = beta
                        .try_slice(o * nv..(o + l) * nv)
                        .context("gdn beta view")?;
                    let mut sv = state
                        .try_slice_mut(s * nv * dk * dv..(s + 1) * nv * dk * dv)
                        .context("gdn state view")?;
                    let mut ov = core
                        .try_slice_mut(o * nv * dv..(o + l) * nv * dv)
                        .context("gdn out view")?;
                    if (chunked || chunked_tc) && l >= chunk_min && dk == 128 && dv == 128 {
                        kernels::gdn_chunked(
                            &self.gpu,
                            if chunked_tc {
                                &self.kernels.gdn_chunked_tc
                            } else {
                                &self.kernels.gdn_chunked
                            },
                            &qv,
                            &kvw,
                            &vvw,
                            &gw,
                            &bw,
                            &mut sv,
                            &mut ov,
                            l,
                            nv,
                            dk,
                            dv,
                        )?;
                    } else if split && l >= min_seq {
                        kernels::gdn_recurrent_split(
                            &self.gpu,
                            &self.kernels.gdn_recurrent_split,
                            &qv,
                            &kvw,
                            &vvw,
                            &gw,
                            &bw,
                            &mut sv,
                            &mut ov,
                            l,
                            nv,
                            dk,
                            dv,
                        )?;
                    } else if shared {
                        kernels::gdn_recurrent_shared(
                            &self.gpu,
                            &self.kernels.gdn_recurrent_shared,
                            &qv,
                            &kvw,
                            &vvw,
                            &gw,
                            &bw,
                            &mut sv,
                            &mut ov,
                            l,
                            nv,
                            dk,
                            dv,
                            dv_chunks,
                        )?;
                    } else {
                        kernels::gdn_recurrent(
                            &self.gpu,
                            &self.kernels.gdn_recurrent,
                            &qv,
                            &kvw,
                            &vvw,
                            &gw,
                            &bw,
                            &mut sv,
                            &mut ov,
                            l,
                            nv,
                            dk,
                            dv,
                        )?;
                    }
                }
                Ok::<(), anyhow::Error>(())?
            }
        );
        drop(qf);
        drop(kf);
        drop(vf);
        drop(gg);
        drop(beta);

        let mut normed = self.alloc_bf16(total * nv * dv)?;
        timed!(
            self,
            "gdn.norm",
            kernels::rmsnorm_gated(
                &self.gpu,
                &self.kernels.rmsnorm_gated,
                &core,
                &z,
                &w.norm,
                &mut normed,
                total * nv,
                dv,
                cfg.rms_norm_eps,
            )?
        );
        drop(core);
        drop(z);
        Ok(timed!(
            self,
            "gemm.out",
            self.gemm(&normed, total, &w.out_proj)?
        ))
    }

    /// Batched last-token hidden state (`[hidden]` per item, post final norm).
    pub fn forward_last_hidden_batch(&self, tokens: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        let (h, b) = self.batch_forward(tokens)?;
        let hidden = self.cfg.hidden_size;
        let rows = self.gather_last_rows(&h, &b, hidden)?;
        let flat = self.to_f32(&rows, b.batch() * hidden)?;
        Ok(Self::vecs_from_flat(flat, b.batch(), hidden))
    }

    /// Batched decision logits at each sequence's last position.
    pub fn decision_logits_batch(
        &self,
        tokens: &[&[u32]],
        head_ids: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>> {
        if head_ids.len() != tokens.len() {
            bail!(
                "head_ids length {} does not match batch size {}",
                head_ids.len(),
                tokens.len()
            );
        }
        for (i, ids) in head_ids.iter().enumerate() {
            if ids.is_empty() {
                bail!("batch item {i} has no head ids");
            }
            for &id in ids.iter() {
                if id as usize >= self.cfg.vocab_size {
                    bail!("head id {id} out of range (vocab {})", self.cfg.vocab_size);
                }
            }
        }
        let (h, b) = self.batch_forward(tokens)?;
        let hidden = self.cfg.hidden_size;
        let total = b.total;
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); tokens.len()];
        // One head GEMM per distinct id set, always over all flattened rows so
        // a batch of one is bit-identical to the single-item path.
        let mut groups: HashMap<Vec<u32>, Vec<usize>> = HashMap::new();
        for (i, ids) in head_ids.iter().enumerate() {
            groups.entry(ids.to_vec()).or_default().push(i);
        }
        for (ids, items) in groups {
            let mut w_host = vec![bf16::ZERO; ids.len() * hidden];
            for (r, &id) in ids.iter().enumerate() {
                self.store.bf16_row(
                    "lm_head.weight",
                    id as usize,
                    &mut w_host[r * hidden..(r + 1) * hidden],
                )?;
            }
            let wd = self.gpu.stream.clone_htod(&w_host)?;
            let lw = LinearWeight::Bf16 {
                out: ids.len(),
                in_: hidden,
                w: wd,
                lora: None,
            };
            let logits = timed!(self, "head", self.gemm(&h, total, &lw)?);
            let mut logits = logits;
            self.add_head_lora(&h, total, &ids, &mut logits)?;
            let vals = self.to_f32(&logits, total * ids.len())?;
            drop(logits);
            for &i in &items {
                let row = b.offsets[i] + b.lens[i] - 1;
                out[i] = vals[row * ids.len()..(row + 1) * ids.len()].to_vec();
            }
        }
        self.prof_report();
        Ok(out)
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
        timed!(
            self,
            "norm.input",
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
            )?
        );
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
            return self.alloc_bf16(0);
        }

        let mut x1 = self.alloc_bf16(seq * hidden)?;
        timed!(
            self,
            "add.residual",
            kernels::add(&self.gpu, &self.kernels.add, x, &mix, &mut x1, seq * hidden)?
        );
        if capture {
            self.dump_dev("03_after_attn_residual", &x1, seq, hidden)?;
        }

        let mut h2 = self.alloc_bf16(seq * hidden)?;
        timed!(
            self,
            "norm.post",
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
            )?
        );
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
        let gate = timed!(self, "gemm.gate", self.gemm(h, seq, &w.gate_proj)?);
        let up = timed!(self, "gemm.up", self.gemm(h, seq, &w.up_proj)?);
        // When the down projection consumes INT8 group-quantized input, fuse
        // SiLU(gate)*up into the quantizer: no intermediate BF16 tensor and no
        // separate elementwise launch.
        if let LinearWeight::Marlin {
            qweight,
            scales,
            zeros,
            ..
        } = &w.down_proj
        {
            let (n, k) = (w.down_proj.out(), w.down_proj.in_());
            if w.down_proj.int8_a()
                && w.down_proj.group_a()
                && w.down_proj.lora().is_none()
                && k == inter
                && k % 64 == 0
            {
                let groups = k / 64;
                let need = seq * k;
                let mut scratch = self.int8_scratch.lock().unwrap();
                if scratch
                    .as_ref()
                    .map_or(true, |s| s.x.len() < need || s.scales.len() < seq * groups)
                {
                    *scratch = Some(Int8Scratch {
                        x: self.gpu.stream.alloc_zeros::<i8>(need)?,
                        scales: self.gpu.stream.alloc_zeros::<f32>(seq * groups)?,
                    });
                }
                let s = scratch.as_mut().unwrap();
                timed!(
                    self,
                    "mlp.act",
                    kernels::quant_int8_group_silu_bf16(
                        &self.gpu,
                        &self.kernels.quant_int8_group_silu_bf16,
                        &gate,
                        &up,
                        &mut s.x,
                        &mut s.scales,
                        seq,
                        k,
                        64,
                        w.down_proj.a_global(),
                        self.int8_clip(),
                    )?
                );
                drop(gate);
                drop(up);
                return timed!(
                    self,
                    "gemm.down",
                    self.marlin_int8_launch(
                        &s.x,
                        &s.scales,
                        qweight,
                        scales,
                        zeros,
                        true,
                        w.down_proj.group_size(),
                        seq,
                        n,
                        k,
                    )
                );
            }
        }
        let mut act = self.alloc_bf16(seq * inter)?;
        timed!(
            self,
            "mlp.act",
            kernels::silu_mul(
                &self.gpu,
                &self.kernels.silu_mul,
                &gate,
                &up,
                &mut act,
                seq * inter,
            )?
        );
        drop(gate);
        drop(up);
        Ok(timed!(
            self,
            "gemm.down",
            self.gemm(&act, seq, &w.down_proj)?
        ))
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
        let qg = timed!(self, "gemm.q", self.gemm(h, seq, &w.q_proj)?);
        let kk = timed!(self, "gemm.k", self.gemm(h, seq, &w.k_proj)?);
        let vv = timed!(self, "gemm.v", self.gemm(h, seq, &w.v_proj)?);
        if capture {
            self.dump_dev("attn_q_proj", &qg, seq, w.q_proj.out())?;
            self.dump_dev("attn_k_proj", &kk, seq, kv * hd)?;
            self.dump_dev("attn_v_proj", &vv, seq, kv * hd)?;
        }

        let (q, gate) = if cfg.attn_output_gate {
            let mut q = self.alloc_bf16(seq * heads * hd)?;
            let mut gate = self.alloc_bf16(seq * heads * hd)?;
            timed!(
                self,
                "attn.split",
                kernels::split_q_gate(
                    &self.gpu,
                    &self.kernels.split_q_gate,
                    &qg,
                    &mut q,
                    &mut gate,
                    seq,
                    heads,
                    hd,
                )?
            );
            if capture {
                self.dump_dev("attn_q", &q, seq, heads * hd)?;
                self.dump_dev("attn_gate", &gate, seq, heads * hd)?;
            }
            drop(qg);
            (q, Some(gate))
        } else {
            (qg, None)
        };

        let q = q;
        let mut qn = self.alloc_bf16(seq * heads * hd)?;
        timed!(
            self,
            "norm.q",
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
            )?
        );
        if capture {
            self.dump_dev("attn_q_norm", &qn, seq, heads * hd)?;
        }
        let mut kn = self.alloc_bf16(seq * kv * hd)?;
        timed!(
            self,
            "norm.k",
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
            )?
        );
        if capture {
            self.dump_dev("attn_k_norm", &kn, seq, kv * hd)?;
        }

        let rd = cfg.rotary_dim();
        let (cos, sin) = self.rope_tables(seq)?;
        timed!(
            self,
            "rope",
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
            )?
        );
        drop(cos);
        drop(sin);
        if capture {
            self.dump_dev("attn_q_rope", &qn, seq, heads * hd)?;
            self.dump_dev("attn_k_rope", &kn, seq, kv * hd)?;
        }

        let mut attn = self.alloc_bf16(seq * heads * hd)?;
        let __attn_t = self.prof_begin("attn");
        if let Some(fi) = self.flashinfer_for(seq, hd, heads, kv) {
            fi.prefill(&self.gpu, &*qn, &*kn, &*vv, &mut *attn, seq)?;
        } else if self.warp_attention.load(Ordering::Relaxed) {
            let tc = match hd {
                64 => Some(&self.kernels.attention_tc_64),
                128 => Some(&self.kernels.attention_tc_128),
                256 => Some(&self.kernels.attention_tc_256),
                _ => None,
            };
            if let Some(tc) = tc {
                kernels::attention_tc(
                    &self.gpu,
                    tc,
                    &*qn,
                    &*kn,
                    &*vv,
                    &mut *attn,
                    seq,
                    heads,
                    kv,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                )?;
            } else {
                kernels::attention_warp(
                    &self.gpu,
                    &self.kernels.attention_warp,
                    &*qn,
                    &*kn,
                    &*vv,
                    &mut *attn,
                    seq,
                    heads,
                    kv,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                )?;
            }
        } else {
            // --legacy-attention: warp-per-query kernel.
            kernels::attention_warp(
                &self.gpu,
                &self.kernels.attention_warp,
                &*qn,
                &*kn,
                &*vv,
                &mut *attn,
                seq,
                heads,
                kv,
                hd,
                1.0 / (hd as f32).sqrt(),
            )?;
        }
        self.prof_end("attn", __attn_t);
        drop(qn);
        drop(kn);
        drop(vv);
        if capture {
            self.dump_dev("attn_weights_out", &attn, seq, heads * hd)?;
        }

        let gated = match gate {
            Some(g) => {
                let mut o = self.alloc_bf16(seq * heads * hd)?;
                timed!(
                    self,
                    "attn.gate",
                    kernels::sigmoid_gate(
                        &self.gpu,
                        &self.kernels.sigmoid_gate,
                        &attn,
                        &g,
                        &mut o,
                        seq * heads * hd,
                    )?
                );
                drop(attn);
                drop(g);
                o
            }
            None => attn,
        };
        if capture {
            self.dump_dev("attn_gated", &gated, seq, heads * hd)?;
        }
        Ok(timed!(self, "gemm.o", self.gemm(&gated, seq, &w.o_proj)?))
    }

    fn gdn(&self, h: &CudaSlice<bf16>, seq: usize, w: &GdnWeights, capture: bool) -> Result<Buf> {
        let cfg = &self.cfg;
        let (nk, nv) = (cfg.linear_num_key_heads, cfg.linear_num_value_heads);
        let (dk, dv) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
        let conv_dim = 2 * cfg.gdn_key_dim() + cfg.gdn_value_dim();

        let qkv = timed!(self, "gemm.in_qkv", self.gemm(h, seq, &w.in_proj_qkv)?);
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
        timed!(
            self,
            "gdn.conv",
            if self.conv_tiled.load(Ordering::Relaxed) {
                kernels::conv1d_silu_tiled(
                    &self.gpu,
                    &self.kernels.conv1d_silu_tiled,
                    &qkv,
                    &w.conv1d,
                    &mut conv,
                    seq,
                    conv_dim,
                )?
            } else {
                kernels::conv1d_silu(
                    &self.gpu,
                    &self.kernels.conv1d_silu,
                    &qkv,
                    &w.conv1d,
                    &mut conv,
                    seq,
                    conv_dim,
                )?
            }
        );
        if capture {
            self.dump_dev("gdn_mixed_qkv_conv", &conv, seq, conv_dim)?;
        }

        let z = timed!(self, "gemm.z", self.gemm(h, seq, &w.in_proj_z)?);
        let b = timed!(self, "gemm.b", self.gemm(h, seq, &w.in_proj_b)?);
        let a = timed!(self, "gemm.a", self.gemm(h, seq, &w.in_proj_a)?);
        if capture {
            self.dump_dev("gdn_z", &z, seq, cfg.gdn_value_dim())?;
            self.dump_dev("gdn_b", &b, seq, nv)?;
            self.dump_dev("gdn_a", &a, seq, nv)?;
        }

        let mut qf = self.alloc_f32(seq * nv * dk)?;
        let mut kf = self.alloc_f32(seq * nv * dk)?;
        let mut vf = self.alloc_f32(seq * nv * dv)?;
        let mut g = self.alloc_f32(seq * nv)?;
        let mut beta = self.alloc_f32(seq * nv)?;
        timed!(
            self,
            "gdn.prepare",
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
            )?
        );
        drop(qkv);
        drop(conv);
        drop(b);
        drop(a);
        if capture {
            let qfv = self.gpu.stream.clone_dtoh(&*qf)?;
            let kfv = self.gpu.stream.clone_dtoh(&*kf)?;
            let vfv = self.gpu.stream.clone_dtoh(&*vf)?;
            let gv = self.gpu.stream.clone_dtoh(&*g)?;
            let bv = self.gpu.stream.clone_dtoh(&*beta)?;
            self.dump_f32("gdn_qf", &qfv, seq * nv, dk)?;
            self.dump_f32("gdn_kf", &kfv, seq * nv, dk)?;
            self.dump_f32("gdn_vf", &vfv, seq * nv, dv)?;
            self.dump_f32("gdn_g", &gv, seq, nv)?;
            self.dump_f32("gdn_beta", &bv, seq, nv)?;
        }

        let mut state = self.alloc_f32_zeros(nv * dk * dv)?;
        let mut core = self.alloc_bf16(seq * nv * dv)?;
        let min_seq = self.gdn_split_min_seq.load(Ordering::Relaxed);
        let chunk_min = self.gdn_chunked_min_seq.load(Ordering::Relaxed);
        if self.gdn_chunked_tc.load(Ordering::Relaxed) && seq >= chunk_min && dk == 128 && dv == 128
        {
            if std::env::var("DECIDRA_GDN_PROF").is_ok() {
                let blocks = (dv / kernels::GDN_CHUNK_VC) * nv;
                let mut prof = self.gpu.stream.alloc_zeros::<u64>(blocks * 10)?;
                timed!(
                    self,
                    "gdn.recurrent",
                    kernels::gdn_chunked_prof(
                        &self.gpu,
                        &self.kernels.gdn_chunked_tc_prof,
                        &*qf,
                        &*kf,
                        &*vf,
                        &*g,
                        &*beta,
                        &mut *state,
                        &mut *core,
                        &mut prof,
                        seq,
                        nv,
                        dk,
                        dv,
                    )?
                );
                self.gpu.synchronize()?;
                let host: Vec<u64> = self.gpu.stream.clone_dtoh(&prof)?;
                let mut acc = self.gdn_prof.lock().unwrap();
                for b in 0..blocks {
                    for p in 0..10 {
                        acc.cycles[p] += host[b * 10 + p];
                    }
                }
                acc.blocks += blocks as u64;
                acc.chunks += seq.div_ceil(kernels::GDN_CHUNK_C) as u64;
            } else {
                timed!(
                    self,
                    "gdn.recurrent",
                    kernels::gdn_chunked(
                        &self.gpu,
                        &self.kernels.gdn_chunked_tc,
                        &*qf,
                        &*kf,
                        &*vf,
                        &*g,
                        &*beta,
                        &mut *state,
                        &mut *core,
                        seq,
                        nv,
                        dk,
                        dv,
                    )?
                );
            }
        } else if self.gdn_chunked.load(Ordering::Relaxed) && seq >= chunk_min {
            timed!(
                self,
                "gdn.recurrent",
                kernels::gdn_chunked(
                    &self.gpu,
                    &self.kernels.gdn_chunked,
                    &*qf,
                    &*kf,
                    &*vf,
                    &*g,
                    &*beta,
                    &mut *state,
                    &mut *core,
                    seq,
                    nv,
                    dk,
                    dv,
                )?
            );
        } else if self.gdn_split.load(Ordering::Relaxed) && seq >= min_seq {
            timed!(
                self,
                "gdn.recurrent",
                kernels::gdn_recurrent_split(
                    &self.gpu,
                    &self.kernels.gdn_recurrent_split,
                    &*qf,
                    &*kf,
                    &*vf,
                    &*g,
                    &*beta,
                    &mut *state,
                    &mut *core,
                    seq,
                    nv,
                    dk,
                    dv,
                )?
            );
        } else if self.gdn_shared.load(Ordering::Relaxed) {
            let dv_chunks = if dv >= 64 {
                self.gdn_chunks.load(Ordering::Relaxed).max(1).min(dv / 16)
            } else {
                1
            };
            timed!(
                self,
                "gdn.recurrent",
                kernels::gdn_recurrent_shared(
                    &self.gpu,
                    &self.kernels.gdn_recurrent_shared,
                    &*qf,
                    &*kf,
                    &*vf,
                    &*g,
                    &*beta,
                    &mut *state,
                    &mut *core,
                    seq,
                    nv,
                    dk,
                    dv,
                    dv_chunks,
                )?
            );
        } else {
            timed!(
                self,
                "gdn.recurrent",
                kernels::gdn_recurrent(
                    &self.gpu,
                    &self.kernels.gdn_recurrent,
                    &*qf,
                    &*kf,
                    &*vf,
                    &*g,
                    &*beta,
                    &mut *state,
                    &mut *core,
                    seq,
                    nv,
                    dk,
                    dv,
                )?
            );
        }
        drop(qf);
        drop(kf);
        drop(vf);
        drop(g);
        drop(beta);
        if capture {
            self.dump_dev("gdn_recurrent_out", &core, seq, nv * dv)?;
            let sv = self.gpu.stream.clone_dtoh(&*state)?;
            self.dump_f32("gdn_state", &sv, nv * dk, dv)?;
        }

        let mut normed = self.alloc_bf16(seq * nv * dv)?;
        timed!(
            self,
            "gdn.norm",
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
            )?
        );
        drop(core);
        drop(z);
        if capture {
            self.dump_dev("gdn_norm_out", &normed, seq * nv, dv)?;
        }
        let out = timed!(self, "gemm.out", self.gemm(&normed, seq, &w.out_proj)?);
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
    /// Launch the Marlin INT8 tile group(s) for an already-quantized activation
    /// matrix. Split out so the fused SiLU-quant MLP path shares the exact
    /// launch/chunking logic.
    #[allow(clippy::too_many_arguments)]
    fn marlin_int8_launch(
        &self,
        xq: &CudaSlice<i8>,
        a_scales: &CudaSlice<f32>,
        qweight: &CudaSlice<i32>,
        scales: &CudaSlice<bf16>,
        zeros: &CudaSlice<i32>,
        group_mode: bool,
        weight_group: usize,
        rows: usize,
        n: usize,
        k: usize,
    ) -> Result<Buf> {
        let __p = self.cpu_prof.begin();
        let mut c = self.alloc_bf16(rows * n)?;
        self.cpu_prof.end("i8.c", __p);
        {
            let mut g = self.marlin_locks.lock().unwrap();
            if g.is_none() {
                *g = Some(self.gpu.stream.alloc_zeros::<i32>(128)?);
            }
        }
        let locks_guard = self.marlin_locks.lock().unwrap();
        let locks = locks_guard.as_ref().expect("locks initialized");
        let mut off = 0usize;
        let mut rest = rows;
        let chunk = marlin_chunk();
        let max_par = if chunk == 0 {
            usize::MAX
        } else {
            (chunk / 64).max(1)
        };
        let __p = self.cpu_prof.begin();
        while rest > 0 {
            let par = (rest / 64).min(max_par);
            let pm = if par > 0 { par * 64 } else { rest };
            let tmb = pm.div_ceil(16).min(4);
            let idx = if pm <= 8 { 0 } else { tmb - 1 };
            let f = if group_mode {
                &self.kernels.marlin_int8_group_cfgs[idx]
            } else if weight_group == 128 {
                &self.kernels.marlin_int8_g128_cfgs[idx]
            } else {
                &self.kernels.marlin_int8_cfgs[idx]
            };
            kernels::marlin_gemm_int8(
                &self.gpu,
                f,
                xq,
                qweight,
                &mut c,
                scales,
                zeros,
                a_scales,
                locks,
                off,
                pm,
                n,
                k,
                if group_mode { 64 } else { 0 },
            )?;
            off += pm;
            rest -= pm;
        }
        self.cpu_prof.end("i8.launch", __p);
        Ok(c)
    }

    pub fn gemm(&self, x: &CudaSlice<bf16>, rows: usize, w: &LinearWeight) -> Result<Buf> {
        let mut c = self.gemm_core(x, rows, w)?;
        if let Some(lora) = w.lora() {
            self.add_lora(x, rows, w.in_(), w.out(), lora, &mut c)?;
        }
        Ok(c)
    }

    fn blas_gemm_bf16(
        &self,
        x: &CudaSlice<bf16>,
        m: usize,
        k: usize,
        w: &CudaSlice<bf16>,
        n: usize,
        out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
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
                .gemm(cfg, w, x, out)
                .context("cublas lora gemm")?;
        }
        Ok(())
    }

    /// Add the low-rank adapter delta: `(x A^T) B^T` (B pre-scaled).
    fn add_lora(
        &self,
        x: &CudaSlice<bf16>,
        rows: usize,
        k: usize,
        n: usize,
        lora: &Lora,
        c: &mut Buf,
    ) -> Result<()> {
        let mut mid = self.alloc_bf16(rows * lora.rank)?;
        self.blas_gemm_bf16(x, rows, k, &lora.a, lora.rank, &mut mid)?;
        let mut delta = self.alloc_bf16(rows * n)?;
        self.blas_gemm_bf16(&mid, rows, lora.rank, &lora.b, n, &mut delta)?;
        let mut out = self.alloc_bf16(rows * n)?;
        kernels::add(&self.gpu, &self.kernels.add, c, &delta, &mut out, rows * n)?;
        *c = out;
        Ok(())
    }

    /// Add the lm_head low-rank delta for the selected head ids.
    fn add_head_lora(
        &self,
        h: &CudaSlice<bf16>,
        rows: usize,
        ids: &[u32],
        logits: &mut Buf,
    ) -> Result<()> {
        let (Some(head), Some(store)) = (&self.lora_head, &self.lora_store) else {
            return Ok(());
        };
        let rank = head.rank;
        let hidden = self.cfg.hidden_size;
        let mut mid = self.alloc_bf16(rows * rank)?;
        self.blas_gemm_bf16(h, rows, hidden, &head.a, rank, &mut mid)?;
        let mut b_host = vec![bf16::ZERO; ids.len() * rank];
        let b_name = "base_model.model.lm_head.lora_B.weight";
        for (i, &id) in ids.iter().enumerate() {
            store.bf16_row(b_name, id as usize, &mut b_host[i * rank..(i + 1) * rank])?;
        }
        for v in &mut b_host {
            *v = bf16::from_f32(v.to_f32() * self.lora_scale);
        }
        let b = self.gpu.stream.clone_htod(&b_host)?;
        let mut delta = self.alloc_bf16(rows * ids.len())?;
        self.blas_gemm_bf16(&mid, rows, rank, &b, ids.len(), &mut delta)?;
        let mut out = self.alloc_bf16(rows * ids.len())?;
        kernels::add(
            &self.gpu,
            &self.kernels.add,
            logits,
            &delta,
            &mut out,
            rows * ids.len(),
        )?;
        *logits = out;
        Ok(())
    }

    fn gemm_core(&self, x: &CudaSlice<bf16>, rows: usize, w: &LinearWeight) -> Result<Buf> {
        let (n, k) = (w.out(), w.in_());
        if x.len() != rows * k {
            bail!("gemm input has {} values, expected {}", x.len(), rows * k);
        }
        let dbg = gemm_debug();
        // Synchronizing here makes any benchmarks with the debug variable set
        // measure serialized launches. Timing requires the explicit
        // DECIDRA_GEMM_DEBUG_SYNC opt-in; DECIDRA_GEMM_DEBUG only logs labels.
        let dbg_sync = gemm_debug_sync();
        let mark = |label: &str, t: std::time::Instant| {
            if dbg_sync {
                let _ = self.gpu.synchronize();
            }
            if dbg || dbg_sync {
                eprintln!("{label}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
            }
        };
        if dbg_sync {
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
            LinearWeight::Marlin {
                qweight,
                scales,
                zeros,
                ..
            } => {
                if w.int8_a() {
                    let __p = self.cpu_prof.begin();
                    let need = rows * k;
                    let group_mode = w.group_a() && k % 64 == 0;
                    let groups = if group_mode { k / 64 } else { 1 };
                    let mut scratch = self.int8_scratch.lock().unwrap();
                    if scratch
                        .as_ref()
                        .map_or(true, |s| s.x.len() < need || s.scales.len() < rows * groups)
                    {
                        *scratch = Some(Int8Scratch {
                            x: self.gpu.stream.alloc_zeros::<i8>(need)?,
                            scales: self.gpu.stream.alloc_zeros::<f32>(rows * groups)?,
                        });
                    }
                    let s = scratch.as_mut().unwrap();
                    self.cpu_prof.end("i8.alloc", __p);
                    let __p = self.cpu_prof.begin();
                    if group_mode {
                        kernels::quant_int8_group_bf16(
                            &self.gpu,
                            &self.kernels.quant_int8_group_bf16,
                            x,
                            &mut s.x,
                            &mut s.scales,
                            rows,
                            k,
                            64,
                            w.a_global(),
                            self.int8_clip(),
                            int8_group_uniform(),
                        )?;
                    } else {
                        kernels::quant_int8_bf16(
                            &self.gpu,
                            &self.kernels.quant_int8_bf16,
                            x,
                            &mut s.x,
                            &mut s.scales,
                            rows,
                            k,
                            w.a_global(),
                            self.int8_clip(),
                        )?;
                    }
                    self.cpu_prof.end("i8.quant", __p);
                    return self.marlin_int8_launch(
                        &s.x,
                        &s.scales,
                        qweight,
                        scales,
                        zeros,
                        group_mode,
                        w.group_size(),
                        rows,
                        n,
                        k,
                    );
                }
                let __p = self.cpu_prof.begin();
                let mut c = self.alloc_bf16(rows * n)?;
                self.cpu_prof.end("w4.c", __p);
                {
                    let mut g = self.marlin_locks.lock().unwrap();
                    if g.is_none() {
                        *g = Some(self.gpu.stream.alloc_zeros::<i32>(128)?);
                    }
                }
                let locks = self.marlin_locks.lock().unwrap();
                let locks = locks.as_ref().expect("locks initialized");
                let mut off = 0usize;
                let mut rest = rows;
                // 0 = one launch for the whole M (the kernel's internal
                // `parallel` loop); >0 caps each launch to that many rows.
                let chunk = marlin_chunk();
                let max_par = if chunk == 0 {
                    usize::MAX
                } else {
                    (chunk / 64).max(1)
                };
                let __p = self.cpu_prof.begin();
                while rest > 0 {
                    let par = (rest / 64).min(max_par);
                    let pm = if par > 0 { par * 64 } else { rest };
                    let tmb = pm.div_ceil(16).min(4);
                    let idx = if pm <= 8 { 0 } else { tmb };
                    let cfgs = if w.group_size() == 128 {
                        &self.kernels.marlin_g128_cfgs
                    } else {
                        &self.kernels.marlin_cfgs
                    };
                    kernels::marlin_gemm(
                        &self.gpu, &cfgs[idx], x, qweight, &mut c, scales, zeros, locks, off, pm,
                        n, k,
                    )?;
                    off += pm;
                    rest -= pm;
                }
                self.cpu_prof.end("w4.launch", __p);
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
                    if rows <= 64 || !self.fused_tile128.load(Ordering::Relaxed) {
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
                    } else {
                        kernels::gemm_w4a16_m128(
                            &self.gpu,
                            &self.kernels.gemm_w4a16_m128,
                            x,
                            packed,
                            scale,
                            zp,
                            &mut c,
                            rows,
                            n,
                            k,
                        )?;
                    }
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
                mark("  dequant", t0);
                let t2 = std::time::Instant::now();
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
                mark("  cublas", t2);
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

    #[allow(clippy::too_many_arguments)]
    fn marlin_fp8_gemm(
        &self,
        w: &LinearWeight,
        xq: &CudaSlice<u8>,
        a_scales: &CudaSlice<f32>,
        c: &mut CudaSlice<bf16>,
        rows: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        {
            let mut g = self.marlin_locks.lock().unwrap();
            if g.is_none() {
                *g = Some(self.gpu.stream.alloc_zeros::<i32>(128)?);
            }
        }
        let locks_guard = self.marlin_locks.lock().unwrap();
        let locks = locks_guard.as_ref().expect("locks initialized");
        let (qweight, scales, zeros) = match w {
            LinearWeight::Marlin {
                qweight,
                scales,
                zeros,
                ..
            } => (qweight, scales, zeros),
            _ => bail!("FP8 path requires a Marlin-repacked module"),
        };
        let mut off = 0usize;
        let mut rest = rows;
        while rest > 0 {
            let par = (rest / 64).min(16);
            let pm = if par > 0 { par * 64 } else { rest };
            let tmb = pm.div_ceil(16).min(4);
            let idx = if pm <= 8 { 0 } else { tmb - 1 };
            let f = &self.kernels.marlin_fp8_cfgs[idx];
            kernels::marlin_gemm_fp8(
                &self.gpu, f, xq, qweight, c, scales, zeros, a_scales, locks, off, pm, n, k,
            )?;
            off += pm;
            rest -= pm;
        }
        Ok(())
    }

    /// Benchmark W4A16 vs W4A8 on one already-loaded module (no repack in the
    /// timed loop). Returns (w4a16_ms, w4a8_ms) per repeat.
    pub fn bench_matmul_fp8(
        &self,
        module: &str,
        input: &[f32],
        rows: usize,
        repeat: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let was = self.marlin_fp8.swap(false, Ordering::Relaxed);
        let prov16 = self.provider()?;
        let w16 = prov16.linear(module)?;
        self.marlin_fp8.store(true, Ordering::Relaxed);
        let prov8 = self.provider()?;
        let w8 = prov8.linear(module)?;
        self.marlin_fp8.store(was, Ordering::Relaxed);
        let (n, k) = (w16.out(), w16.in_());
        let host: Vec<bf16> = input.iter().map(|v| bf16::from_f32(*v)).collect();
        let x16 = self.gpu.stream.clone_htod(&host)?;
        let xf = self.gpu.stream.clone_htod(input)?;
        let mut xq = self.gpu.stream.alloc_zeros::<u8>(rows * k)?;
        let mut as_ = self.gpu.stream.alloc_zeros::<f32>(rows)?;
        kernels::quant_fp8(
            &self.gpu,
            &self.kernels.quant_fp8,
            &xf,
            &mut xq,
            &mut as_,
            rows,
            k,
        )?;
        let mut c16 = self.alloc_bf16(rows * n)?;
        let mut c8 = self.alloc_bf16(rows * n)?;
        // Warm-up (fills the allocator/Marlin locks).
        let _ = self.gemm(&x16, rows, &w16)?;
        self.marlin_fp8_gemm(&w8, &xq, &as_, &mut c8, rows, n, k)?;
        self.gpu.synchronize()?;
        let mut t16 = Vec::with_capacity(repeat);
        let mut t8 = Vec::with_capacity(repeat);
        for _ in 0..repeat {
            self.gpu.synchronize()?;
            let t = std::time::Instant::now();
            let _ = self.gemm(&x16, rows, &w16)?;
            self.gpu.synchronize()?;
            t16.push(t.elapsed().as_secs_f64() * 1e3);
            self.gpu.synchronize()?;
            let t = std::time::Instant::now();
            self.marlin_fp8_gemm(&w8, &xq, &as_, &mut c8, rows, n, k)?;
            self.gpu.synchronize()?;
            t8.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let _ = (&mut c16, &mut c8);
        Ok((t16, t8))
    }

    #[allow(clippy::too_many_arguments)]
    fn marlin_int8_gemm(
        &self,
        w: &LinearWeight,
        xq: &CudaSlice<i8>,
        a_scales: &CudaSlice<f32>,
        c: &mut CudaSlice<bf16>,
        rows: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        {
            let mut g = self.marlin_locks.lock().unwrap();
            if g.is_none() {
                *g = Some(self.gpu.stream.alloc_zeros::<i32>(128)?);
            }
        }
        let locks_guard = self.marlin_locks.lock().unwrap();
        let locks = locks_guard.as_ref().expect("locks initialized");
        let (qweight, scales, zeros) = match w {
            LinearWeight::Marlin {
                qweight,
                scales,
                zeros,
                ..
            } => (qweight, scales, zeros),
            _ => bail!("INT8 path requires a Marlin-repacked module"),
        };
        let mut off = 0usize;
        let mut rest = rows;
        while rest > 0 {
            let par = (rest / 64).min(16);
            let pm = if par > 0 { par * 64 } else { rest };
            let tmb = pm.div_ceil(16).min(4);
            let idx = if pm <= 8 { 0 } else { tmb - 1 };
            let f = &self.kernels.marlin_int8_cfgs[idx];
            kernels::marlin_gemm_int8(
                &self.gpu, f, xq, qweight, c, scales, zeros, a_scales, locks, off, pm, n, k, 0,
            )?;
            off += pm;
            rest -= pm;
        }
        Ok(())
    }

    /// Experimental W4A8-INT8 path for one module.
    pub fn matmul_module_int8(
        &self,
        module: &str,
        input: &[f32],
        rows: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let prov = self.provider()?;
        let w = prov.linear(module)?;
        let (n, k) = (w.out(), w.in_());
        if input.len() != rows * k {
            bail!("input has {} values, expected {}", input.len(), rows * k);
        }
        let xf = self.gpu.stream.clone_htod(input)?;
        let mut xq = self.gpu.stream.alloc_zeros::<i8>(rows * k)?;
        let mut as_ = self.gpu.stream.alloc_zeros::<f32>(rows)?;
        kernels::quant_int8(
            &self.gpu,
            &self.kernels.quant_int8,
            &xf,
            &mut xq,
            &mut as_,
            rows,
            k,
            w.a_global(),
            self.int8_clip(),
        )?;
        let mut c = self.alloc_bf16(rows * n)?;
        self.marlin_int8_gemm(&w, &xq, &as_, &mut c, rows, n, k)?;
        let got = self.to_f32(&c, rows * n)?;
        let was8 = self.marlin_int8.swap(false, Ordering::Relaxed);
        let wasf = self.marlin_fp8.swap(false, Ordering::Relaxed);
        let reference = self.matmul_module(module, input, rows, false);
        self.marlin_int8.store(was8, Ordering::Relaxed);
        self.marlin_fp8.store(wasf, Ordering::Relaxed);
        Ok((got, reference?))
    }

    /// Experimental W4A8-FP8 path for one module: per-token FP8 activation
    /// quantization + a8-repacked Marlin, compared against W4A16. The caller
    /// must enable `set_marlin(true)` and `set_marlin_fp8(true)`.
    pub fn matmul_module_fp8(
        &self,
        module: &str,
        input: &[f32],
        rows: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let prov = self.provider()?;
        let w = prov.linear(module)?;
        let (n, k) = (w.out(), w.in_());
        if input.len() != rows * k {
            bail!("input has {} values, expected {}", input.len(), rows * k);
        }
        let xf = self.gpu.stream.clone_htod(input)?;
        let mut xq = self.gpu.stream.alloc_zeros::<u8>(rows * k)?;
        let mut as_ = self.gpu.stream.alloc_zeros::<f32>(rows)?;
        kernels::quant_fp8(
            &self.gpu,
            &self.kernels.quant_fp8,
            &xf,
            &mut xq,
            &mut as_,
            rows,
            k,
        )?;
        let mut c = self.alloc_bf16(rows * n)?;
        self.marlin_fp8_gemm(&w, &xq, &as_, &mut c, rows, n, k)?;
        let got = self.to_f32(&c, rows * n)?;
        let was = self.marlin_fp8.swap(false, Ordering::Relaxed);
        let reference = self.matmul_module(module, input, rows, false);
        self.marlin_fp8.store(was, Ordering::Relaxed);
        Ok((got, reference?))
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
        let prov = self.provider()?;
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
        let prov = self.provider()?;
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

    /// Dump a Marlin-repacked module: (qweight i32, scales f32, zeros i32).
    pub fn dump_marlin(&self, module: &str) -> Result<(Vec<i32>, Vec<f32>, Vec<i32>)> {
        let prov = self.provider()?;
        let w = prov.linear(module)?;
        match w {
            LinearWeight::Marlin {
                qweight,
                scales,
                zeros,
                ..
            } => {
                let q = self.gpu.stream.clone_dtoh(&qweight)?;
                let s: Vec<bf16> = self.gpu.stream.clone_dtoh(&scales)?;
                let z = self.gpu.stream.clone_dtoh(&zeros)?;
                Ok((q, s.iter().map(|v| v.to_f32()).collect(), z))
            }
            _ => bail!("{module} did not take the Marlin path"),
        }
    }

    /// Dequantize one packed module to f32 (validation hook for AWQ).
    pub fn dequant_module(&self, module: &str) -> Result<Vec<f32>> {
        let prov = self.provider()?;
        let w = prov.linear(module)?;
        match &w {
            LinearWeight::Bf16 { .. } => bail!("{module} is not packed"),
            LinearWeight::Marlin { .. } => {
                bail!("{module} is repacked to Marlin; rerun without --marlin")
            }
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
            lora: None,
        };
        let mut logits = timed!(self, "head", self.gemm(h, seq, &lw)?);
        self.add_head_lora(h, seq, ids, &mut logits)?;
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
        let head_mid = if let Some(head) = &self.lora_head {
            let mut mid = self.alloc_bf16(rows * head.rank)?;
            self.blas_gemm_bf16(&x, rows, hidden, &head.a, head.rank, &mut mid)?;
            Some(mid)
        } else {
            None
        };
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
                lora: None,
            };
            let mut c = timed!(self, "head", self.gemm(&x, rows, &lw)?);
            if let (Some(mid), Some(head), Some(store)) =
                (&head_mid, &self.lora_head, &self.lora_store)
            {
                let rank = head.rank;
                let mut b_host = vec![bf16::ZERO; n * rank];
                for r in 0..n {
                    store.bf16_row(
                        "base_model.model.lm_head.lora_B.weight",
                        start + r,
                        &mut b_host[r * rank..(r + 1) * rank],
                    )?;
                }
                for v in &mut b_host {
                    *v = bf16::from_f32(v.to_f32() * self.lora_scale);
                }
                let b = self.gpu.stream.clone_htod(&b_host)?;
                let mut delta = self.alloc_bf16(rows * n)?;
                self.blas_gemm_bf16(mid, rows, rank, &b, n, &mut delta)?;
                let mut next = self.alloc_bf16(rows * n)?;
                kernels::add(
                    &self.gpu,
                    &self.kernels.add,
                    &c,
                    &delta,
                    &mut next,
                    rows * n,
                )?;
                c = next;
            }
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

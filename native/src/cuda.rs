//! CUDA context, cubin module, cuBLAS handle.
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use cudarc::cublas::CudaBlas;
use cudarc::cublaslt::safe::CudaBlasLT;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, sys,
};
use cudarc::nvrtc::Ptx;

use crate::arch::{ArchProfile, GpuArch};

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    /// Vendored Marlin W4A16 module (third_party/marlin).
    pub marlin_module: Arc<CudaModule>,
    pub blas: CudaBlas,
    /// cuBLASLt handle and workspace for SM120 FP8 matmul.
    pub blaslt: CudaBlasLT,
    pub blaslt_workspace: CudaSlice<u8>,
    /// Streaming multiprocessor count of the open device. Marlin launches one
    /// persistent block per SM; a fixed 128 leaves SMs idle on larger parts.
    pub sms: usize,
    /// Detected runtime architecture and its host-side tuning profile.
    pub arch: GpuArch,
    pub profile: ArchProfile,
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)
            .with_context(|| format!("creating CUDA context on device {ordinal}"))?;
        let stream = ctx.default_stream();

        let (major, minor) = ctx
            .compute_capability()
            .context("querying compute capability")?;
        let arch = GpuArch::from_compute(major, minor);
        let compiled =
            GpuArch::from_compile_arch(env!("DECIDRA_COMPILED_ARCH")).with_context(|| {
                format!(
                    "invalid build architecture {:?}",
                    env!("DECIDRA_COMPILED_ARCH")
                )
            })?;
        if compiled != arch {
            let hint = match arch {
                GpuArch::Sm89 | GpuArch::Sm80 => "sm_89".to_owned(),
                GpuArch::Sm90 => "sm_90".to_owned(),
                GpuArch::Sm120 => "sm_120".to_owned(),
                GpuArch::Other { major, minor } => format!("sm_{major}{minor}"),
            };
            bail!(
                "this binary embeds {} kernels but the GPU is {} (SM {major}.{minor}); \
                 rebuild with DECIDRA_CUDA_ARCH={hint}",
                compiled.label(),
                arch.label(),
            );
        }
        let cubin: &[u8] = include_bytes!(env!("DECIDRA_KERNELS_CUBIN"));
        let marlin: &[u8] = include_bytes!(env!("DECIDRA_MARLIN_CUBIN"));
        let module = ctx
            .load_module(Ptx::from_binary(cubin.to_vec()))
            .context("loading kernel cubin")?;
        let marlin_module = ctx
            .load_module(Ptx::from_binary(marlin.to_vec()))
            .context("loading Marlin cubin")?;
        let blas = CudaBlas::new(stream.clone()).context("creating cuBLAS handle")?;
        let blaslt = CudaBlasLT::new(stream.clone()).context("creating cuBLASLt handle")?;
        let blaslt_workspace = stream.alloc_zeros::<u8>(32 * 1024 * 1024)?;
        let sms = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .context("querying multiprocessor count")? as usize;
        let profile = arch.profile(sms);
        Ok(Self {
            ctx,
            stream,
            module,
            marlin_module,
            blas,
            blaslt,
            blaslt_workspace,
            sms,
            arch,
            profile,
        })
    }

    pub fn func(&self, name: &str) -> Result<CudaFunction> {
        self.module
            .load_function(name)
            .with_context(|| format!("kernel {name} not found in cubin"))
    }

    pub fn func_marlin(&self, name: &str) -> Result<CudaFunction> {
        self.marlin_module
            .load_function(name)
            .with_context(|| format!("Marlin kernel {name} not found"))
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().context("stream synchronize")
    }

    pub fn device_name(&self) -> Result<String> {
        self.ctx.name().context("device name")
    }

    /// Return cached blocks in the CUDA default memory pool to the driver.
    /// cudarc allocates through `cuMemAllocAsync` when memory pools are
    /// supported, so dropping a CudaSlice alone does not increase
    /// `cudaMemGetInfo` free memory until the pool is trimmed.
    pub fn trim_allocator(&self) -> Result<()> {
        self.synchronize()?;
        unsafe {
            let mut device: sys::CUdevice = 0;
            let rc = sys::cuDeviceGet(&mut device, self.ctx.ordinal() as i32);
            if rc != sys::CUresult::CUDA_SUCCESS {
                bail!("cuDeviceGet failed: {rc:?}");
            }
            let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
            let rc = sys::cuDeviceGetDefaultMemPool(&mut pool, device);
            if rc != sys::CUresult::CUDA_SUCCESS {
                bail!("cuDeviceGetDefaultMemPool failed: {rc:?}");
            }
            if pool.is_null() {
                bail!("CUDA default memory pool is null");
            }
            let rc = sys::cuMemPoolTrimTo(pool, 0);
            if rc != sys::CUresult::CUDA_SUCCESS {
                bail!("cuMemPoolTrimTo failed: {rc:?}");
            }
        }
        Ok(())
    }

    /// (free_bytes, total_bytes)
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.ctx.mem_get_info().context("cudaMemGetInfo")
    }

    /// (major, minor) compute capability of the open device.
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        self.ctx
            .compute_capability()
            .context("querying compute capability")
    }

    /// Start a timing-enabled event pair on the default stream. The end event
    /// is recorded later with [`PhaseTimer::stop`].
    pub fn phase_timer(&self) -> Result<PhaseTimer> {
        let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let start = self
            .ctx
            .new_event(flags)
            .context("creating phase start event")?;
        start
            .record(&self.stream)
            .context("recording phase start")?;
        let end = self
            .ctx
            .new_event(flags)
            .context("creating phase end event")?;
        Ok(PhaseTimer { start, end })
    }
}

/// CUDA-event timer for an asynchronous phase (capture, vision, restore).
/// `seconds` is valid once the stream has synchronized past the end event.
pub struct PhaseTimer {
    start: CudaEvent,
    end: CudaEvent,
}

impl PhaseTimer {
    pub fn stop(&mut self, gpu: &Gpu) -> Result<()> {
        self.end.record(&gpu.stream).context("recording phase end")
    }

    pub fn seconds(&self) -> Result<f64> {
        self.start
            .elapsed_ms(&self.end)
            .map(|ms| f64::from(ms) / 1000.0)
            .context("reading phase timer")
    }
}

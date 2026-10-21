//! CUDA context, cubin module, cuBLAS handle.
use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, sys};
use cudarc::nvrtc::Ptx;

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    /// Vendored Marlin W4A16 module (third_party/marlin).
    pub marlin_module: Arc<CudaModule>,
    pub blas: CudaBlas,
    /// Streaming multiprocessor count of the open device. Marlin launches one
    /// persistent block per SM; a fixed 128 leaves SMs idle on larger parts.
    pub sms: usize,
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)
            .with_context(|| format!("creating CUDA context on device {ordinal}"))?;
        let stream = ctx.default_stream();

        let cubin: &[u8] = include_bytes!(env!("DECIDRA_KERNELS_CUBIN"));
        let marlin: &[u8] = include_bytes!(env!("DECIDRA_MARLIN_CUBIN"));
        let module = ctx
            .load_module(Ptx::from_binary(cubin.to_vec()))
            .context("loading kernel cubin")?;
        let marlin_module = ctx
            .load_module(Ptx::from_binary(marlin.to_vec()))
            .context("loading Marlin cubin")?;
        let blas = CudaBlas::new(stream.clone()).context("creating cuBLAS handle")?;
        let sms = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .context("querying multiprocessor count")? as usize;
        Ok(Self {
            ctx,
            stream,
            module,
            marlin_module,
            blas,
            sms,
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
}

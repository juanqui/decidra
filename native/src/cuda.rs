//! CUDA context, cubin module, cuBLAS handle.
use std::fs;
use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::Ptx;

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    pub blas: CudaBlas,
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)
            .with_context(|| format!("creating CUDA context on device {ordinal}"))?;
        let stream = ctx.default_stream();

        let cubin_path = env!("JEV_KERNELS_CUBIN");
        let cubin = fs::read(cubin_path)
            .with_context(|| format!("reading compiled kernels at {cubin_path}"))?;
        let module = ctx
            .load_module(Ptx::from_binary(cubin))
            .context("loading kernel cubin")?;
        let blas = CudaBlas::new(stream.clone()).context("creating cuBLAS handle")?;
        Ok(Self {
            ctx,
            stream,
            module,
            blas,
        })
    }

    pub fn func(&self, name: &str) -> Result<CudaFunction> {
        self.module
            .load_function(name)
            .with_context(|| format!("kernel {name} not found in cubin"))
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
}

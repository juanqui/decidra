//! Raw C ABI to the vendored FlashInfer FA2 prefill library.
//!
//! The library is compiled by `build.rs` from
//! `third_party/flashinfer/fi_prefill.cu` (header-only FlashInfer templates,
//! no PyTorch/ATen/TVM). It exports:
//!
//! `int fi_prefill_bf16(const void* q, const void* k, const void* v, void* out,
//!                      int seq, cudaStream_t stream);`
//!
//! Shapes are fixed for this engine: BF16 q `[S,24,256]`, k/v `[S,4,256]`,
//! causal, NHD, head_dim 256, scale 0.0625.

use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::Arc;

use anyhow::{Result, bail};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::bf16;

type PrefillFn = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *const c_void,
    *mut c_void,
    c_int,
    *mut c_void,
) -> c_int;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

const RTLD_NOW: c_int = 2;

pub struct FlashInfer {
    handle: *mut c_void,
    func: PrefillFn,
}

// The handle is opened once and only used on the engine's single CUDA stream.
unsafe impl Send for FlashInfer {}
unsafe impl Sync for FlashInfer {}

impl FlashInfer {
    /// Load the library baked into the build. Returns `None` when the build did
    /// not produce it or `dlopen` fails; callers then keep the native kernel.
    pub fn load() -> Option<Arc<Self>> {
        let path = env!("JEV_FI_PREFILL_SO");
        let cpath = CString::new(path).ok()?;
        unsafe {
            let handle = dlopen(cpath.as_ptr(), RTLD_NOW);
            if handle.is_null() {
                eprintln!("flashinfer: dlopen({path}) failed; using native attention");
                return None;
            }
            let sym = CString::new("fi_prefill_bf16").ok()?;
            let p = dlsym(handle, sym.as_ptr());
            if p.is_null() {
                eprintln!("flashinfer: fi_prefill_bf16 missing in {path}");
                dlclose(handle);
                return None;
            }
            let func: PrefillFn = std::mem::transmute(p);
            Some(Arc::new(Self { handle, func }))
        }
    }

    pub fn prefill<Q, K, V, O>(
        &self,
        gpu: &crate::cuda::Gpu,
        q: &Q,
        k: &K,
        v: &V,
        out: &mut O,
        seq: usize,
    ) -> Result<()>
    where
        Q: DevicePtr<bf16>,
        K: DevicePtr<bf16>,
        V: DevicePtr<bf16>,
        O: DevicePtrMut<bf16>,
    {
        let (qp, _g0) = q.device_ptr(&gpu.stream);
        let (kp, _g1) = k.device_ptr(&gpu.stream);
        let (vp, _g2) = v.device_ptr(&gpu.stream);
        let (op, _g3) = out.device_ptr_mut(&gpu.stream);
        let stream = gpu.stream.cu_stream() as *mut c_void;
        let rc = unsafe {
            (self.func)(
                qp as *const c_void,
                kp as *const c_void,
                vp as *const c_void,
                op as *mut c_void,
                seq as c_int,
                stream,
            )
        };
        if rc != 0 {
            bail!("FlashInfer prefill failed with cudaError {rc}");
        }
        Ok(())
    }
}

impl Drop for FlashInfer {
    fn drop(&mut self) {
        unsafe {
            dlclose(self.handle);
        }
    }
}

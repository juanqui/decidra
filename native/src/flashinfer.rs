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
use std::path::PathBuf;
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
const EMBEDDED_SO: &[u8] = include_bytes!(env!("DECIDRA_FI_PREFILL_SO"));

/// Write the embedded prefill library to a private temp directory so `dlopen`
/// can map it. The image then needs no kernel or `.so` files on disk.
fn materialize() -> Option<PathBuf> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .subsec_nanos();
    let dir = std::env::temp_dir().join(format!("decidra-{}-{nanos}", std::process::id()));
    std::fs::create_dir(&dir).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    }
    let path = dir.join("libfi_prefill.so");
    std::fs::write(&path, EMBEDDED_SO).ok()?;
    Some(path)
}

pub struct FlashInfer {
    handle: *mut c_void,
    func: PrefillFn,
    func_noncausal: PrefillFn,
}

// The handle is opened once and only used on the engine's single CUDA stream.
unsafe impl Send for FlashInfer {}
unsafe impl Sync for FlashInfer {}

impl FlashInfer {
    /// Load the library embedded in the binary. Falls back to the build path
    /// when the temp-file write fails (local development). Returns `None` if
    /// `dlopen` fails; callers then keep the native kernel.
    pub fn load() -> Option<Arc<Self>> {
        let (path, cleanup) = match materialize() {
            Some(path) => (path, true),
            None => (PathBuf::from(env!("DECIDRA_FI_PREFILL_SO")), false),
        };
        let cpath = CString::new(path.to_string_lossy().as_bytes()).ok()?;
        unsafe {
            let handle = dlopen(cpath.as_ptr(), RTLD_NOW);
            if cleanup {
                let _ = std::fs::remove_file(&path);
                if let Some(dir) = path.parent() {
                    let _ = std::fs::remove_dir(dir);
                }
            }
            if handle.is_null() {
                eprintln!(
                    "flashinfer: dlopen({}) failed; using native attention",
                    path.display()
                );
                return None;
            }
            let sym = CString::new("fi_prefill_bf16").ok()?;
            let p = dlsym(handle, sym.as_ptr());
            if p.is_null() {
                eprintln!("flashinfer: fi_prefill_bf16 missing in {}", path.display());
                dlclose(handle);
                return None;
            }
            let func: PrefillFn = std::mem::transmute(p);
            let sym_nc = CString::new("fi_prefill_bf16_noncausal").ok()?;
            let p_nc = dlsym(handle, sym_nc.as_ptr());
            if p_nc.is_null() {
                eprintln!(
                    "flashinfer: fi_prefill_bf16_noncausal missing in {}",
                    path.display()
                );
                dlclose(handle);
                return None;
            }
            let func_noncausal: PrefillFn = std::mem::transmute(p_nc);
            Some(Arc::new(Self {
                handle,
                func,
                func_noncausal,
            }))
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

    /// Noncausal (full) attention over the whole sequence. Used by the
    /// AutoTrust/d3/d3-turbo/pplx decision family; the GDN layers stay causal.
    pub fn prefill_noncausal<Q, K, V, O>(
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
            (self.func_noncausal)(
                qp as *const c_void,
                kp as *const c_void,
                vp as *const c_void,
                op as *mut c_void,
                seq as c_int,
                stream,
            )
        };
        if rc != 0 {
            bail!("FlashInfer noncausal prefill failed with cudaError {rc}");
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

//! Raw C ABI to the vendored FlashInfer prefill libraries.
//!
//! Two libraries are embedded:
//! - `fi_prefill.cu` (FA2, SM80+) exports `fi_prefill_bf16` and
//!   `fi_prefill_bf16_noncausal`.
//! - `fi_prefill_fa3.cu` (FA3, SM90a, optional) exports
//!   `fi_prefill_bf16_fa3` and `fi_prefill_bf16_noncausal_fa3`.
//!
//! Shapes are fixed for this engine: BF16 q `[S,24,256]`, k/v `[S,4,256]`,
//! causal or full attention, NHD, head_dim 256, scale 0.0625. `build.rs`
//! embeds an empty placeholder when FA3 is not compiled, so the SM89 build is
//! unchanged.

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

/// `(q, k, v, bit-packed mask, out, seq, stream)`.
type CustomFn = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *mut c_void,
    c_int,
    *mut c_void,
) -> c_int;

/// `(q, k, v, out, qo_len, kv_len, stream)`.
type ExtFn = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *const c_void,
    *mut c_void,
    c_int,
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
const EMBEDDED_FA3: &[u8] = include_bytes!(env!("DECIDRA_FI_PREFILL_FA3_SO"));

/// Write embedded bytes to a private temp directory so `dlopen` can map them.
/// The file is removed immediately after the library is opened.
fn materialize(bytes: &[u8], name: &str, fallback: &str) -> Option<PathBuf> {
    if bytes.is_empty() {
        return None;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .subsec_nanos();
    let dir = std::env::temp_dir().join(format!(
        "decidra-{}-{nanos}-{}",
        std::process::id(),
        name.trim_end_matches(".so")
    ));
    std::fs::create_dir(&dir).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    }
    let path = dir.join(name);
    match std::fs::write(&path, bytes) {
        Ok(()) => Some(path),
        Err(_) => Some(PathBuf::from(fallback)),
    }
}

unsafe fn open_lib(
    bytes: &[u8],
    name: &str,
    fallback: &str,
    causal_sym: &str,
    noncausal_sym: &str,
) -> Option<(*mut c_void, PrefillFn, PrefillFn)> {
    let path = materialize(bytes, name, fallback)?;
    let is_temp = path != PathBuf::from(fallback);
    let cpath = CString::new(path.to_string_lossy().as_bytes()).ok()?;
    unsafe {
        let handle = dlopen(cpath.as_ptr(), RTLD_NOW);
        if is_temp {
            let _ = std::fs::remove_file(&path);
            if let Some(dir) = path.parent() {
                let _ = std::fs::remove_dir(dir);
            }
        }
        if handle.is_null() {
            eprintln!("flashinfer: dlopen({}) failed", path.display());
            return None;
        }
        let sym = CString::new(causal_sym).ok()?;
        let p = dlsym(handle, sym.as_ptr());
        if p.is_null() {
            eprintln!("flashinfer: {causal_sym} missing in {}", path.display());
            dlclose(handle);
            return None;
        }
        let func: PrefillFn = std::mem::transmute(p);
        let sym_nc = CString::new(noncausal_sym).ok()?;
        let p_nc = dlsym(handle, sym_nc.as_ptr());
        if p_nc.is_null() {
            eprintln!("flashinfer: {noncausal_sym} missing in {}", path.display());
            dlclose(handle);
            return None;
        }
        let func_noncausal: PrefillFn = std::mem::transmute(p_nc);
        Some((handle, func, func_noncausal))
    }
}

pub struct FlashInfer {
    handle: *mut c_void,
    func: PrefillFn,
    func_noncausal: PrefillFn,
    func_custom: Option<CustomFn>,
    func_ext: Option<ExtFn>,
    handle_fa3: *mut c_void,
    func_fa3: Option<PrefillFn>,
    func_noncausal_fa3: Option<PrefillFn>,
}

// The handles are opened once and only used on the engine's single CUDA stream.
unsafe impl Send for FlashInfer {}
unsafe impl Sync for FlashInfer {}

impl FlashInfer {
    /// Load the embedded libraries. Returns `None` if the FA2 library cannot be
    /// opened; callers then keep the native attention kernel. A missing or
    /// unloadable FA3 library only disables the Hopper path.
    pub fn load() -> Option<Arc<Self>> {
        let (handle, func, func_noncausal) = unsafe {
            open_lib(
                EMBEDDED_SO,
                "libfi_prefill.so",
                env!("DECIDRA_FI_PREFILL_SO"),
                "fi_prefill_bf16",
                "fi_prefill_bf16_noncausal",
            )?
        };
        let custom = unsafe {
            let sym = CString::new("fi_prefill_bf16_custom").ok()?;
            let ptr = dlsym(handle, sym.as_ptr());
            if ptr.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut c_void, CustomFn>(ptr))
            }
        };
        let ext = unsafe {
            let sym = CString::new("fi_prefill_bf16_ext").ok()?;
            let ptr = dlsym(handle, sym.as_ptr());
            if ptr.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut c_void, ExtFn>(ptr))
            }
        };
        let fa3 = unsafe {
            open_lib(
                EMBEDDED_FA3,
                "libfi_prefill_fa3.so",
                env!("DECIDRA_FI_PREFILL_FA3_SO"),
                "fi_prefill_bf16_fa3",
                "fi_prefill_bf16_noncausal_fa3",
            )
        };
        let (handle_fa3, func_fa3, func_noncausal_fa3) = match fa3 {
            Some((h, f, fnc)) => (h, Some(f), Some(fnc)),
            None => (std::ptr::null_mut(), None, None),
        };
        Some(Arc::new(Self {
            handle,
            func,
            func_noncausal,
            func_custom: custom,
            func_ext: ext,
            handle_fa3,
            func_fa3,
            func_noncausal_fa3,
        }))
    }

    pub fn has_fa3(&self) -> bool {
        self.func_fa3.is_some()
    }

    fn invoke<Q, K, V, O>(
        &self,
        func: PrefillFn,
        label: &str,
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
        let need_q = seq * 24 * 256;
        let need_kv = seq * 4 * 256;
        if q.len() < need_q || k.len() < need_kv || v.len() < need_kv || out.len() < need_q {
            bail!(
                "{label}: buffers too small for seq {seq}: q={} k={} v={} out={}",
                q.len(), k.len(), v.len(), out.len()
            );
        }
        let (qp, _g0) = q.device_ptr(&gpu.stream);
        let (kp, _g1) = k.device_ptr(&gpu.stream);
        let (vp, _g2) = v.device_ptr(&gpu.stream);
        let (op, _g3) = out.device_ptr_mut(&gpu.stream);
        let stream = gpu.stream.cu_stream() as *mut c_void;
        let rc = unsafe {
            func(
                qp as *const c_void,
                kp as *const c_void,
                vp as *const c_void,
                op as *mut c_void,
                seq as c_int,
                stream,
            )
        };
        if rc != 0 {
            bail!("{label} failed with cudaError {rc}");
        }
        Ok(())
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
        self.invoke(self.func, "FlashInfer prefill", gpu, q, k, v, out, seq)
    }

    /// Noncausal (full) attention over the whole sequence.
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
        self.invoke(
            self.func_noncausal,
            "FlashInfer noncausal prefill",
            gpu,
            q,
            k,
            v,
            out,
            seq,
        )
    }

    /// Hopper FA3 causal prefill.
    pub fn prefill_fa3<Q, K, V, O>(
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
        let Some(func) = self.func_fa3 else {
            bail!("FlashInfer FA3 causal prefill is not available");
        };
        self.invoke(func, "FlashInfer FA3 prefill", gpu, q, k, v, out, seq)
    }

    /// Hopper FA3 noncausal prefill.
    pub fn prefill_noncausal_fa3<Q, K, V, O>(
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
        let Some(func) = self.func_noncausal_fa3 else {
            bail!("FlashInfer FA3 noncausal prefill is not available");
        };
        self.invoke(
            func,
            "FlashInfer FA3 noncausal prefill",
            gpu,
            q,
            k,
            v,
            out,
            seq,
        )
    }

    /// Dense custom-mask prefill: `mask` is bit-packed `[seq, seq]`, 1 = visible.
    pub fn prefill_custom<Q, K, V, M, O>(
        &self,
        gpu: &crate::cuda::Gpu,
        q: &Q,
        k: &K,
        v: &V,
        mask: &M,
        out: &mut O,
        seq: usize,
    ) -> Result<()>
    where
        Q: DevicePtr<bf16>,
        K: DevicePtr<bf16>,
        V: DevicePtr<bf16>,
        M: DevicePtr<u8>,
        O: DevicePtrMut<bf16>,
    {
        let Some(func) = self.func_custom else {
            bail!("FlashInfer custom-mask prefill is not available");
        };
        let need_q = seq * 24 * 256;
        let need_kv = seq * 4 * 256;
        let need_mask = (seq * seq).div_ceil(8);
        if q.len() < need_q
            || k.len() < need_kv
            || v.len() < need_kv
            || mask.len() < need_mask
            || out.len() < need_q
        {
            bail!("custom prefill buffers too small for seq {seq}");
        }
        let (qp, _g0) = q.device_ptr(&gpu.stream);
        let (kp, _g1) = k.device_ptr(&gpu.stream);
        let (vp, _g2) = v.device_ptr(&gpu.stream);
        let (mp, _g3) = mask.device_ptr(&gpu.stream);
        let (op, _g4) = out.device_ptr_mut(&gpu.stream);
        let stream = gpu.stream.cu_stream() as *mut c_void;
        let rc = unsafe {
            func(
                qp as *const c_void,
                kp as *const c_void,
                vp as *const c_void,
                mp as *const c_void,
                op as *mut c_void,
                seq as c_int,
                stream,
            )
        };
        if rc != 0 {
            bail!("FlashInfer custom-mask prefill failed with cudaError {rc}");
        }
        Ok(())
    }

    /// Non-causal prefill with independent query/key lengths.
    pub fn prefill_ext<Q, K, V, O>(
        &self,
        gpu: &crate::cuda::Gpu,
        q: &Q,
        k: &K,
        v: &V,
        out: &mut O,
        qo_len: usize,
        kv_len: usize,
    ) -> Result<()>
    where
        Q: DevicePtr<bf16>,
        K: DevicePtr<bf16>,
        V: DevicePtr<bf16>,
        O: DevicePtrMut<bf16>,
    {
        let Some(func) = self.func_ext else {
            bail!("FlashInfer extended prefill is not available");
        };
        if q.len() < qo_len * 24 * 256
            || k.len() < kv_len * 4 * 256
            || v.len() < kv_len * 4 * 256
            || out.len() < qo_len * 24 * 256
        {
            bail!("extended prefill buffers too small for qo {qo_len} kv {kv_len}");
        }
        let (qp, _g0) = q.device_ptr(&gpu.stream);
        let (kp, _g1) = k.device_ptr(&gpu.stream);
        let (vp, _g2) = v.device_ptr(&gpu.stream);
        let (op, _g3) = out.device_ptr_mut(&gpu.stream);
        let stream = gpu.stream.cu_stream() as *mut c_void;
        let rc = unsafe {
            func(
                qp as *const c_void,
                kp as *const c_void,
                vp as *const c_void,
                op as *mut c_void,
                qo_len as c_int,
                kv_len as c_int,
                stream,
            )
        };
        if rc != 0 {
            bail!("FlashInfer extended prefill failed with cudaError {rc}");
        }
        Ok(())
    }

}

impl Drop for FlashInfer {
    fn drop(&mut self) {
        unsafe {
            if !self.handle_fa3.is_null() {
                dlclose(self.handle_fa3);
            }
            dlclose(self.handle);
        }
    }
}

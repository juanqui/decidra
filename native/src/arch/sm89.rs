//! Ada (RTX 4090 / L40S) tuning. SM89 also covers SM80 for the generic
//! vendored kernels.
use super::{ArchProfile, GpuArch};

pub fn profile(arch: GpuArch, sms: usize) -> ArchProfile {
    ArchProfile {
        arch,
        sms,
        shared_memory_bytes: 100 * 1024,
        max_threads_per_sm: 1536,
        supports_fa3: false,
        default_batch_rows: 32,
        default_state_rows: 64,
        flashinfer_min_seq: 1024,
        gdn_chunk_min_seq: 16,
        description: "Ada / Ampere W4A8 path; 32-row scheduler budget".to_owned(),
    }
}

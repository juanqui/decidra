//! Hopper (H100 / H200) tuning. BF16 is the measured d3-turbo path; FA3 is
//! available but opt-in because FA2 is faster on this workload.
use super::{ArchProfile, GpuArch};

pub fn profile(arch: GpuArch, sms: usize) -> ArchProfile {
    ArchProfile {
        arch,
        sms,
        shared_memory_bytes: 228 * 1024,
        max_threads_per_sm: 2048,
        supports_fa3: true,
        default_batch_rows: 32,
        default_state_rows: 64,
        flashinfer_min_seq: 1024,
        gdn_chunk_min_seq: 16,
        description: "Hopper BF16 path; FA2 default, FA3 opt-in".to_owned(),
    }
}

//! Blackwell workstation (RTX PRO 6000, GB202, SM120) tuning.
//!
//! SM120 is a hybrid ISA: fifth-generation tensor cores with FP4/FP8 support,
//! but SM80-style register `mma.sync` instead of SM90 WGMMA or SM100 tcgen05.
//! It has 99 KiB opt-in shared memory per block, 128 MiB L2, and no TMEM.
//! Decidra therefore uses the vendored SM80-style FlashInfer/Marlin kernels,
//! compiled for `sm_120`, and tunes the scheduler around the larger SM count.
use super::{ArchProfile, GpuArch};

pub fn profile(arch: GpuArch, sms: usize) -> ArchProfile {
    ArchProfile {
        arch,
        sms,
        // Opt-in dynamic shared memory limit for one SM120 CTA. The full
        // L1/shared pool is 128 KiB, but the usable CTA allocation is 99 KiB.
        shared_memory_bytes: 99 * 1024,
        max_threads_per_sm: 1536,
        supports_fa3: false,
        // Keep the measured accuracy-preserving budget by default. Larger
        // passes are available explicitly for throughput experiments.
        default_batch_rows: 32,
        default_state_rows: 64,
        flashinfer_min_seq: 1024,
        gdn_chunk_min_seq: 16,
        description:
            "Blackwell workstation SM120; register MMA + FP4 tensor cores, 99 KiB CTA smem"
                .to_owned(),
    }
}

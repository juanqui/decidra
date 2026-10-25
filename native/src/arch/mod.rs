//! Architecture detection and per-GPU tuning profiles.
//!
//! Kernel source stays in `native/kernels`; this module owns the host-side
//! differences between the GPU generations Decidra supports. A binary embeds
//! one cubin built for `DECIDRA_CUDA_ARCH`; `Gpu::new` verifies that the
//! embedded cubin matches the runtime device instead of silently loading SASS
//! for the wrong SM.

pub mod sm120;
pub mod sm89;
pub mod sm90;

/// Runtime architecture family, derived from the CUDA compute capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArch {
    Sm80,
    Sm89,
    Sm90,
    Sm120,
    Other { major: i32, minor: i32 },
}

impl GpuArch {
    pub fn from_compute(major: i32, minor: i32) -> Self {
        match (major, minor) {
            (8, 0) => Self::Sm80,
            (8, 9) => Self::Sm89,
            (9, _) => Self::Sm90,
            (12, _) => Self::Sm120,
            _ => Self::Other { major, minor },
        }
    }

    /// Parse the build-time `DECIDRA_CUDA_ARCH` spelling (`sm_120`, `sm_120a`).
    pub fn from_compile_arch(value: &str) -> Option<Self> {
        let digits = value
            .strip_prefix("sm_")
            .or_else(|| value.strip_prefix("compute_"))?;
        let digits = digits.trim_end_matches(|c: char| c.is_ascii_alphabetic());
        let (major, minor) = if digits.len() >= 3 {
            (&digits[..2], &digits[2..])
        } else {
            (&digits[..1], &digits[1..])
        };
        let major = major.parse().ok()?;
        let minor = minor.parse().ok()?;
        Some(Self::from_compute(major, minor))
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Sm80 => "SM80 (Ampere)",
            Self::Sm89 => "SM89 (Ada)",
            Self::Sm90 => "SM90 (Hopper)",
            Self::Sm120 => "SM120 (Blackwell workstation)",
            Self::Other { .. } => "unknown CUDA architecture",
        }
    }

    /// Normalized build/runtime label used in metrics (`sm_89`, `sm_120`, ...).
    /// This is deliberately distinct from [`Self::label`], which is for logs.
    pub fn arch_label(self) -> String {
        match self {
            Self::Sm80 => "sm_80".to_owned(),
            Self::Sm89 => "sm_89".to_owned(),
            Self::Sm90 => "sm_90".to_owned(),
            Self::Sm120 => "sm_120".to_owned(),
            Self::Other { major, minor } => format!("sm_{major}{minor}"),
        }
    }

    pub fn profile(self, sms: usize) -> ArchProfile {
        match self {
            Self::Sm80 => sm89::profile(self, sms),
            Self::Sm89 => sm89::profile(self, sms),
            Self::Sm90 => sm90::profile(self, sms),
            Self::Sm120 => sm120::profile(self, sms),
            Self::Other { major, minor } => ArchProfile {
                arch: self,
                sms,
                shared_memory_bytes: 48 * 1024,
                max_threads_per_sm: 1024,
                supports_fa3: false,
                default_batch_rows: 32,
                default_state_rows: 64,
                flashinfer_min_seq: 1024,
                gdn_chunk_min_seq: 16,
                description: format!("unknown SM {major}.{minor}"),
            },
        }
    }
}

/// Host-side tuning values that differ between supported architectures.
///
/// These are defaults, not hard limits: the server's environment variables
/// still win. Keeping them here makes an architecture-specific choice visible
/// in review instead of scattering magic numbers through the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchProfile {
    pub arch: GpuArch,
    pub sms: usize,
    pub shared_memory_bytes: usize,
    pub max_threads_per_sm: usize,
    pub supports_fa3: bool,
    pub default_batch_rows: usize,
    pub default_state_rows: usize,
    pub flashinfer_min_seq: usize,
    pub gdn_chunk_min_seq: usize,
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_supported_families() {
        assert_eq!(GpuArch::from_compute(8, 9), GpuArch::Sm89);
        assert_eq!(GpuArch::from_compute(9, 0), GpuArch::Sm90);
        assert_eq!(GpuArch::from_compute(12, 0), GpuArch::Sm120);
        assert_eq!(GpuArch::from_compute(12, 1), GpuArch::Sm120);
    }

    #[test]
    fn parses_compile_arch_spellings() {
        assert_eq!(GpuArch::from_compile_arch("sm_89"), Some(GpuArch::Sm89));
        assert_eq!(GpuArch::from_compile_arch("sm_120"), Some(GpuArch::Sm120));
        assert_eq!(GpuArch::from_compile_arch("sm_120a"), Some(GpuArch::Sm120));
        assert_eq!(
            GpuArch::from_compile_arch("compute_90"),
            Some(GpuArch::Sm90)
        );
        assert_eq!(GpuArch::from_compile_arch("nope"), None);
    }

    #[test]
    fn sm120_profile_has_blackwell_limits() {
        let p = GpuArch::Sm120.profile(188);
        assert_eq!(p.shared_memory_bytes, 99 * 1024);
        assert_eq!(p.max_threads_per_sm, 1536);
        assert!(!p.supports_fa3);
        assert_eq!(p.sms, 188);
    }
}

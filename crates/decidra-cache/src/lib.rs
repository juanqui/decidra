//! CUDA-free prefix-cache logic: mask algebra, cache keys, LRU and matching.
//!
//! See `docs/specs/2026-10-20-prefix-caching.md`. This crate is deliberately
//! dependency-light and has no GPU code so the correctness tests run in CI.

pub mod key;
pub mod lru;
pub mod mask;
pub mod page_table;

pub use key::{CacheKey, ExecVariant, NumericConfig, ProgramHash, SiteHiddenKey};
pub use lru::Lru;
pub use mask::{MaskSpec, merge_state, softmax_reference, visible};
pub use page_table::{PrefixEntry, PrefixIndex, PrefixKey, PrefixMatch};

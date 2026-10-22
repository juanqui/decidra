//! Native CUDA engine for `vllm-sr/d3-turbo` (`prefix_bidir`, separate F32
//! readout).
//!
//! Typical server use:
//! ```ignore
//! let engine = decidra_native::Engine::new(model_dir)?;
//! engine.preload_all()?;                      // resident weights
//! let cache = engine.capture_state(&state)?;  // state K/V + GDN state
//! let rows = engine.readout_state_once_batch(&cache, &question_blocks)?;
//! ```
//!
//! Move the engine to one blocking worker and serialize forwards; it is
//! `Send` but not for concurrent use.
pub mod arch;
pub mod config;
pub mod cuda;
pub mod engine;
pub mod flashinfer;
pub mod kernels;
pub mod tensor;
pub mod weights;

pub use engine::{Engine, HostStateCache, LogitsOpts, Stage};

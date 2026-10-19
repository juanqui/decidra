//! Native CUDA inference engine for JEV-27B (text path).
//!
//! Typical server use:
//! ```ignore
//! let engine = jev_native::Engine::new(model_dir)?;
//! engine.preload_all()?;                 // ~12.4 GiB packed weights resident
//! let hidden = engine.forward_last_hidden(&tokens)?;
//! let logits = engine.decision_logits(&tokens, &[3721, 1802])?;
//! ```
//!
//! Move the engine to one blocking worker (`spawn_blocking`) and serialize
//! forwards; it is `Send` but not for concurrent use.
pub mod config;
pub mod cuda;
pub mod engine;
pub mod kernels;
pub mod tensor;
pub mod weights;

pub use engine::{Engine, LogitsOpts, Stage};

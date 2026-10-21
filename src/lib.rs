//! Decidra decision server for the `JEV-27B-VL` AWQ W4A16 artifact.
//!
//! The HTTP/API layer, tokenization, prompt construction, decision-head math,
//! concurrency control, signing, and metrics live here. Inference runs in the
//! in-process `decidra-native` CUDA engine (feature `native`); tests use
//! [`backend::MockBackend`]. There is no HTTP upstream, Python, PyTorch, or
//! external model server on the runtime path.

pub mod api;
pub mod backend;
pub mod batch;
pub mod cache;
pub mod config;
pub mod d3;
pub mod engine;
pub mod error;
pub mod head;
#[cfg(feature = "native")]
pub mod native_backend;
pub mod prompt;
pub mod pyjson;
pub mod rng;
pub mod telemetry;

/// Wire protocol name shared with the fp8 stack and the Python shim.
pub const PROTOCOL: &str = "jev27-bare-v1";
/// Client-facing model name.
pub const PUBLIC_MODEL: &str = "autotrust/JEV-27B-VL";
/// Backwards-compatible alias exposed by `/v1/models`.
pub const PUBLIC_ALIAS: &str = "autotrust/JEV-27B";
/// Wide-choice ceiling; labels are A-P, Q-Z, AA.. up to this many single tokens.
pub const MAX_OPTIONS: usize = 256;
/// Trained A-P verbalizers in the decision head.
pub const NATIVE_CHOICE_OPTIONS: usize = 16;
/// The shim's serving default for choices above 16 options.
pub const DEFAULT_STRATEGY: &str = "tournament";

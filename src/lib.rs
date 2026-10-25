//! Decidra: state-once decision inference server for `vllm-sr/d3-turbo`.
//!
//! The HTTP/API layer, tokenization, `prefix_bidir` plan, and state-once
//! scheduling live here. Inference runs in the in-process `decidra-native`
//! CUDA engine (feature `native`). The engine keeps the state's post-RoPE
//! full-attention K/V, GDN recurrent state and convolution tail, then forks
//! every question from that cache in one batched question pass. No HTTP
//! upstream, Python, PyTorch, or external model server is on the runtime path.

pub mod api;
pub mod backend;
#[cfg(feature = "native")]
pub mod batch;
pub mod config;
pub mod d3;
pub mod error;
pub mod image_media;
#[cfg(feature = "native")]
pub mod native_backend;
pub mod pyjson;
pub mod telemetry;

/// Client-facing model name.
pub const PUBLIC_MODEL: &str = "vllm-sr/d3-turbo";
/// Readout rows in the d3-turbo code head.
pub const MAX_OPTIONS: usize = 255;
/// Maximum questions in one state-once request.
pub const MAX_QUESTIONS: usize = 64;
/// Maximum tokens per question's state+block in the serving path.
pub const MAX_STATE_TOKENS: usize = 32_768;

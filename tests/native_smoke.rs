//! Real GPU smoke test. Run explicitly:
//!
//! ```sh
//! scripts/native-smoke.sh
//! ```
//!
//! Ignored by default: it needs the AWQ checkpoint, the CUDA driver/cuBLAS, and
//! a free GPU selected with `CUDA_VISIBLE_DEVICES`.

#![cfg(feature = "native")]

use jev_engine::{backend::Backend, config::Config, native_backend::NativeBackend};

#[test]
#[ignore = "requires the AWQ checkpoint and GPU1 (use scripts/native-smoke.sh)"]
fn native_forward_produces_finite_head_logits() {
    let config = Config::from_env().unwrap();
    let backend = NativeBackend::load(&config).unwrap();
    let prompt = jev_engine::prompt::text_prompt(
        "noul",
        "native smoke test",
        "Is the native engine ready?",
        &["false".to_owned(), "true".to_owned()],
    );
    let ids = [3721u32, 1802];
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let logits = runtime.block_on(backend.logits(&prompt, &ids)).unwrap();
    assert_eq!(logits.len(), ids.len(), "logits: {logits:?}");
    assert!(
        logits.iter().all(|value| value.is_finite()),
        "non-finite logits: {logits:?}"
    );
    println!("head logits for {ids:?}: {logits:?}");
}

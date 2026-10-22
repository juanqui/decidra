# decidra-native

The CUDA engine behind [Decidra](../README.md). It serves exactly one model
family: **`vllm-sr/d3-turbo`** — a Qwen3.5 hybrid with `prefix_bidir` full
attention, Gated DeltaNet layers, merged BF16 or packed W4A8 weights, and a
separate F32 `[255, 5120]` readout.

Rust + custom CUDA kernels + cuBLAS. No Python or PyTorch is on the runtime
path.

## Execution contract

- `capture_state(tokens)` runs the state prefix once and stores, per full
  layer, post-RoPE K/V `[S, 4, 256]`; per GDN layer it stores the final f32
  `[48, 128, 128]` recurrent state and the last three pre-conv
  `in_proj_qkv` rows.
- `readout_state_once_batch(cache, questions)` runs all question blocks as one
  ragged batch: one GEMM per projection, one cascade attention launch per row,
  a seeded GDN recurrence, and one F32 readout.
- `decision_logits_split(tokens, split, codes)` is the per-row block reference
  used by the accuracy and state-once tests.
- The engine rejects any checkpoint that is not `attention_mode:
  prefix_bidir` or lacks `readout.safetensors`.

The state cache is valid only because no state query sees a question key. A
strict sub-prefix of a state is not a valid cache key.

## Attention

- **Full layers:** a non-causal state pass, then question queries over
  `[state ‖ own block]`. No state query sees a question.
- **GDN layers:** causal; short question blocks use the register-resident
  four-lane recurrence (default), longer blocks use the chunked tensor-core
  prefill.
- **Architecture profiles:** `src/arch/` selects SM89, SM90, or SM120 host
  tuning and verifies the embedded cubin against the runtime compute
  capability.
- **FlashInfer:** FA2 is the default on SM89 and SM90. FA3 is compiled in with
  `DECIDRA_FA3=1` and selected at runtime with `DECIDRA_FLASHINFER_FA3=1`; on the
  d3 2K–24K workload FA2 measured faster.

## Layout

```
native/
  build.rs              nvcc cubins, vendored Marlin, optional FlashInfer
  kernels/kernels.cu    RMSNorm, RoPE, attention, GDN, conv, readout
  kernels/marlin.cu     Marlin W4A16/W4A8 launcher and repack
  third_party/          Marlin, FlashInfer FA2/FA3, CCCL headers
  src/arch/              SM89/SM90/SM120 profiles and cubin guard
  src/config.rs         d3-turbo checkpoint contract
  src/weights.rs        merged BF16 / packed W4A8 loader
  src/engine.rs         layers, state capture, state-once batch, readout
  src/flashinfer.rs     FA2/FA3 dlopen and cascade entry points
  tests/                parity, state-once, negative-control, and perf probes
```

## Build

```bash
# SM89 (RTX 4090 / L40S)
cargo build --release --locked --features native

# SM90 (H100 / H200)
DECIDRA_CUDA_ARCH=sm_90 cargo build --release --locked --features native

# SM120 (RTX PRO 6000 Blackwell)
DECIDRA_CUDA_ARCH=sm_120 cargo build --release --locked --features native

# Optional FA3: CUTLASS headers plus the FA3 library
DECIDRA_FA3=1 \
DECIDRA_CUTLASS_INCLUDE="/path/cutlass/include:/path/cutlass/tools/util/include" \
DECIDRA_CUDA_ARCH=sm_90 cargo build --release --locked --features native
```

`build.rs` accepts `DECIDRA_NVCC` for a local `nvcc` and `DECIDRA_CUDA_IMAGE` for a
containerized CUDA toolkit. Cubins and the FlashInfer prefill library are
embedded in the binary.

## Environment

The server maps its configuration onto the engine. Engine-only knobs include:

| variable | default | meaning |
|---|---|---|
| `DECIDRA_NATIVE_CACHE` | on | keep packed weights resident |
| `DECIDRA_NATIVE_MARLIN` | on | Marlin W4A8 layout and kernel |
| `DECIDRA_NATIVE_INT8` | on | per-token INT8 activations |
| `DECIDRA_NATIVE_INT8_GROUP_SCALE` | on | per-64-group INT8 scales |
| `DECIDRA_NATIVE_GDN_CHUNKED` | on | chunked GDN prefill |
| `DECIDRA_GDN_REG` | `1` | register GDN for short question blocks |
| `DECIDRA_STATE_ONCE_MAX_ROWS` | `64` | question rows per coalesced pass |
| `DECIDRA_FLASHINFER_FA3` | off | opt-in Hopper FA3 |
| `DECIDRA_PROFILE`, `DECIDRA_CPU_PROFILE`, `DECIDRA_GDN_PROF` | off | timing dumps |

See the root [README](../README.md) for serving configuration, benchmarks, and
the full test matrix.

# FlashInfer FA2 prefill integration

Status: implemented in this worktree; runtime benchmark pending a free GPU1.

## What is vendored

- `include/flashinfer/**` — FlashInfer **0.7.0.post1** headers (Apache-2.0).
- `cccl/libcudacxx/include/**` — NVIDIA CCCL libcudacxx headers, needed by
  `flashinfer/fastdiv.cuh` for `cuda::fast_mod_div`.
- `fi_prefill.cu` — C ABI wrapper for the fixed engine shape:
  BF16 q `[S,24,256]`, k/v `[S,4,256]`, causal, NHD, head_dim 256,
  scale `1/sqrt(256)=0.0625`.
- `NOTICE.md`, `LICENSE-APACHE`.

No CUTLASS, CUB, TVM-FFI, Python, PyTorch, or ATen is used. The `.so` links
`libcudart_static`, so the host does not need a CUDA runtime library on
`LD_LIBRARY_PATH`; only the NVIDIA driver is required.

## Build

`native/build.rs` compiles `fi_prefill.cu` into
`OUT_DIR/libfi_prefill.so` with `-arch=sm_89` and sets
`DECIDRA_FI_PREFILL_SO`. It uses `DECIDRA_NVCC`, native `nvcc`, or the
`DECIDRA_CUDA_IMAGE` docker toolchain (same fallback as the cubins). Example:

```bash
DECIDRA_CUDA_IMAGE=vllm/vllm-openai:v0.31.0 \
  cargo build --release -p decidra-native
```

The build fails if the FlashInfer library cannot be produced.

## Rust FFI

`native/src/flashinfer.rs` uses `dlopen`/`dlsym` for `fi_prefill_bf16`
(no new crate dependency). `Engine::new` loads it once; if loading fails the
engine logs and keeps the native attention kernels. The wrapper is used only
when all of these hold:

- full-attention layer (the call sites are already the full-attention path),
- `seq >= DECIDRA_FLASHINFER_MIN_SEQ` (default **1024**),
- `head_dim == 256`, `n_heads == 24`, `n_kv == 4`,
- batch size 1 (the FFI is single-request; batched prefill falls back),
- `DECIDRA_FLASHINFER_ATTENTION` is not `0`.

Fallback: the existing tiled `attention_tc_bf16_d256` kernel (or the warp kernel
when selected) is used unchanged otherwise.

Environment:

```bash
DECIDRA_FLASHINFER_ATTENTION=0   # disable (default 1 if the .so loads)
DECIDRA_FLASHINFER_MIN_SEQ=1024  # threshold; measure 512/1024/2048
```

## Validation / benchmark commands

```bash
BIN=target/release/decidra-native
MODEL=/srv/models/jev-awq/awq-w4a16-v2
DATA=benchdata

# full 16K forward, FI off/on
DECIDRA_FLASHINFER_ATTENTION=0 DECIDRA_PROFILE=1 $BIN batch --model $MODEL \
  --tokens $DATA/long_16k.json --repeat 3 --head-ids 3721,1802 \
  --marlin --preload --out out16_off.json
DECIDRA_FLASHINFER_ATTENTION=1 DECIDRA_PROFILE=1 $BIN batch --model $MODEL \
  --tokens $DATA/long_16k.json --repeat 3 --head-ids 3721,1802 \
  --marlin --preload --out out16_on.json

# full-attention layer 3 output, 4K, FI off/on
DECIDRA_FLASHINFER_ATTENTION=0 $BIN layer --model $MODEL --tokens $DATA/long_4k.json \
  --layer 3 --stage layer --marlin --preload --out layer4k_off.f32
DECIDRA_FLASHINFER_ATTENTION=1 $BIN layer --model $MODEL --tokens $DATA/long_4k.json \
  --layer 3 --stage layer --marlin --preload --out layer4k_on.f32
$BIN compare --ref layer4k_off.f32 --got layer4k_on.f32

# suite28 (short prompts fall back; confirms no regression)
DECIDRA_FLASHINFER_ATTENTION=1 $BIN batch --model $MODEL \
  --tokens $DATA/suite_tokens.json --head-ids-json $DATA/suite_heads.json \
  --repeat 1 --marlin --preload --out suite_on.json
```

Expected: layer 3 attention output max abs ~2e-3 vs native; 16K attention
~20.5 ms vs 35.5 ms standalone (the full forward gain is only the attention
share). Do not enable for <1K without measuring: the launch/workspace overhead
is not amortized.

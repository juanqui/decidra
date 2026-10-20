# FlashInfer vendored headers

- Upstream: https://github.com/flashinfer-ai/flashinfer
- Version: **0.7.0.post1** (the version shipped in `vllm/vllm-openai:v0.31.0`)
- Vendored paths:
  - `include/flashinfer/**` — FlashInfer FA2 prefill headers, Apache-2.0.
  - `cccl/libcudacxx/include/**` — NVIDIA CCCL libcudacxx headers, Apache-2.0
    with LLVM exceptions, required by `flashinfer/fastdiv.cuh`
    (`cuda::fast_mod_div`).
- Only `flashinfer/attention/prefill.cuh` and its transitive dependencies are
  compiled. CUTLASS, CUB, TVM-FFI, and the Python/JIT binding sources are not
  vendored.
- `fi_prefill.cu` is a thin C ABI wrapper written for this engine; it uses the
  FlashInfer header templates with `--arch=sm_89`.
- Upstream license texts:
  - FlashInfer: Apache-2.0, see `LICENSE-APACHE`.
  - CCCL/libcudacxx: Apache-2.0 with LLVM exceptions,
    https://github.com/NVIDIA/cccl/blob/main/LICENSE
- Pin note: FlashInfer headers have no stable C ABI. Keep this copy pinned to
  0.7.0.post1 and re-validate the numerical output if it is upgraded.

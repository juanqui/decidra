# CUDA kernel layout

`kernels.cu` contains the shared Decidra forward kernels: RMSNorm/residual
fusion, RoPE, convolution, GDN preparation/recurrence, quantizers, embedding,
and the SM80-style `mma.sync` attention used by non-FlashInfer paths.
It is compiled once for the selected `DECIDRA_CUDA_ARCH`.

`marlin.cu` is the vendored Marlin W4 kernel. The SM120 default is BF16.
Marlin W4A8 remains available but its single-vs-batch shape delta is larger.
The opt-in SM120 FP8 path uses E4M3 weights/activations and cuBLASLt for
large question passes; state capture stays BF16.

Architecture policy lives in `native/src/arch/`:

| profile | target | measured path |
|---|---|---|
| `sm89` | RTX 4090 / L40S | W4A8 g64, Marlin |
| `sm90` | H100 / H200 | BF16, FA2 (FA3 opt-in) |
| `sm120` | RTX PRO 6000 Blackwell | BF16 default; opt-in FP8, register GDN |

SM120 is not SM100: it has fifth-generation tensor cores with FP4/FP8 but no
TMEM, tcgen05, WGMMA, or thread-block clusters. Its CTA shared-memory budget is
99 KiB, and Blackwell GeForce/WORKSTATION FP8 requires the TN layout. Those
constraints are the reason the SM120 path uses SM80-style `mma.sync` kernels and
cuBLAS rather than CUTLASS SM100 collectives.

The build embeds exactly one cubin. `Gpu::new` verifies the runtime compute
capability against `DECIDRA_COMPILED_ARCH` and refuses to run a binary built for
the wrong SM.

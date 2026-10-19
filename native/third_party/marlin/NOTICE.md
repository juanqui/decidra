# Marlin W4A16 kernel (vendored)

Files `marlin.cuh`, `marlin_dtypes.cuh`, `marlin_mma.h`, `marlin_template.h`,
`dequant.h`, `kernel.h`, and `core/scalar_type.hpp` are taken from
vLLM v0.31.0 `csrc/libtorch_stable/quantization/marlin/` and `csrc/core/`,
which in turn derive from Elias Frantar's Marlin
(https://github.com/IST-DASLab/marlin).

Copyright (C) Marlin.2024 Elias Frantar
Copyright contributors to the vLLM project

Licensed under the Apache License, Version 2.0. See `LICENSE-APACHE`.
Local changes: none to the vendored kernel sources; `torch/headeronly/util/Exception.h`
is replaced by a small shim (no PyTorch in this project), and `kernel_selector.h`
is generated for the single configuration used here
(BF16 activations, U4B8 weights, BF16 scales, group 64, SM89).

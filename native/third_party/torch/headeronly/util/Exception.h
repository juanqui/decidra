#pragma once
// Minimal shim for the header-only torch exception used by vLLM's
// scalar_type.hpp. This project links no PyTorch; the checks are build/host
// validation only.
#include <stdexcept>
#include <string>

#define STD_TORCH_CHECK(cond, ...)                       \
  do {                                                   \
    if (!(cond)) {                                       \
      throw std::runtime_error(std::string("check failed: ") + #cond); \
    }                                                    \
  } while (0)
#define STD_TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) STD_TORCH_CHECK(cond, ##__VA_ARGS__)

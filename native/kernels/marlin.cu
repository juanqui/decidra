// Marlin W4A16 (group 64, asymmetric) launcher and repack kernels.
// Vendored vLLM Marlin sources live in native/third_party/marlin; see the
// NOTICE there for attribution.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "core/scalar_type.hpp"
#include "kernel.h"
#include "marlin_template.h"

namespace marlin {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};
using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          int stages) {
  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  int tb_scales = tb_groups * tb_n * 2;
  return tb_scales * stages;
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, int has_zp, bool is_zp_float,
                          bool is_a_8bit, int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size = get_scales_cache_size(th_config, prob_m, prob_n, prob_k,
                                        num_bits, group_size, stages);
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, int has_zp, bool is_zp_float,
                     bool is_a_8bit, int stages, int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
      has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(const vllm::ScalarType a_type,
                                const vllm::ScalarType b_type,
                                const vllm::ScalarType c_type,
                                const vllm::ScalarType s_type,
                                int thread_m_blocks, int thread_n_blocks,
                                int thread_k_blocks, bool m_block_size_8,
                                bool has_zp, int group_blocks, int threads,
                                bool is_zp_float, int stages) {
  int num_bits = b_type.size_bits();
  auto kernel = MarlinDefault;

  #include "kernel_selector.h"

  return kernel;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, bool has_zp, bool is_zp_float, int is_a_8bit,
    int stages, int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                         num_bits, group_size, has_zp, is_zp_float, is_a_8bit,
                         stages, max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits,
        group_size, has_zp, is_zp_float, is_a_8bit, stages);

    int group_blocks = group_size == -1 ? -1 : group_size / 16;

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks,
        th_config.thread_n / 16, th_config.thread_k / 16, m_block_size_8,
        has_zp, group_blocks, th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    return {1, th_config};
  }

  return exec_cfg;
}

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, int prob_m,
               int prob_n, int prob_k, int lda, void* workspace,
               vllm::ScalarType const& a_type, vllm::ScalarType const& b_type,
               vllm::ScalarType const& c_type, vllm::ScalarType const& s_type,
               bool has_bias, bool has_zp, int num_groups, int group_size,
               int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  bool is_a_8bit = a_type.size_bits() == 8;
  STD_TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [",
                  prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks;
  if (group_size == -1) {
    group_blocks = -1;
  } else {
    group_blocks = group_size / 16;
    STD_TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                    " is not divisible by group_blocks = ", group_blocks);
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;

  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const float* g_s_ptr = (const float*)g_s;

  const int4* zp_ptr = (const int4*)zp;
  int* locks = (int*)workspace;

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  STD_TORCH_CHECK(max_shared_mem > 0);

  int major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         dev);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         dev);
  STD_TORCH_CHECK(major_capability * 10 + minor_capability >= 75,
                  "marlin kernel only support Turing or newer GPUs.");
  int stages = 4;
  if (major_capability == 7 && minor_capability == 5) {
    stages = 2;
    STD_TORCH_CHECK(a_type == vllm::kFloat16 || a_type == vllm::kS8,
                    "Turing only support FP16 or INT8 activation.");
  }
  if (a_type == vllm::kFE4M3fn) {
    STD_TORCH_CHECK(major_capability * 10 + minor_capability >= 89,
                    "FP8 only support Ada Lovelace or newer GPUs.");
    STD_TORCH_CHECK(
        major_capability * 10 + minor_capability == 89 ||
            major_capability == 12,
        "Marlin W4A8-FP8 only support SM89 or SM12x device (It is slower than "
        "Marlin W4A16 on other devices).");
  }

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
      exec_cfg = exec_config_t{1, thread_tfg};
      STD_TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                      " is not divisible by thread_n = ", thread_n);
      STD_TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                      " is not divisible by thread_k = ", thread_k);
    } else {
      // Auto config
      exec_cfg = determine_exec_config(
          a_type, b_type, c_type, s_type, prob_m_split, prob_n, prob_k,
          thread_m_blocks, m_block_size_8, num_bits, group_size, has_zp,
          is_zp_float, is_a_8bit, stages, max_shared_mem, sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_n != -1) {
        if (prob_n / thread_tfg.thread_n *
                div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
            sms) {
          if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split,
                              prob_n, prob_k, num_bits, group_size, has_zp,
                              is_zp_float, is_a_8bit, stages,
                              max_shared_mem_new)) {
            thread_tfg = {128, 64, 128};
            exec_cfg = {1, thread_tfg};
          }
        }
      }

      if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    STD_TORCH_CHECK(
        is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n,
                        prob_k, num_bits, group_size, has_zp, is_zp_float,
                        is_a_8bit, stages, max_shared_mem_new),
        "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
        ", thread_k = ", thread_tfg.thread_k,
        ", thread_n = ", thread_tfg.thread_n,
        ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m,
        ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
        ", prob_m_split = ", prob_m_split, ", group_size = ", group_size,
        ", has_zp = ", has_zp, ", is_zp_float = ", is_zp_float,
        ", stages = ", stages, ", max_shared_mem_new = ", max_shared_mem_new);

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
        thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
        is_zp_float, stages);

    if (kernel == MarlinDefault) {
      STD_TORCH_CHECK(
          false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n, ", ",
          prob_k, "]", ", num_groups = ", num_groups,
          ", group_size = ", group_size, ", prob_m_split = ", prob_m_split,
          ", thread_m_blocks = ", thread_m_blocks,
          ", thread_n_blocks = ", thread_n_blocks,
          ", thread_k_blocks = ", thread_k_blocks,
          ", num_threads = ", num_threads, ", num_bits = ", num_bits);
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_new);

    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new);
    // clang-format on

    bool is_a_8bit = a_type.size_bits() == 8;
    A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
    a_s_ptr += prob_m_split;
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}

}

namespace marlin {
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, true, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 16, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 16, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 16, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFE4M3fn.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFE4M3fn.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 16, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFE4M3fn.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 16, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFE4M3fn.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 16, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);

}

extern "C" int jev_marlin_gemm_bf16(
    const void* A, const void* B, void* C, const void* b_s, const void* zp,
    int prob_m, int prob_n, int prob_k, int lda, int num_groups, int group_size,
    int sms, cudaStream_t stream) {
  marlin::marlin_mm(
      A, B, C, nullptr, nullptr, nullptr, const_cast<void*>(b_s), nullptr,
      const_cast<void*>(zp), prob_m, prob_n, prob_k, lda, nullptr,
      vllm::kBFloat16, vllm::kU4, vllm::kBFloat16, vllm::kBFloat16,
      /*has_bias=*/false, /*has_zp=*/true, num_groups, group_size,
      /*dev=*/0, stream, /*thread_k_init=*/-1, /*thread_n_init=*/-1, sms,
      /*use_atomic_add=*/false, /*use_fp32_reduce=*/false,
      /*is_zp_float=*/false);
  return 0;
}

// ---------------------------------------------------------------------------
// Repack: CT packed [N, K/8] -> GPTQ [K/8, N] -> Marlin [K/16, N/2] (4-bit).
// ---------------------------------------------------------------------------
__global__ void transpose_packed_q4_kernel(const uint32_t* __restrict__ src,
                                           uint32_t* __restrict__ dst, int rows,
                                           int cols) {
  __shared__ uint32_t tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
#pragma unroll
  for (int j = 0; j < 32; j += 8) {
    if (x < rows && y + j < cols) tile[threadIdx.y + j][threadIdx.x] = src[(long)x * cols + y + j];
  }
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
  for (int j = 0; j < 32; j += 8) {
    if (x < cols && y + j < rows) dst[(long)x * rows + y + j] = tile[threadIdx.x][threadIdx.y + j];
  }
}

extern "C" int jev_marlin_transpose_q4(const void* src, void* dst, int rows,
                                       int cols, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((rows + 31) / 32, (cols + 31) / 32);
  transpose_packed_q4_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(src), reinterpret_cast<uint32_t*>(dst),
      rows, cols);
  return 0;
}

namespace marlin {

template <int const num_threads, int const num_bits, bool is_a_8bit>
__global__ void jev_gptq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr, uint32_t* __restrict__ out_ptr,
    int size_k, int size_n) {
  constexpr int pack_factor = 32 / num_bits;
  constexpr int target_tile_n_size = tile_n_size / (is_a_8bit ? 2 : 1);
  constexpr int target_tile_k_size = tile_k_size * (is_a_8bit ? 2 : 1);
  int k_tiles = size_k / target_tile_k_size;
  int n_tiles = size_n / target_tile_n_size;
  int block_k_tiles = div_ceil(k_tiles, gridDim.x);
  auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) return;
  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);
  auto wait_for_stage = [&]() {
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };
  extern __shared__ int4 sh[];
  int4* sh_pipe_ptr = sh;
  constexpr int tile_ints = target_tile_k_size / pack_factor;
  constexpr int stage_n_threads = target_tile_n_size / 4;
  constexpr int stage_k_threads = tile_ints;
  constexpr int stage_size = stage_k_threads * stage_n_threads;
  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) { cp_async_fence(); return; }
    int first_n = n_tile_id * target_tile_n_size;
    int4* sh_ptr = sh_pipe_ptr + stage_size * pipe;
    if (threadIdx.x < stage_size) {
      auto k_id = threadIdx.x / stage_n_threads;
      auto n_id = threadIdx.x % stage_n_threads;
      int first_k = k_tile_id * target_tile_k_size;
      int first_k_packed = first_k / pack_factor;
      cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                reinterpret_cast<int4 const*>(
                    &(b_q_weight_ptr[(first_k_packed + k_id) * size_n + first_n + (n_id * 4)])));
    }
    cp_async_fence();
  };
  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) return;
    auto warp_id = threadIdx.x / 32;
    auto th_id = threadIdx.x % 32;
    if (warp_id >= 4) return;
    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * (is_a_8bit ? 4 : 2);
    constexpr int tc_offsets[4] = {0, 1, 8, 9};
    int cur_n = (warp_id / (is_a_8bit ? 2 : 1)) * 16 + tc_col;
    constexpr int sh_stride = target_tile_n_size;
    constexpr uint32_t mask = (1 << num_bits) - 1;
    int4* sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);
    uint32_t vals[8];
    uint32_t b1_vals[tile_ints];
    uint32_t b2_vals[tile_ints];
#pragma unroll
    for (int i = 0; i < tile_ints; i++) {
      if constexpr (is_a_8bit) {
        b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i + (warp_id % 2) * 8];
      } else {
        b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i];
        b2_vals[i] = sh_stage_int_ptr[cur_n + 8 + sh_stride * i];
      }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int cur_elem = tc_row + (is_a_8bit ? i : tc_offsets[i]);
      int cur_int = cur_elem / pack_factor;
      int cur_pos = cur_elem % pack_factor;
      vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
      if constexpr (is_a_8bit)
        vals[4 + i] = (b1_vals[cur_int + tile_ints / 2] >> (cur_pos * num_bits)) & mask;
      else
        vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
    }
    constexpr int tile_size = target_tile_k_size * target_tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;
    if constexpr (!is_a_8bit && num_bits == 4) {
      int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) res |= vals[pack_idx[i]] << (i * 4);
      out_ptr[out_offset + th_id * 4 + warp_id] = res;
    } else if constexpr (is_a_8bit && num_bits == 4) {
      int pack_idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) res |= vals[pack_idx[i]] << (i * 4);
      out_ptr[out_offset + th_id * 4 + warp_id] = res;
    } else {
      constexpr int pack_idx[4] = {0, 2, 1, 3};
      uint32_t res1 = 0, res2 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int ii = is_a_8bit ? i : pack_idx[i];
        res1 |= vals[ii] << (i * 8);
        res2 |= vals[4 + ii] << (i * 8);
      }
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };
  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++)
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    wait_for_stage();
  };
#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;
    start_pipes(k_tile_id, n_tile_id);
    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id,
                        n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }
}

extern "C" int jev_marlin_repack_q4_a8(const void* gptq, void* out, int size_k,
                                       int size_n, int sms, cudaStream_t stream) {
  constexpr int shmem = repack_stages * (tile_k_size / 8) * (tile_n_size / 4) * 16;
  cudaFuncSetAttribute(jev_gptq_marlin_repack_kernel<repack_threads, 4, true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
  jev_gptq_marlin_repack_kernel<repack_threads, 4, true>
      <<<sms, repack_threads, shmem, stream>>>(
          reinterpret_cast<const uint32_t*>(gptq),
          reinterpret_cast<uint32_t*>(out), size_k, size_n);
  return 0;
}

extern "C" int jev_marlin_repack_q4(const void* gptq, void* out, int size_k,
                                    int size_n, int sms, cudaStream_t stream) {
  constexpr int shmem = repack_stages * (tile_k_size / 8) * (tile_n_size / 4) * 16;
  cudaFuncSetAttribute(jev_gptq_marlin_repack_kernel<repack_threads, 4, false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
  jev_gptq_marlin_repack_kernel<repack_threads, 4, false>
      <<<sms, repack_threads, shmem, stream>>>(
          reinterpret_cast<const uint32_t*>(gptq),
          reinterpret_cast<uint32_t*>(out), size_k, size_n);
  return 0;
}
}

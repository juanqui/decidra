// Standalone FlashInfer FA3 (Hopper) prefill for BF16 q[S,24,256], k/v[S,4,256].
//
// This is the SM90a companion to `fi_prefill.cu` (FA2). It is compiled
// separately with -arch=sm_90a and uses CUTLASS/CuTe WGMMA + TMA loads.
// `build.rs` embeds the resulting library when DECIDRA_FA3_SO is supplied.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>

#include <flashinfer/attention/hopper/default_params.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>

using bf16 = cutlass::bfloat16_t;
using Params = flashinfer::SinglePrefillParams<bf16, bf16, bf16>;
using Variant = flashinfer::DefaultAttention<false>;

static Params make_params(const void* q, const void* k, const void* v, void* out,
                          int seq, bool causal) {
  Params p{};
  p.q_ptr = const_cast<bf16*>(static_cast<const bf16*>(q));
  p.k_ptr = const_cast<bf16*>(static_cast<const bf16*>(k));
  p.v_ptr = const_cast<bf16*>(static_cast<const bf16*>(v));
  p.o_ptr = static_cast<bf16*>(out);
  p.lse_ptr = nullptr;
  p.additional_params = {0.0f, 0.0625f, nullptr, nullptr, nullptr};
  p.q_stride_n = 24 * 256;
  p.k_stride_n = 4 * 256;
  p.v_stride_n = 4 * 256;
  p.o_stride_n = 24 * 256;
  p.q_stride_h = 256;
  p.k_stride_h = 256;
  p.v_stride_h = 256;
  p.o_stride_h = 256;
  p.qo_len = seq;
  p.kv_len = seq;
  p.num_qo_heads = 24;
  p.num_kv_heads = 4;
  p.group_size = 6;
  p.window_left = -1;
  p.causal = causal;
  return p;
}

extern "C" __attribute__((visibility("default")))
int fi_prefill_bf16_fa3(const void* q, const void* k, const void* v, void* out,
                        int seq, cudaStream_t stream) {
  if (seq <= 0) return -1;
  Params p = make_params(q, k, v, out, seq, true);
  cudaError_t e = flashinfer::SinglePrefillWithKVCacheDispatched<
      256, 256, flashinfer::MaskMode::kCausal, false, Variant>(p, stream);
  return static_cast<int>(e);
}

extern "C" __attribute__((visibility("default")))
int fi_prefill_bf16_noncausal_fa3(const void* q, const void* k, const void* v,
                                  void* out, int seq, cudaStream_t stream) {
  if (seq <= 0) return -1;
  Params p = make_params(q, k, v, out, seq, false);
  cudaError_t e = flashinfer::SinglePrefillWithKVCacheDispatched<
      256, 256, flashinfer::MaskMode::kNone, false, Variant>(p, stream);
  return static_cast<int>(e);
}

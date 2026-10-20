// Standalone FlashInfer FA2 prefill for BF16 q[S,24,256], k/v[S,4,256], causal.
// No PyTorch / ATen / TVM-FFI linkage: only FlashInfer headers + cudart.
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>

using bf16 = __nv_bfloat16;
using Params = flashinfer::SinglePrefillParams<bf16, bf16, bf16>;
using Variant = flashinfer::DefaultAttention<false, false, false, false>;

extern "C" __attribute__((visibility("default")))
int fi_prefill_bf16(const void* q, const void* k, const void* v, void* out,
                    int seq, cudaStream_t stream) {
  if (seq <= 0) return -1;
  Params p;
  p.q = const_cast<bf16*>(static_cast<const bf16*>(q));
  p.k = const_cast<bf16*>(static_cast<const bf16*>(k));
  p.v = const_cast<bf16*>(static_cast<const bf16*>(v));
  p.o = static_cast<bf16*>(out);
  p.lse = nullptr;
  p.maybe_alibi_slopes = nullptr;
  p.maybe_custom_mask = nullptr;
  p.qo_len = static_cast<uint32_t>(seq);
  p.kv_len = static_cast<uint32_t>(seq);
  p.num_qo_heads = 24;
  p.num_kv_heads = 4;
  p.group_size = flashinfer::uint_fastdiv(24 / 4);
  p.q_stride_n = 24 * 256;
  p.q_stride_h = 256;
  p.k_stride_n = 4 * 256;
  p.k_stride_h = 256;
  p.v_stride_n = 4 * 256;
  p.v_stride_h = 256;
  p.head_dim = 256;
  p.window_left = -1;
  p.logits_soft_cap = 0.0f;
  p.sm_scale = 0.0625f;   // 1/sqrt(256)
  p.rope_rcp_scale = 1.0f;
  p.rope_rcp_theta = 1.0f;
  p.partition_kv = false;
  cudaError_t e = flashinfer::SinglePrefillWithKVCacheDispatched<
      256, 256, flashinfer::PosEncodingMode::kNone,
      /*USE_FP16_QK_REDUCTION=*/false, flashinfer::MaskMode::kCausal,
      Variant>(p, nullptr, stream);
  return static_cast<int>(e);
}

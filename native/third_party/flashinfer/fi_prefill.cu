// Standalone FlashInfer FA2 prefill for BF16 q[S,24,256], k/v[S,4,256].
// No PyTorch / ATen / TVM-FFI linkage: only FlashInfer headers + cudart.
// Two entries: causal and noncausal (d3-turbo full layers).
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>

using bf16 = __nv_bfloat16;
using Params = flashinfer::SinglePrefillParams<bf16, bf16, bf16>;
using Variant = flashinfer::DefaultAttention<false, false, false, false>;

static Params make_params(const void* q, const void* k, const void* v, void* out, int seq) {
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
  return p;
}


// FlashInfer splits the KV range only when it receives a workspace. A 50-query
// cascade launch otherwise uses ~20 CTAs; a 5-20 MiB workspace gives it 6-12
// KV chunks and makes the state K/V read parallel. Large-q launches (the state
// pass) already have enough CTAs and keep the no-workspace path.
static bf16* split_workspace(int qo_len, int kv_len, cudaStream_t stream) {
  if (qo_len > 256 || kv_len < 512) return nullptr;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int q_tiles = (qo_len * 6 + 63) / 64;
  int chunks = (2 * sms) / (4 * q_tiles) + 1;
  if (chunks < 2) chunks = 2;
  if (chunks > 128) chunks = 128;
  int by_kv = (kv_len + 255) / 256 + 1;
  if (chunks > by_kv) chunks = by_kv;
  size_t elems = (size_t)chunks * qo_len * 24 * 256;
  size_t bytes = elems * sizeof(bf16) +
                 (size_t)chunks * qo_len * 24 * sizeof(float);
  void* ptr = nullptr;
  if (cudaMallocAsync(&ptr, bytes, stream) != cudaSuccess) return nullptr;
  return static_cast<bf16*>(ptr);
}

static void free_workspace(bf16* ptr, cudaStream_t stream) {
  if (ptr != nullptr) cudaFreeAsync(ptr, stream);
}

extern "C" __attribute__((visibility("default")))
int fi_prefill_bf16(const void* q, const void* k, const void* v, void* out,
                    int seq, cudaStream_t stream) {
  if (seq <= 0) return -1;
  Params p = make_params(q, k, v, out, seq);
  cudaError_t e = flashinfer::SinglePrefillWithKVCacheDispatched<
      256, 256, flashinfer::PosEncodingMode::kNone,
      /*USE_FP16_QK_REDUCTION=*/false, flashinfer::MaskMode::kCausal,
      Variant>(p, nullptr, stream);
  return static_cast<int>(e);
}

extern "C" __attribute__((visibility("default")))
int fi_prefill_bf16_noncausal(const void* q, const void* k, const void* v, void* out,
                              int seq, cudaStream_t stream) {
  if (seq <= 0) return -1;
  Params p = make_params(q, k, v, out, seq);
  cudaError_t e = flashinfer::SinglePrefillWithKVCacheDispatched<
      256, 256, flashinfer::PosEncodingMode::kNone,
      /*USE_FP16_QK_REDUCTION=*/false, flashinfer::MaskMode::kNone,
      Variant>(p, nullptr, stream);
  return static_cast<int>(e);
}

// ---------------------------------------------------------------------------
// Custom-mask and cascade prefill.
//
// `mask` is bit-packed [qo_len, kv_len] (bit i of byte k = mask[k*8+i]),
// 1 = key visible. The non-causal d3-turbo layout is a dense block mask:
// state queries see only the state; question queries see the whole row.
// ---------------------------------------------------------------------------
using MaskVariant = flashinfer::DefaultAttention<true, false, false, false>;

static Params make_params_ext(const void* q, const void* k, const void* v, void* out,
                              const void* mask, int qo_len, int kv_len) {
  Params p = make_params(q, k, v, out, qo_len);
  p.maybe_custom_mask = const_cast<uint8_t*>(static_cast<const uint8_t*>(mask));
  p.qo_len = static_cast<uint32_t>(qo_len);
  p.kv_len = static_cast<uint32_t>(kv_len);
  return p;
}

extern "C" __attribute__((visibility("default")))
int fi_prefill_bf16_custom(const void* q, const void* k, const void* v, const void* mask,
                           void* out, int seq, cudaStream_t stream) {
  if (seq <= 0) return -1;
  Params p = make_params_ext(q, k, v, out, mask, seq, seq);
  cudaError_t e = flashinfer::SinglePrefillWithKVCacheDispatched<
      256, 256, flashinfer::PosEncodingMode::kNone,
      /*USE_FP16_QK_REDUCTION=*/false, flashinfer::MaskMode::kCustom,
      MaskVariant>(p, nullptr, stream);
  return static_cast<int>(e);
}


// Non-causal prefill with independent query/key lengths (cascade without a
// dense mask): qo_len question rows over kv_len = state + question keys.
extern "C" __attribute__((visibility("default")))
int fi_prefill_bf16_ext(const void* q, const void* k, const void* v, void* out,
                        int qo_len, int kv_len, cudaStream_t stream) {
  if (qo_len <= 0 || kv_len < qo_len) return -1;
  Params p = make_params(q, k, v, out, qo_len);
  p.qo_len = static_cast<uint32_t>(qo_len);
  p.kv_len = static_cast<uint32_t>(kv_len);
  bf16* tmp = split_workspace(qo_len, kv_len, stream);
  cudaError_t e = flashinfer::SinglePrefillWithKVCacheDispatched<
      256, 256, flashinfer::PosEncodingMode::kNone,
      /*USE_FP16_QK_REDUCTION=*/false, flashinfer::MaskMode::kNone,
      Variant>(p, tmp, stream);
  free_workspace(tmp, stream);
  return static_cast<int>(e);
}

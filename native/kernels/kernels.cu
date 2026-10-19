// Native CUDA kernels for the JEV engine (W4A16 Qwen3.5-family forward).
// Compiled as a single cubin for sm_89 by native/build.rs.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float bf2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2bf(float x) { return __float2bfloat16(x); }
// round a float through bf16 (emulates a bf16 elementwise op in torch)
__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
  int d = threadIdx.x;
  scratch[d] = v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (d < off) scratch[d] += scratch[d + off];
    __syncthreads();
  }
  return scratch[0];
}

// ---------------------------------------------------------------------------
// Milestone 1: bootstrap / device proof
// ---------------------------------------------------------------------------
extern "C" __global__ void self_test_kernel(const bf16* __restrict__ in,
                                            bf16* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = bf2f(in[i]);
    out[i] = f2bf(2.0f * x + 1.0f);
  }
}

// ---------------------------------------------------------------------------
// RMSNorm. One block per row.
//   zero_centered=1: effective weight = 1 + w   (Qwen3_5RMSNorm)
//   zero_centered=0: effective weight = w       (Qwen3_5RMSNormGated without gate)
// ---------------------------------------------------------------------------
extern "C" __global__ void rmsnorm_bf16(const bf16* __restrict__ x,
                                        const bf16* __restrict__ w,
                                        bf16* __restrict__ out, int cols,
                                        float eps, int zero_centered) {
  extern __shared__ float red[];
  int row = blockIdx.x;
  const bf16* xr = x + (size_t)row * cols;
  bf16* orow = out + (size_t)row * cols;
  float ss = 0.f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    float v = bf2f(xr[j]);
    ss += v * v;
  }
  float mean = block_reduce_sum(ss, red) / (float)cols;
  float scale = 1.0f / sqrtf(mean + eps);
  __syncthreads();
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    float wv = bf2f(w[j]) + (zero_centered ? 1.0f : 0.0f);
    orow[j] = f2bf(bf2f(xr[j]) * scale * wv);
  }
}

// Gated RMSNorm: out = bf16( bf16(norm(x) * w) * silu(gate) ), matching
// Qwen3_5RMSNormGated (weight multiply is a bf16 op, then f32 gate).
extern "C" __global__ void rmsnorm_gated_bf16(const bf16* __restrict__ x,
                                              const bf16* __restrict__ gate,
                                              const bf16* __restrict__ w,
                                              bf16* __restrict__ out, int cols,
                                              float eps) {
  extern __shared__ float red[];
  int row = blockIdx.x;
  const bf16* xr = x + (size_t)row * cols;
  const bf16* gr = gate + (size_t)row * cols;
  bf16* orow = out + (size_t)row * cols;
  float ss = 0.f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    float v = bf2f(xr[j]);
    ss += v * v;
  }
  float mean = block_reduce_sum(ss, red) / (float)cols;
  float scale = 1.0f / sqrtf(mean + eps);
  __syncthreads();
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    float normed = rb(bf2f(xr[j]) * scale);      // normed cast to bf16 by torch
    float scaled = rb(normed * bf2f(w[j]));       // bf16 weight multiply
    float g = bf2f(gr[j]);
    float silu = g / (1.0f + expf(-g));
    orow[j] = f2bf(scaled * rb(silu));            // silu cast to bf16 by torch
  }
}

// ---------------------------------------------------------------------------
// Elementwise, bf16-rounded like the torch ops they replace.
// ---------------------------------------------------------------------------
extern "C" __global__ void add_bf16(const bf16* __restrict__ a,
                                    const bf16* __restrict__ b,
                                    bf16* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = f2bf(bf2f(a[i]) + bf2f(b[i]));
}

extern "C" __global__ void silu_mul_bf16(const bf16* __restrict__ gate,
                                         const bf16* __restrict__ up,
                                         bf16* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = bf2f(gate[i]);
    float s = rb(g / (1.0f + expf(-g)));
    out[i] = f2bf(s * bf2f(up[i]));
  }
}

// attn_output * sigmoid(gate)
extern "C" __global__ void sigmoid_gate_bf16(const bf16* __restrict__ x,
                                             const bf16* __restrict__ gate,
                                             bf16* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float s = rb(1.0f / (1.0f + expf(-bf2f(gate[i]))));
    out[i] = f2bf(bf2f(x[i]) * s);
  }
}

// Split a fused q_proj output [seq, heads, 2*hd] into q [seq, heads*hd] and
// gate [seq, heads*hd].
extern "C" __global__ void split_q_gate_bf16(const bf16* __restrict__ qg,
                                             bf16* __restrict__ q,
                                             bf16* __restrict__ gate, int seq,
                                             int heads, int hd) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq * heads * hd;
  if (idx >= total) return;
  int d = idx % hd;
  int h = (idx / hd) % heads;
  int t = idx / (hd * heads);
  size_t src = ((size_t)t * heads + h) * (2 * hd) + d;
  q[idx] = qg[src];
  gate[idx] = qg[src + hd];
}

// ---------------------------------------------------------------------------
// RoPE: first `rhalf*2` dims of each head rotate in NeoX pairs (i, i+rhalf).
// cos/sin are bf16 [seq, rhalf].
// grid: (seq, n_q_heads + n_k_heads), block: rhalf
// ---------------------------------------------------------------------------
extern "C" __global__ void rope_bf16(bf16* __restrict__ q, bf16* __restrict__ k,
                                     const bf16* __restrict__ cos,
                                     const bf16* __restrict__ sin, int n_q_heads,
                                     int n_k_heads, int hd, int rhalf) {
  int t = blockIdx.x;
  int hz = blockIdx.y;
  bf16* base;
  if (hz < n_q_heads)
    base = q + ((size_t)t * n_q_heads + hz) * hd;
  else
    base = k + ((size_t)t * n_k_heads + (hz - n_q_heads)) * hd;
  for (int i = threadIdx.x; i < rhalf; i += blockDim.x) {
    float c = bf2f(cos[t * rhalf + i]);
    float s = bf2f(sin[t * rhalf + i]);
    float a = bf2f(base[i]);
    float b = bf2f(base[i + rhalf]);
    float y0 = rb(rb(a * c) - rb(b * s));
    float y1 = rb(rb(b * c) + rb(a * s));
    base[i] = f2bf(y0);
    base[i + rhalf] = f2bf(y1);
  }
}

// ---------------------------------------------------------------------------
// Causal GQA attention. One block per (query position, head).
// q: [seq, n_heads, hd]; k,v: [seq, n_kv, hd]; out: [seq, n_heads, hd]
// grid: (seq, n_heads), block: 128 threads, dynamic shared.
// ---------------------------------------------------------------------------
extern "C" __global__ void attention_bf16(const bf16* __restrict__ q,
                                          const bf16* __restrict__ k,
                                          const bf16* __restrict__ v,
                                          bf16* __restrict__ out, int seq,
                                          int n_heads, int n_kv, int hd,
                                          float scale) {
  extern __shared__ float smem[];
  float* q_s = smem;             // hd
  float* acc = q_s + hd;         // hd
  float* red = acc + hd;         // blockDim.x
  float* sc = red + blockDim.x;  // [m, l, alpha, p]
  int t = blockIdx.x;
  int h = blockIdx.y;
  int kvh = h / (n_heads / n_kv);
  int d = threadIdx.x;

  for (int j = d; j < hd; j += blockDim.x)
    q_s[j] = bf2f(q[((size_t)t * n_heads + h) * hd + j]);
  for (int j = d; j < hd; j += blockDim.x) acc[j] = 0.f;
  if (d == 0) {
    sc[0] = -INFINITY;  // m
    sc[1] = 0.f;        // l
  }
  __syncthreads();

  for (int s = 0; s <= t; ++s) {
    const bf16* kb = k + ((size_t)s * n_kv + kvh) * hd;
    float part = 0.f;
    for (int j = d; j < hd; j += blockDim.x) part += q_s[j] * bf2f(kb[j]);
    float dot = block_reduce_sum(part, red);
    if (d == 0) {
      float m_old = sc[0], l_old = sc[1];
      float m_new = fmaxf(m_old, dot * scale);
      float alpha = expf(m_old - m_new);
      float p = expf(dot * scale - m_new);
      sc[0] = m_new;
      sc[1] = l_old * alpha + p;
      sc[2] = alpha;
      sc[3] = p;
    }
    __syncthreads();
    float alpha = sc[2], p = sc[3];
    const bf16* vb = v + ((size_t)s * n_kv + kvh) * hd;
    for (int j = d; j < hd; j += blockDim.x)
      acc[j] = acc[j] * alpha + p * bf2f(vb[j]);
    __syncthreads();
  }
  float inv_l = 1.f / sc[1];
  for (int j = d; j < hd; j += blockDim.x)
    out[((size_t)t * n_heads + h) * hd + j] = f2bf(acc[j] * inv_l);
}

// ---------------------------------------------------------------------------
// Causal GQA attention, one warp per query position. Lanes hold 8 head dims in
// registers; a warp shuffle reduces the dot product, so there is no block
// synchronization inside the key loop. This replaces the block-per-query kernel
// for long contexts.
// grid: (n_heads, ceil(seq/4)), block: 128
// ---------------------------------------------------------------------------
extern "C" __global__ void attention_warp_bf16(
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ v, bf16* __restrict__ out, int seq, int n_heads,
    int n_kv, int hd, float scale) {
  int h = blockIdx.x;
  int qpos = blockIdx.y * 4 + (threadIdx.x >> 5);
  if (qpos >= seq) return;
  int lane = threadIdx.x & 31;
  int kvh = h / (n_heads / n_kv);
  const int stride = 32;

  float qreg[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int d = lane + stride * i;
    qreg[i] = (d < hd) ? bf2f(q[((size_t)qpos * n_heads + h) * hd + d]) : 0.f;
  }
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  float m = -INFINITY, l = 0.f;

  for (int sx = 0; sx <= qpos; ++sx) {
    const bf16* kb = k + ((size_t)sx * n_kv + kvh) * hd;
    float part = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int d = lane + stride * i;
      if (d < hd) part += qreg[i] * bf2f(kb[d]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      part += __shfl_xor_sync(0xffffffffu, part, off);
    float score = part * scale;
    float m_new = fmaxf(m, score);
    float alpha = expf(m - m_new);
    float p = expf(score - m_new);
    m = m_new;
    l = l * alpha + p;
    const bf16* vb = v + ((size_t)sx * n_kv + kvh) * hd;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int d = lane + stride * i;
      if (d < hd) acc[i] = acc[i] * alpha + p * bf2f(vb[d]);
    }
  }
  float inv = 1.f / l;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int d = lane + stride * i;
    if (d < hd) out[((size_t)qpos * n_heads + h) * hd + d] = f2bf(acc[i] * inv);
  }
}

// ---------------------------------------------------------------------------
// Depthwise causal conv (kernel 4, padding 3) + SiLU.
// x: [seq, ch] (pre-conv), w: [ch, 4], out: [seq, ch]
// ---------------------------------------------------------------------------
extern "C" __global__ void conv1d_silu_bf16(const bf16* __restrict__ x,
                                            const bf16* __restrict__ w,
                                            bf16* __restrict__ out, int seq,
                                            int ch) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ch) return;
  for (int t = 0; t < seq; ++t) {
    float acc = 0.f;
    for (int j = 0; j < 4; ++j) {
      int src = t - 3 + j;
      if (src >= 0) acc += bf2f(x[(size_t)src * ch + c]) * bf2f(w[c * 4 + j]);
    }
    float v = rb(acc);  // conv output is bf16
    out[(size_t)t * ch + c] = f2bf(v / (1.0f + expf(-v)));
  }
}

// ---------------------------------------------------------------------------
// GDN preparation: split q/k/v, L2-normalize q/k (f32), scale q, and compute
// g (log decay) / beta.
// qkv: [seq, 2*key_dim + value_dim] bf16, rows are the conv+silu output.
// grid: (seq, num_v_heads), block: 128
// ---------------------------------------------------------------------------
extern "C" __global__ void gdn_prepare(const bf16* __restrict__ qkv,
                                       const bf16* __restrict__ b,
                                       const bf16* __restrict__ a,
                                       const float* __restrict__ A_log,
                                       const float* __restrict__ dt_bias,
                                       float* __restrict__ qf,
                                       float* __restrict__ kf,
                                       float* __restrict__ vf,
                                       float* __restrict__ g,
                                       float* __restrict__ beta, int num_k,
                                       int num_v, int dk, int dv) {
  extern __shared__ float red[];
  int t = blockIdx.x;
  int vh = blockIdx.y;
  int rep = num_v / num_k;
  int kh = vh / rep;
  int key_dim = num_k * dk;
  int row_stride = 2 * key_dim + num_v * dv;
  const bf16* row = qkv + (size_t)t * row_stride;

  float ss = 0.f;
  for (int j = threadIdx.x; j < dk; j += blockDim.x) {
    float qv = bf2f(row[kh * dk + j]);
    ss += qv * qv;
  }
  float qn = block_reduce_sum(ss, red);
  float qscale = (1.0f / sqrtf(qn + 1e-6f)) * (1.0f / sqrtf((float)dk));
  __syncthreads();
  for (int j = threadIdx.x; j < dk; j += blockDim.x)
    qf[((size_t)t * num_v + vh) * dk + j] =
        bf2f(row[kh * dk + j]) * qscale;

  ss = 0.f;
  for (int j = threadIdx.x; j < dk; j += blockDim.x) {
    float kv = bf2f(row[key_dim + kh * dk + j]);
    ss += kv * kv;
  }
  float kn = block_reduce_sum(ss, red);
  float kscale = 1.0f / sqrtf(kn + 1e-6f);
  __syncthreads();
  for (int j = threadIdx.x; j < dk; j += blockDim.x)
    kf[((size_t)t * num_v + vh) * dk + j] =
        bf2f(row[key_dim + kh * dk + j]) * kscale;

  for (int j = threadIdx.x; j < dv; j += blockDim.x)
    vf[((size_t)t * num_v + vh) * dv + j] =
        bf2f(row[2 * key_dim + vh * dv + j]);

  if (threadIdx.x == 0) {
    float av = bf2f(a[t * num_v + vh]) + dt_bias[vh];
    // softplus in f32, stable form
    float sp = av > 20.f ? av : log1pf(expf(av));
    g[t * num_v + vh] = -expf(A_log[vh]) * sp;
    // torch computes sigmoid on the bf16 projection, so the result is rounded
    // to bf16 before the rule converts it to f32.
    beta[t * num_v + vh] = rb(1.f / (1.f + expf(-bf2f(b[t * num_v + vh]))));
  }
}

// ---------------------------------------------------------------------------
// GDN recurrent (gated delta rule), float32 state.
// qf,kf,vf: [seq, heads, d]; g,beta: [seq, heads]; state: [heads, dk, dv]
// out: bf16 [seq, heads, dv] (torch casts core_attn_out to bf16 here)
// One block per value head, 128 threads; thread j owns value dimension j.
// ---------------------------------------------------------------------------
extern "C" __global__ void gdn_recurrent(const float* __restrict__ qf,
                                         const float* __restrict__ kf,
                                         const float* __restrict__ vf,
                                         const float* __restrict__ g,
                                         const float* __restrict__ beta,
                                         float* __restrict__ state,
                                         bf16* __restrict__ out, int seq,
                                         int heads, int dk, int dv) {
  int h = blockIdx.x;
  int j = threadIdx.x;
  if (j >= dv) return;
  float* S = state + (size_t)h * dk * dv;
  for (int t = 0; t < seq; ++t) {
    float dec = expf(g[t * heads + h]);
    float bt = beta[t * heads + h];
    const float* kt = kf + ((size_t)t * heads + h) * dk;
    // Decay the state BEFORE reading it: kv_mem is computed from the decayed
    // state in the reference rule.
    for (int i = 0; i < dk; ++i) S[(size_t)i * dv + j] *= dec;
    float kv = 0.f;
    for (int i = 0; i < dk; ++i) kv += S[(size_t)i * dv + j] * kt[i];
    float del = (vf[((size_t)t * heads + h) * dv + j] - kv) * bt;
    float o = 0.f;
    const float* qt = qf + ((size_t)t * heads + h) * dk;
    for (int i = 0; i < dk; ++i) {
      float s = S[(size_t)i * dv + j] + kt[i] * del;
      S[(size_t)i * dv + j] = s;
      o += s * qt[i];
    }
    out[((size_t)t * heads + h) * dv + j] = f2bf(o);
  }
}

// ---------------------------------------------------------------------------
// compressed-tensors pack-quantized W4A16 dequantization.
// packed: I32 [out, ceil(in/8)], nibble k at bits 4k -> input 8*j+k
// zp:     I32 [ceil(out/8), in/64], nibble k at bits 4k -> output 8*i+k
// scale:  BF16 [out, in/64]; group_size = 64
// w[o,i] = bf16(float(q - z) * float(scale))
// ---------------------------------------------------------------------------
extern "C" __global__ void dequant_w4a16_bf16(const int* __restrict__ packed,
                                              const bf16* __restrict__ scale,
                                              const int* __restrict__ zp,
                                              bf16* __restrict__ out, int out_dim,
                                              int in_dim, int group_size) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)out_dim * in_dim;
  if (idx >= total) return;
  int i = (int)(idx % in_dim);
  int o = (int)(idx / in_dim);
  int words_per_row = (in_dim + 7) / 8;
  int groups = (in_dim + group_size - 1) / group_size;
  int word = packed[(long)o * words_per_row + (i >> 3)];
  int q = (word >> (4 * (i & 7))) & 0xF;
  int zword = zp[(long)(o >> 3) * groups + (i / group_size)];
  int z = (zword >> (4 * (o & 7))) & 0xF;
  float s = bf2f(scale[(long)o * groups + (i / group_size)]);
  out[idx] = f2bf((float)(q - z) * s);
}

// ---------------------------------------------------------------------------
// Fused W4A16 GEMM: C[M,N] = X[M,K] @ W[N,K]^T, no BF16 weight image in global
// memory. BM=BN=BK=64 (BK is one quantization group), 8 warps, bf16 tensor
// cores. Double-buffered shared memory with register prefetch: the packed next
// group and activation tile are in flight while the current tile multiplies.
//
// Staging maps one warp to 4 output rows: threads 0..7 cover the 8 packed words
// (32B) of row 0, threads 8..15 row 1, etc. Packed loads are coalesced and each
// thread stores one aligned uint4 (8 bf16) to shared.
// ---------------------------------------------------------------------------
#define FBM 64
#define FBN 64
#define FBK 64
#define FWARPS 8
// Padded shared row stride: avoids 128-byte strides, which put every row in
// the same bank set for column-wise tensor-core loads.
#define FLD (FBK + 8)

__device__ __forceinline__ uint4 pack8(const unsigned w, int z, float sc) {
  bf16 v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int q = (w >> (4 * j)) & 0xF;
    v[j] = f2bf((float)(q - z) * sc);
  }
  return make_uint4(
      (unsigned)__bfloat16_as_ushort(v[0]) |
          ((unsigned)__bfloat16_as_ushort(v[1]) << 16),
      (unsigned)__bfloat16_as_ushort(v[2]) |
          ((unsigned)__bfloat16_as_ushort(v[3]) << 16),
      (unsigned)__bfloat16_as_ushort(v[4]) |
          ((unsigned)__bfloat16_as_ushort(v[5]) << 16),
      (unsigned)__bfloat16_as_ushort(v[6]) |
          ((unsigned)__bfloat16_as_ushort(v[7]) << 16));
}

extern "C" __global__ void gemm_w4a16_bf16(
    const bf16* __restrict__ x, const int* __restrict__ packed,
    const bf16* __restrict__ scale, const int* __restrict__ zp,
    bf16* __restrict__ out, int M, int N, int K) {
  __shared__ __align__(32) bf16 sw[2][FBN][FLD];
  __shared__ __align__(32) bf16 sx[2][FBM][FLD];
  __shared__ float sacc[FWARPS][16 * 16];

  const int n0 = blockIdx.x * FBN;
  const int m0 = blockIdx.y * FBM;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int words_per_row = K >> 3;
  const int groups = K >> 6;

  // Weight staging: 8 threads per output row, one packed word each.
  const int wrow0 = tid >> 3;  // 0..31
  const int wcol = tid & 7;    // packed word within the 64-k group
  // Activation staging: 4 threads per row, 16 k each.
  const int xrow = tid >> 2;   // 0..63
  const int xkq = tid & 3;

  using namespace nvcuda;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
  wmma::fill_fragment(acc0, 0.f);
  wmma::fill_fragment(acc1, 0.f);

#pragma unroll
  for (int p = 0; p < 2; ++p) {
    int row = wrow0 + (p << 5);
    int gn = n0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gn < N) {
      unsigned w = (unsigned)packed[(long)gn * words_per_row + wcol];
      unsigned zw = (unsigned)zp[(long)(gn >> 3) * groups];
      int z = (zw >> (4 * (gn & 7))) & 0xF;
      float sc = bf2f(scale[(long)gn * groups]);
      v = pack8(w, z, sc);
    }
    *reinterpret_cast<uint4*>(&sw[0][row][wcol * 8]) = v;
  }
  if (m0 + xrow < M) {
    const bf16* src = x + (long)(m0 + xrow) * K + xkq * 16;
    uint4 v0 = *reinterpret_cast<const uint4*>(src);
    uint4 v1 = *reinterpret_cast<const uint4*>(src + 8);
    bf16* d = &sx[0][xrow][xkq * 16];
    *reinterpret_cast<uint4*>(d) = v0;
    *reinterpret_cast<uint4*>(d + 8) = v1;
  } else {
    bf16* d = &sx[0][xrow][xkq * 16];
    for (int jj = 0; jj < 16; ++jj) d[jj] = f2bf(0.f);
  }
  __syncthreads();

  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += FBK) {
    const int next = k0 + FBK;
    const bool has_next = next < K;

    // Prefetch the next quantization group (one packed word + one scale + one
    // zp word per row) and the next 16 activation values per thread.
    unsigned pw[2] = {0u, 0u};
    unsigned pz[2] = {0u, 0u};
    float psc[2] = {0.f, 0.f};
    uint4 px0 = make_uint4(0, 0, 0, 0);
    uint4 px1 = make_uint4(0, 0, 0, 0);
    if (has_next) {
      int wbase = (next >> 3) + wcol;
      int g = next >> 6;
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        int row = wrow0 + (p << 5);
        int gn = n0 + row;
        if (gn < N) {
          pw[p] = (unsigned)packed[(long)gn * words_per_row + wbase];
          pz[p] = (unsigned)zp[(long)(gn >> 3) * groups + g];
          psc[p] = bf2f(scale[(long)gn * groups + g]);
        }
      }
      if (m0 + xrow < M) {
        const bf16* src = x + (long)(m0 + xrow) * K + next + xkq * 16;
        px0 = *reinterpret_cast<const uint4*>(src);
        px1 = *reinterpret_cast<const uint4*>(src + 8);
      }
    }

    // MMA on the current group: four 16-wide K sub-blocks.
#pragma unroll
    for (int ti = 0; ti < 2; ++ti) {
      int t = ti == 0 ? warp : warp + FWARPS;
      int tr = t >> 2;
      int tc = t & 3;
#pragma unroll
      for (int ki = 0; ki < FBK; ki += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
        wmma::load_matrix_sync(a, &sx[buf][tr * 16][ki], FLD);
        wmma::load_matrix_sync(b, &sw[buf][tc * 16][ki], FLD);
        if (ti == 0)
          wmma::mma_sync(acc0, a, b, acc0);
        else
          wmma::mma_sync(acc1, a, b, acc1);
      }
    }

    if (has_next) {
      const int nb = buf ^ 1;
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        int row = wrow0 + (p << 5);
        int gn = n0 + row;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (gn < N) {
          int z = (pz[p] >> (4 * (gn & 7))) & 0xF;
          v = pack8(pw[p], z, psc[p]);
        }
        *reinterpret_cast<uint4*>(&sw[nb][row][wcol * 8]) = v;
      }
      bf16* dx = &sx[nb][xrow][xkq * 16];
      *reinterpret_cast<uint4*>(dx) = px0;
      *reinterpret_cast<uint4*>(dx + 8) = px1;
    }
    __syncthreads();
    buf ^= 1;
  }

  // Epilogue: 16 warp tiles of 16x16.
#pragma unroll
  for (int ti = 0; ti < 2; ++ti) {
    int t = ti == 0 ? warp : warp + FWARPS;
    int tr = t >> 2;
    int tc = t & 3;
    if (ti == 0)
      wmma::store_matrix_sync(&sacc[warp][0], acc0, 16, wmma::mem_row_major);
    else
      wmma::store_matrix_sync(&sacc[warp][0], acc1, 16, wmma::mem_row_major);
    __syncwarp();
    for (int idx = lane; idx < 256; idx += 32) {
      int r = idx >> 4;
      int cc = idx & 15;
      int ogm = m0 + tr * 16 + r;
      int ogn = n0 + tc * 16 + cc;
      if (ogm < M && ogn < N) out[(long)ogm * N + ogn] = f2bf(sacc[warp][idx]);
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// Large-tile fused W4A16 GEMM for 65 <= M <= 128 (one 128-row M tile) and as an
// experiment for 1K prefills. BM=BN=128, BK=64, 16 warps, dynamic shared:
//   sw[2] 128x72 bf16 (36 KiB), sx[2] 128x72 bf16 (36 KiB), sacc 16 KiB.
// Same coalesced packed staging as the 64x64 kernel.
// ---------------------------------------------------------------------------
#define M128_BM 128
#define M128_BN 128
#define M128_BK 64
#define M128_WARPS 16
#define M128_FLD 72
#define M128_SMEM_BYTES (4 * M128_BM * M128_FLD * 2 + M128_WARPS * 16 * 16 * 4)

extern "C" __global__ void gemm_w4a16_m128_bf16(
    const bf16* __restrict__ x, const int* __restrict__ packed,
    const bf16* __restrict__ scale, const int* __restrict__ zp,
    bf16* __restrict__ out, int M, int N, int K) {
  extern __shared__ __align__(32) unsigned char smem_raw[];
  bf16* sbase = reinterpret_cast<bf16*>(smem_raw);
  bf16* sw[2] = {sbase, sbase + M128_BM * M128_FLD};
  bf16* sx[2] = {sbase + 2 * M128_BM * M128_FLD,
                 sbase + 3 * M128_BM * M128_FLD};
  float* sacc = reinterpret_cast<float*>(sbase + 4 * M128_BM * M128_FLD);

  const int n0 = blockIdx.x * M128_BN;
  const int m0 = blockIdx.y * M128_BM;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int words_per_row = K >> 3;
  const int groups = K >> 6;
  const int wrow0 = tid >> 3;  // 0..63, two passes -> 128 rows
  const int wcol = tid & 7;
  const int xrow = tid >> 2;  // 0..127
  const int xkq = tid & 3;

  using namespace nvcuda;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) wmma::fill_fragment(acc[i], 0.f);

#pragma unroll
  for (int p = 0; p < 2; ++p) {
    int row = wrow0 + (p << 6);
    int gn = n0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gn < N) {
      unsigned w = (unsigned)packed[(long)gn * words_per_row + wcol];
      unsigned zw = (unsigned)zp[(long)(gn >> 3) * groups];
      int z = (zw >> (4 * (gn & 7))) & 0xF;
      float sc = bf2f(scale[(long)gn * groups]);
      v = pack8(w, z, sc);
    }
    *reinterpret_cast<uint4*>(sw[0] + row * M128_FLD + wcol * 8) = v;
  }
  if (m0 + xrow < M) {
    const bf16* src = x + (long)(m0 + xrow) * K + xkq * 16;
    uint4 v0 = *reinterpret_cast<const uint4*>(src);
    uint4 v1 = *reinterpret_cast<const uint4*>(src + 8);
    bf16* d = sx[0] + xrow * M128_FLD + xkq * 16;
    *reinterpret_cast<uint4*>(d) = v0;
    *reinterpret_cast<uint4*>(d + 8) = v1;
  } else {
    bf16* d = sx[0] + xrow * M128_FLD + xkq * 16;
    for (int jj = 0; jj < 16; ++jj) d[jj] = f2bf(0.f);
  }
  __syncthreads();

  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += M128_BK) {
    const int next = k0 + M128_BK;
    const bool has_next = next < K;

    unsigned pw[2] = {0u, 0u};
    unsigned pz[2] = {0u, 0u};
    float psc[2] = {0.f, 0.f};
    uint4 px0 = make_uint4(0, 0, 0, 0);
    uint4 px1 = make_uint4(0, 0, 0, 0);
    if (has_next) {
      int wbase = (next >> 3) + wcol;
      int g = next >> 6;
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        int row = wrow0 + (p << 6);
        int gn = n0 + row;
        if (gn < N) {
          pw[p] = (unsigned)packed[(long)gn * words_per_row + wbase];
          pz[p] = (unsigned)zp[(long)(gn >> 3) * groups + g];
          psc[p] = bf2f(scale[(long)gn * groups + g]);
        }
      }
      if (m0 + xrow < M) {
        const bf16* src = x + (long)(m0 + xrow) * K + next + xkq * 16;
        px0 = *reinterpret_cast<const uint4*>(src);
        px1 = *reinterpret_cast<const uint4*>(src + 8);
      }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int t = warp * 4 + i;
      int tr = t >> 3;
      int tc = t & 7;
#pragma unroll
      for (int ki = 0; ki < M128_BK; ki += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
        wmma::load_matrix_sync(a, sx[buf] + (tr * 16) * M128_FLD + ki, M128_FLD);
        wmma::load_matrix_sync(b, sw[buf] + (tc * 16) * M128_FLD + ki, M128_FLD);
        wmma::mma_sync(acc[i], a, b, acc[i]);
      }
    }

    if (has_next) {
      int nb = buf ^ 1;
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        int row = wrow0 + (p << 6);
        int gn = n0 + row;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (gn < N) {
          int z = (pz[p] >> (4 * (gn & 7))) & 0xF;
          v = pack8(pw[p], z, psc[p]);
        }
        *reinterpret_cast<uint4*>(sw[nb] + row * M128_FLD + wcol * 8) = v;
      }
      bf16* d = sx[nb] + xrow * M128_FLD + xkq * 16;
      *reinterpret_cast<uint4*>(d) = px0;
      *reinterpret_cast<uint4*>(d + 8) = px1;
    }
    __syncthreads();
    buf ^= 1;
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int t = warp * 4 + i;
    int tr = t >> 3;
    int tc = t & 7;
    float* sa = sacc + warp * 256;
    wmma::store_matrix_sync(sa, acc[i], 16, wmma::mem_row_major);
    __syncwarp();
    for (int idx = lane; idx < 256; idx += 32) {
      int r = idx >> 4;
      int cc = idx & 15;
      int ogm = m0 + tr * 16 + r;
      int ogn = n0 + tc * 16 + cc;
      if (ogm < M && ogn < N) out[(long)ogm * N + ogn] = f2bf(sa[idx]);
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// Embedding gather (device-side variant, not used by the CPU-gather path).
// ---------------------------------------------------------------------------
extern "C" __global__ void embed_gather_bf16(const bf16* __restrict__ table,
                                             const int* __restrict__ tokens,
                                             bf16* __restrict__ out, int hidden) {
  int t = blockIdx.x;
  int tok = tokens[t];
  for (int j = threadIdx.x; j < hidden; j += blockDim.x)
    out[(size_t)t * hidden + j] = table[(size_t)tok * hidden + j];
}


// ===========================================================================
// Tiled causal GQA attention (FlashAttention-2 style) for sm_89, BF16.
// Added by the standalone /tmp/jev-attn work; see attn.cu for the host wrapper.
// Grid: (ceil(seq/64), n_heads), block: 256, dynamic smem: 101376 B at hd=256.
// ===========================================================================

#include <cstdint>

namespace {

constexpr int BLOCK_M = 64;   // queries per CTA
constexpr int BLOCK_N = 32;   // keys per K/V tile
constexpr int THREADS = 256;  // 8 warps
constexpr int PAD = 8;        // bf16 elements of row padding (16 B)

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst),
               "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}

// Row-major 8x16 K block -> B fragment of m16n8k16 (K is [n=key][k=dim]).
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(r[0]), "=r"(r[1])
      : "r"(addr));
}

// Two 8x8 V blocks, transposed -> B fragment for P*V.
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2],
                                                  uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(r[0]), "=r"(r[1])
      : "r"(addr));
}

// D += A * B, fp32 accumulate, bf16 inputs.
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t pack2(float x, float y) {
  __nv_bfloat162 p = __floats2bfloat162_rn(x, y);
  uint32_t r;
  __builtin_memcpy(&r, &p, sizeof(r));
  return r;
}

__device__ __forceinline__ float warp_max4(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
  return x;
}

__device__ __forceinline__ float warp_sum4(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  x += __shfl_xor_sync(0xffffffffu, x, 2);
  return x;
}

// D is head_dim. Rows in shared memory are padded by PAD bf16 elements.
template <int D>
__device__ __forceinline__ void attention_tc_body(
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ v, bf16* __restrict__ out, int seq, int n_heads,
    int n_kv, float scale) {
  constexpr int QS = D + PAD;
  constexpr int HALF = D / 2;    // output dims handled per warp column
  constexpr int NT8 = HALF / 8;  // n8 mma tiles per warp column
  constexpr int NK = D / 8;      // 16-byte chunks per row

  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* Qs = reinterpret_cast<bf16*>(smem_raw);
  bf16* Ks = Qs + BLOCK_M * QS;
  bf16* Vs = Ks + 2 * BLOCK_N * QS;

  const int q_start = blockIdx.x * BLOCK_M;
  const int h = blockIdx.y;
  const int kvh = h / (n_heads / n_kv);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp & 3;   // query row group
  const int wn = warp >> 2;  // output dim group
  const int row0 = wm * 16;
  const int dim0 = wn * HALF;

  const int nkeys = min(q_start + BLOCK_M, seq);
  const int ntiles = (nkeys + BLOCK_N - 1) / BLOCK_N;

  // ---- prologue: Q + first K/V tile -------------------------------------
  for (int idx = tid; idx < BLOCK_M * NK; idx += THREADS) {
    const int r = idx / NK;
    const int c = idx - r * NK;
    const int gr = min(q_start + r, seq - 1);
    cp_async16(smem_addr(&Qs[r * QS + c * 8]),
               q + (static_cast<size_t>(gr) * n_heads + h) * D + c * 8);
  }
  for (int idx = tid; idx < BLOCK_N * NK; idx += THREADS) {
    const int r = idx / NK;
    const int c = idx - r * NK;
    const int gk = min(r, seq - 1);
    const bf16* src_k = k + (static_cast<size_t>(gk) * n_kv + kvh) * D + c * 8;
    const bf16* src_v = v + (static_cast<size_t>(gk) * n_kv + kvh) * D + c * 8;
    cp_async16(smem_addr(&Ks[r * QS + c * 8]), src_k);
    cp_async16(smem_addr(&Vs[r * QS + c * 8]), src_v);
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();

  // ---- running state (per thread: row r = lane/4, row r+8) --------------
  float m0 = -INFINITY, m1 = -INFINITY;
  float l0 = 0.f, l1 = 0.f;
  float o[NT8][4];
#pragma unroll
  for (int t = 0; t < NT8; ++t) {
    o[t][0] = 0.f;
    o[t][1] = 0.f;
    o[t][2] = 0.f;
    o[t][3] = 0.f;
  }

  const float scale_l2 = scale * 1.4426950408889634f;  // log2(e)
  const int qpos0 = q_start + row0 + (lane >> 2);
  const int qpos1 = qpos0 + 8;
  const int lane_c = 2 * (lane & 3);

  for (int j = 0; j < ntiles; ++j) {
    const int cb = j & 1;

    // Prefetch the next K/V tile into the other buffer while computing.
    if (j + 1 < ntiles) {
      const int nk0 = (j + 1) * BLOCK_N;
      for (int idx = tid; idx < BLOCK_N * NK; idx += THREADS) {
        const int r = idx / NK;
        const int c = idx - r * NK;
        const int gk = min(nk0 + r, seq - 1);
        const bf16* src_k =
            k + (static_cast<size_t>(gk) * n_kv + kvh) * D + c * 8;
        const bf16* src_v =
            v + (static_cast<size_t>(gk) * n_kv + kvh) * D + c * 8;
        cp_async16(smem_addr(&Ks[((cb ^ 1) * BLOCK_N + r) * QS + c * 8]), src_k);
        cp_async16(smem_addr(&Vs[((cb ^ 1) * BLOCK_N + r) * QS + c * 8]), src_v);
      }
    }
    cp_async_commit();

    // ---- S = Q * K^T ----------------------------------------------------
    float s[4][4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      s[t][0] = 0.f;
      s[t][1] = 0.f;
      s[t][2] = 0.f;
      s[t][3] = 0.f;
    }

    const int qa_row = row0 + (lane & 15);
    const int qa_col_hi = (lane >> 4) * 8;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      uint32_t a[4];
      ldmatrix_x4(a, smem_addr(&Qs[qa_row * QS + kk * 16 + qa_col_hi]));
      const int kb_row = (lane & 7);
      const int kb_col_hi = ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        uint32_t b[2];
        ldmatrix_x2(
            b, smem_addr(&Ks[(cb * BLOCK_N + t * 8 + kb_row) * QS + kk * 16 +
                             kb_col_hi]));
        mma16816(s[t], a, b);
      }
    }

    // ---- causal mask + online softmax -----------------------------------
    const bool need_mask = ((j + 1) * BLOCK_N - 1) > q_start;
    if (need_mask) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int col = j * BLOCK_N + t * 8 + lane_c;
        s[t][0] = (col > qpos0) ? -INFINITY : s[t][0] * scale_l2;
        s[t][1] = (col + 1 > qpos0) ? -INFINITY : s[t][1] * scale_l2;
        s[t][2] = (col > qpos1) ? -INFINITY : s[t][2] * scale_l2;
        s[t][3] = (col + 1 > qpos1) ? -INFINITY : s[t][3] * scale_l2;
      }
    } else {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        s[t][0] *= scale_l2;
        s[t][1] *= scale_l2;
        s[t][2] *= scale_l2;
        s[t][3] *= scale_l2;
      }
    }

    float mt0 = s[0][0], mt1 = s[0][2];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      mt0 = fmaxf(mt0, fmaxf(s[t][0], s[t][1]));
      mt1 = fmaxf(mt1, fmaxf(s[t][2], s[t][3]));
    }
    mt0 = warp_max4(mt0);
    mt1 = warp_max4(mt1);

    const float mn0 = fmaxf(m0, mt0);
    const float mn1 = fmaxf(m1, mt1);
    const float a0 = (mn0 == -INFINITY) ? 0.f : exp2f(m0 - mn0);
    const float a1 = (mn1 == -INFINITY) ? 0.f : exp2f(m1 - mn1);

    float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      s[t][0] = (mn0 == -INFINITY) ? 0.f : exp2f(s[t][0] - mn0);
      s[t][1] = (mn0 == -INFINITY) ? 0.f : exp2f(s[t][1] - mn0);
      s[t][2] = (mn1 == -INFINITY) ? 0.f : exp2f(s[t][2] - mn1);
      s[t][3] = (mn1 == -INFINITY) ? 0.f : exp2f(s[t][3] - mn1);
      sum0 += s[t][0] + s[t][1];
      sum1 += s[t][2] + s[t][3];
    }
    sum0 = warp_sum4(sum0);
    sum1 = warp_sum4(sum1);
    l0 = l0 * a0 + sum0;
    l1 = l1 * a1 + sum1;
    m0 = mn0;
    m1 = mn1;

    // ---- O = O * alpha + P * V ------------------------------------------
#pragma unroll
    for (int t = 0; t < NT8; ++t) {
      o[t][0] *= a0;
      o[t][1] *= a0;
      o[t][2] *= a1;
      o[t][3] *= a1;
    }

    uint32_t pa[2][4];
#pragma unroll
    for (int f = 0; f < 2; ++f) {
      const int t0 = 2 * f, t1 = 2 * f + 1;
      pa[f][0] = pack2(s[t0][0], s[t0][1]);
      pa[f][1] = pack2(s[t0][2], s[t0][3]);
      pa[f][2] = pack2(s[t1][0], s[t1][1]);
      pa[f][3] = pack2(s[t1][2], s[t1][3]);
    }

    const int vb_row = (lane & 7) + (((lane >> 3) & 1) * 8);
#pragma unroll
    for (int f = 0; f < 2; ++f) {
      const int vbase = cb * BLOCK_N + f * 16;
#pragma unroll
      for (int t = 0; t < NT8; ++t) {
        uint32_t b[2];
        ldmatrix_x2_trans(
            b, smem_addr(&Vs[(vbase + vb_row) * QS + dim0 + t * 8]));
        mma16816(o[t], pa[f], b);
      }
    }

    if (j + 1 < ntiles) {
      cp_async_wait_all();
    }
    __syncthreads();
  }

  // ---- epilogue ---------------------------------------------------------
  const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f;
  const float inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
  const int qrow0 = q_start + row0 + (lane >> 2);
  const int qrow1 = qrow0 + 8;
  const size_t obase0 =
      (static_cast<size_t>(qrow0) * n_heads + h) * D + dim0 + lane_c;
  const size_t obase1 =
      (static_cast<size_t>(qrow1) * n_heads + h) * D + dim0 + lane_c;
  const bool w0 = qrow0 < seq;
  const bool w1 = qrow1 < seq;
#pragma unroll
  for (int t = 0; t < NT8; ++t) {
    if (w0) {
      __nv_bfloat162 p =
          __floats2bfloat162_rn(o[t][0] * inv0, o[t][1] * inv0);
      __builtin_memcpy(out + obase0 + t * 8, &p, sizeof(p));
    }
    if (w1) {
      __nv_bfloat162 p =
          __floats2bfloat162_rn(o[t][2] * inv1, o[t][3] * inv1);
      __builtin_memcpy(out + obase1 + t * 8, &p, sizeof(p));
    }
  }
}

}  // namespace

// Fixed-symbol entry points (one per supported head_dim). These are the
// functions the engine's cubin loader looks up; the templated body above is
// inlined into each.
#define ATTN_TC_WRAPPER(D)                                                    \
  extern "C" __global__ void __launch_bounds__(256, 1)                        \
      attention_tc_bf16_d##D(const __nv_bfloat16* __restrict__ q,             \
                             const __nv_bfloat16* __restrict__ k,             \
                             const __nv_bfloat16* __restrict__ v,             \
                             __nv_bfloat16* __restrict__ out, int seq,        \
                             int n_heads, int n_kv, float scale) {            \
    attention_tc_body<D>(q, k, v, out, seq, n_heads, n_kv, scale);            \
  }

ATTN_TC_WRAPPER(64)
ATTN_TC_WRAPPER(128)
ATTN_TC_WRAPPER(256)

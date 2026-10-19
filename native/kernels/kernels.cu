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
// Fused W4A16 GEMM: C[M,N] = X[M,K] @ W[N,K]^T with W dequantized in shared
// memory. BM=BN=64, BK=64 (one quantization group), 8 warps, bf16 tensor cores.
// No intermediate BF16 weight ever reaches global memory.
// ---------------------------------------------------------------------------
#define FBM 64
#define FBN 64
#define FBK 64
#define FWARPS 8

extern "C" __global__ void gemm_w4a16_bf16(
    const bf16* __restrict__ x, const int* __restrict__ packed,
    const bf16* __restrict__ scale, const int* __restrict__ zp,
    bf16* __restrict__ out, int M, int N, int K) {
  __shared__ bf16 sx[FBM][FBK];
  __shared__ bf16 sw[FBN][FBK];
  __shared__ float sacc[FWARPS][16 * 16];

  const int n0 = blockIdx.x * FBN;
  const int m0 = blockIdx.y * FBM;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int words_per_row = (K + 7) / 8;
  const int groups = K / 64;

  using namespace nvcuda;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
  wmma::fill_fragment(acc0, 0.f);
  wmma::fill_fragment(acc1, 0.f);

  for (int k0 = 0; k0 < K; k0 += FBK) {
    // Dequantize the 64x64 weight tile into shared memory.
    for (int idx = tid; idx < FBN * FBK; idx += blockDim.x) {
      int n = idx >> 6;
      int kk = idx & 63;
      int gn = n0 + n;
      bf16 w = f2bf(0.f);
      if (gn < N) {
        int k = k0 + kk;
        int g = k0 >> 6;
        int word = packed[(long)gn * words_per_row + (k >> 3)];
        int q = (word >> (4 * (k & 7))) & 0xF;
        int zword = zp[(long)(gn >> 3) * groups + g];
        int z = (zword >> (4 * (gn & 7))) & 0xF;
        float sc = bf2f(scale[(long)gn * groups + g]);
        w = f2bf((float)(q - z) * sc);
      }
      sw[n][kk] = w;
    }
    // Stage the X tile; rows past M are zero.
    for (int idx = tid; idx < FBM * FBK; idx += blockDim.x) {
      int m = idx >> 6;
      int kk = idx & 63;
      int gm = m0 + m;
      sx[m][kk] = (gm < M) ? x[(long)gm * K + k0 + kk] : f2bf(0.f);
    }
    __syncthreads();

    int t0 = warp;
    int t1 = warp + FWARPS;
#pragma unroll
    for (int ti = 0; ti < 2; ++ti) {
      int t = ti == 0 ? t0 : t1;
      int tr = t >> 2;
      int tc = t & 3;
      // The staged K extent is 64 (one quantization group), but each MMA
      // consumes 16 columns; accumulate all four sub-blocks.
#pragma unroll
      for (int ki = 0; ki < FBK; ki += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
        wmma::load_matrix_sync(a, &sx[tr * 16][ki], FBK);
        wmma::load_matrix_sync(b, &sw[tc * 16][ki], FBK);
        if (ti == 0)
          wmma::mma_sync(acc0, a, b, acc0);
        else
          wmma::mma_sync(acc1, a, b, acc1);
      }
    }
    __syncthreads();
  }

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
      int gm = m0 + tr * 16 + r;
      int gn = n0 + tc * 16 + cc;
      if (gm < M && gn < N) out[(long)gm * N + gn] = f2bf(sacc[warp][idx]);
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

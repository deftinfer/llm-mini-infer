// Gemma 4's fused elementwise and norm kernels in BF16. They round where the torch references in kernels/gemma_ops.py
// do (fp32 inside each fused op, one rounding at the store). Only the order of the sum-of-squares
// reductions differs from torch.
//
// Quantizing producers (block-FP8 checkpoints): the RMS norm, the sandwich, and the GELU-tanh x up. Each copies the
// arithmetic of its BF16 kernel: same threads, reduction order and rounding. It then quantizes the BF16 result per
// token and group of 128 (quant.cuh). So each equals its BF16 kernel followed by the standalone quantization, bit
// for bit, as Qwen's producers in quant.cu do. All three round to BF16 before they quantize, unlike the SiLU-and-mul
// in quant.cu, which quantizes the fp32 product.
//
// Programmatic dependent launch (PDL, common.cuh): each kernel waits for its predecessor first, then lets the next
// kernel launch. The launchers add the attribute while the library's flag is on.
#include "common.cuh"
#include "kv_fp8.cuh"
#include "quant.cuh"

namespace {

using lmk_quant::fp8_byte;
using lmk_quant::group_inv;
using lmk_quant::group_scale;
using lmk_quant::kGroup;
using lmk_quant::quant_store4;

// Gemma 4 RMSNorm `x / rms(x) * w` (the weight as is): out = bf16((x * rsqrt(sum(x^2) * (1/D) + eps)) * w).
// One block per row. x has row stride `xs`; out is [rows, D].
__global__ void g4_rmsnorm_kernel(const bf16* __restrict__ x, int64_t xs, const bf16* __restrict__ w,
                                  bf16* __restrict__ out, int D, float eps) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  extern __shared__ float row[];
  __shared__ float red[32];
  const int64_t r = blockIdx.x;
  const bf16* xr = x + r * xs;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = bf2f(xr[i]);
    row[i] = v;
    ss = fadd(ss, fmul(v, v));
  }
  ss = block_sum(ss, red);
  const float inv = rsqrtf(fadd(fmul(ss, 1.0f / (float)D), eps));
  for (int i = threadIdx.x; i < D; i += blockDim.x) out[r * D + i] = f2bf(fmul(fmul(row[i], inv), bf2f(w[i])));
}

// End of a sandwich block plus the next norm (gemma_ops.sandwich). All fp32: s = residual + (a / rms(a)) * w_post
// (times `scalar` when given). r_out = bf16(s) is the new residual. x_out = bf16((r / rms(r)) * w_next) reads that
// rounded residual. One block per row; a has row stride `as`; residual and the outputs are [rows, D].
__global__ void g4_sandwich_kernel(const bf16* __restrict__ a, int64_t as, const bf16* __restrict__ residual,
                                   const bf16* __restrict__ w_post, const bf16* __restrict__ w_next,
                                   const bf16* __restrict__ scalar, bf16* __restrict__ r_out,
                                   bf16* __restrict__ x_out, int D, float eps) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  extern __shared__ float row[];
  __shared__ float red[32];
  const int64_t r = blockIdx.x;
  const bf16* ar = a + r * as;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = bf2f(ar[i]);
    row[i] = v;
    ss = fadd(ss, fmul(v, v));
  }
  ss = block_sum(ss, red);
  const float inv = rsqrtf(fadd(fmul(ss, 1.0f / (float)D), eps));
  const float sc = scalar != nullptr ? bf2f(scalar[0]) : 1.0f;
  float ss2 = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float s = fadd(bf2f(residual[r * D + i]), fmul(fmul(row[i], inv), bf2f(w_post[i])));
    if (scalar != nullptr) s = fmul(s, sc);
    const bf16 rb = f2bf(s);
    r_out[r * D + i] = rb;
    const float rv = bf2f(rb);
    row[i] = rv;
    ss2 = fadd(ss2, fmul(rv, rv));
  }
  ss2 = block_sum(ss2, red);
  const float inv2 = rsqrtf(fadd(fmul(ss2, 1.0f / (float)D), eps));
  for (int i = threadIdx.x; i < D; i += blockDim.x) x_out[r * D + i] = f2bf(fmul(fmul(row[i], inv2), bf2f(w_next[i])));
}

// load8 reads 8 BF16 values at p (16-byte aligned) as floats. store8 converts back (round to nearest even, as f2bf).
__device__ __forceinline__ void load8(const bf16* p, float* f) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}
__device__ __forceinline__ void store8(bf16* p, const float* f) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}

// The sandwich of `g4_sandwich_kernel` with 16-byte loads and the row in registers. Each thread holds NV vectors of 8
// values (D % 8 == 0, D / 8 <= NV * blockDim.x). A decode step has few rows, one block each, so the latency of one
// row is the latency of the kernel. The scalar kernel takes 6.9 µs per call at batch 1, and a step makes 120 calls.
template <int NV>
__global__ void g4_sandwich_vec_kernel(const bf16* __restrict__ a, int64_t as, const bf16* __restrict__ residual,
                                       const bf16* __restrict__ w_post, const bf16* __restrict__ w_next,
                                       const bf16* __restrict__ scalar, bf16* __restrict__ r_out,
                                       bf16* __restrict__ x_out, int D, float eps) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  __shared__ float red[32];
  const int64_t r = blockIdx.x;
  const int nvec = D >> 3;
  float v[NV][8];
  float ss = 0.0f;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    const int j = threadIdx.x + k * blockDim.x;
    if (j < nvec) {
      load8(a + r * as + j * 8, v[k]);
#pragma unroll
      for (int e = 0; e < 8; ++e) ss = fadd(ss, fmul(v[k][e], v[k][e]));
    }
  }
  ss = block_sum(ss, red);
  const float inv = rsqrtf(fadd(fmul(ss, 1.0f / (float)D), eps));
  const float sc = scalar != nullptr ? bf2f(scalar[0]) : 1.0f;
  float ss2 = 0.0f;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    const int j = threadIdx.x + k * blockDim.x;
    if (j < nvec) {
      float res[8], wp[8];
      load8(residual + r * D + j * 8, res);
      load8(w_post + j * 8, wp);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float t = fadd(res[e], fmul(fmul(v[k][e], inv), wp[e]));
        if (scalar != nullptr) t = fmul(t, sc);
        v[k][e] = bf2f(f2bf(t));  // the stored residual, which the next norm reads
        ss2 = fadd(ss2, fmul(v[k][e], v[k][e]));
      }
      store8(r_out + r * D + j * 8, v[k]);
    }
  }
  ss2 = block_sum(ss2, red);
  const float inv2 = rsqrtf(fadd(fmul(ss2, 1.0f / (float)D), eps));
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    const int j = threadIdx.x + k * blockDim.x;
    if (j < nvec) {
      float wn[8], o[8];
      load8(w_next + j * 8, wn);
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = fmul(fmul(v[k][e], inv2), wn[e]);
      store8(x_out + r * D + j * 8, o);
    }
  }
}

// torch's tanh-approximate GELU (aten/src/ATen/native/cuda/ActivationGeluKernel.cu), in fp32.
__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float kBeta = 0.7978845608028654f;  // sqrt(2 / pi)
  constexpr float kKappa = 0.044715f;
  const float x_cube = x * x * x;
  const float inner = kBeta * (x + kKappa * x_cube);
  return 0.5f * x * (1.0f + tanhf(inner));
}

// MLP activation. gu is [T, 2I] (gate, then up) at row stride `gs`; out [T, I] = bf16(gelu_tanh(g) * u).
__global__ void gelu_tanh_mul_kernel(const bf16* __restrict__ gu, int64_t gs, bf16* __restrict__ out, int64_t T,
                                     int I) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * I) return;
  const int64_t t = idx / I;
  const int i = (int)(idx % I);
  out[idx] = f2bf(fmul(gelu_tanh(bf2f(gu[t * gs + i])), bf2f(gu[t * gs + I + i])));
}

// `g4_rmsnorm_kernel`, then quantization: same threads and reduction; each output is rounded to BF16, then quantized
// (one warp per group of 128). q is [rows, D] e4m3; s is [D / 128, ld], MN-major: the scale of row r and group g is at
// s[g * ld + r].
__global__ void g4_rmsnorm_quant_kernel(const bf16* __restrict__ x, int64_t xs, const bf16* __restrict__ w,
                                        uint8_t* __restrict__ q, float* __restrict__ s, int64_t ld, int D, float eps) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  extern __shared__ float row[];
  __shared__ float red[32];
  const int64_t r = blockIdx.x;
  const bf16* xr = x + r * xs;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = bf2f(xr[i]);
    row[i] = v;
    ss = fadd(ss, fmul(v, v));
  }
  ss = block_sum(ss, red);  // its barriers also make row[] visible to all warps
  const float inv = rsqrtf(fadd(fmul(ss, 1.0f / (float)D), eps));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int g = warp; g < D / kGroup; g += nw) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int i = g * kGroup + 4 * lane + j;
      v[j] = bf2f(f2bf(fmul(fmul(row[i], inv), bf2f(w[i]))));
    }
    quant_store4(v, q + r * D + g * kGroup, s + (int64_t)g * ld + r, lane);
  }
}

// `g4_sandwich_vec_kernel<NV>`, but the next norm's output is quantized instead of stored. Thread t holds vectors
// t + k * blockDim, 8 values each. blockDim is a multiple of 32 and D of 128, so a group of 128 is 16 consecutive
// vectors in the 16 lanes of a half-warp. The half-warp reduces the group's absmax. r_out, the new BF16 residual, is
// written as in the BF16 kernel.
template <int NV>
__global__ void g4_sandwich_quant_kernel(const bf16* __restrict__ a, int64_t as, const bf16* __restrict__ residual,
                                         const bf16* __restrict__ w_post, const bf16* __restrict__ w_next,
                                         const bf16* __restrict__ scalar, bf16* __restrict__ r_out,
                                         uint8_t* __restrict__ q, float* __restrict__ s, int64_t ld, int D,
                                         float eps) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  __shared__ float red[32];
  const int64_t r = blockIdx.x;
  const int nvec = D >> 3;
  float v[NV][8];
  float ss = 0.0f;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    const int j = threadIdx.x + k * blockDim.x;
    if (j < nvec) {
      load8(a + r * as + j * 8, v[k]);
#pragma unroll
      for (int e = 0; e < 8; ++e) ss = fadd(ss, fmul(v[k][e], v[k][e]));
    }
  }
  ss = block_sum(ss, red);
  const float inv = rsqrtf(fadd(fmul(ss, 1.0f / (float)D), eps));
  const float sc = scalar != nullptr ? bf2f(scalar[0]) : 1.0f;
  float ss2 = 0.0f;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    const int j = threadIdx.x + k * blockDim.x;
    if (j < nvec) {
      float res[8], wp[8];
      load8(residual + r * D + j * 8, res);
      load8(w_post + j * 8, wp);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float t = fadd(res[e], fmul(fmul(v[k][e], inv), wp[e]));
        if (scalar != nullptr) t = fmul(t, sc);
        v[k][e] = bf2f(f2bf(t));  // the stored residual, which the next norm reads
        ss2 = fadd(ss2, fmul(v[k][e], v[k][e]));
      }
      store8(r_out + r * D + j * 8, v[k]);
    }
  }
  ss2 = block_sum(ss2, red);
  const float inv2 = rsqrtf(fadd(fmul(ss2, 1.0f / (float)D), eps));
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    const int j = threadIdx.x + k * blockDim.x;
    const bool live = j < nvec;  // a half-warp is wholly live or idle (D % 128 == 0); all lanes shuffle
    float o[8];
    float am = 0.0f;
    if (live) {
      float wn[8];
      load8(w_next + j * 8, wn);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        o[e] = bf2f(f2bf(fmul(fmul(v[k][e], inv2), wn[e])));
        am = fmaxf(am, fabsf(o[e]));
      }
    }
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
    if (live) {
      const float gs = group_scale(am);
      const float qinv = group_inv(gs);
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        lo |= fp8_byte(o[e], qinv) << (8 * e);
        hi |= fp8_byte(o[4 + e], qinv) << (8 * e);
      }
      *reinterpret_cast<uint2*>(q + r * D + j * 8) = make_uint2(lo, hi);
      if ((lane & 15) == 0) s[(int64_t)(j >> 4) * ld + r] = gs;
    }
  }
}

// `gelu_tanh_mul_kernel`, then quantization: one warp per (token, group of 128); lane l holds values 4l .. 4l + 3.
// gu is [T, 2I] (gate, then up) at row stride `gs`; q is [T, I]; s is [I / 128, ld].
__global__ void gelu_tanh_mul_quant_kernel(const bf16* __restrict__ gu, int64_t gs, uint8_t* __restrict__ q,
                                           float* __restrict__ s, int64_t T, int I, int64_t ld) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  const int G = I / kGroup;
  const int64_t item = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (item >= T * G) return;
  const int lane = threadIdx.x & 31;
  const int64_t t = item / G;
  const int g = (int)(item % G);
  float gv[4], uv[4], v[4];
  lmk_quant::load_bf16x4(gu + t * gs + g * kGroup + 4 * lane, gv);
  lmk_quant::load_bf16x4(gu + t * gs + I + g * kGroup + 4 * lane, uv);
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = bf2f(f2bf(fmul(gelu_tanh(gv[j]), uv[j])));
  quant_store4(v, q + t * I + g * kGroup, s + (int64_t)g * ld + t, lane);
}

constexpr int kMaxHeadDim = 512;

// Input side of the attention layer (gemma_ops.qkv_store): one warp per (token, head), over the H q heads, then the
// Hkv k heads.
//   q: q_out = bf16(rope(x / rms(x) * q_w)).
//   k: K = bf16(rope(x / rms(x) * k_w)) into the K plane, and V = bf16(v / rms(v)) into the V plane.
//   `k_eq_v`: V = bf16(k / rms(k)) into the single plane; there is no K.
// RoPE is NeoX over the whole head, with pairs (i, i + D / 2). It runs in fp32 against the BF16 cos/sin cache
// [max_pos, D / 2]. Lane l holds elements l + 32 j, so each pair is in one lane. qkv is [T, *] at row stride `qs`.
// The q heads start at 0, the k heads at H * D, and the v heads at (H + Hkv) * D. The KV layer view has page stride
// `ps_kv` and shape [page_size, Hkv, D] (k_eq_v) or [2, page_size, Hkv, D].
// `kTrigger`: FlashInfer's sliding attention follows this kernel and launches with PDL by default. So this kernel
// lets the next launch start only while the library's flag is on (common.cuh).
// `kFp8` (k_eq_v only): the plane holds E4M3 bytes, each the BF16 value stored otherwise rounded (kv_fp8.cuh); the KV
// layer view's offsets count bytes.
template <bool kTrigger, bool kFp8>
__global__ void g4_qkv_kernel(const bf16* __restrict__ qkv, int64_t qs, int k_eq_v, const bf16* __restrict__ qw,
                              const bf16* __restrict__ kw, const int64_t* __restrict__ pos,
                              const bf16* __restrict__ cos, const bf16* __restrict__ sin, bf16* __restrict__ q_out,
                              bf16* __restrict__ kv, int64_t ps_kv, int page_size,
                              const int64_t* __restrict__ kv_page, const int64_t* __restrict__ kv_off, int64_t T,
                              int H, int Hkv, int D, float eps) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  if (kTrigger) lmk_pdl_trigger();
  const int lane = threadIdx.x & 31;
  const int64_t item = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (item >= T * (H + Hkv)) return;
  const int64_t t = item / (H + Hkv);
  const int hh = (int)(item % (H + Hkv));
  const bool is_q = hh < H;
  const int h = is_q ? hh : hh - H;
  const int per = D / 32, half = D / 2;
  const bf16* src = qkv + t * qs + (is_q ? (int64_t)h * D : (int64_t)(H + h) * D);
  float x[kMaxHeadDim / 32];
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < kMaxHeadDim / 32; ++j) {
    if (j < per) {
      x[j] = bf2f(src[lane + 32 * j]);
      ss = fadd(ss, fmul(x[j], x[j]));
    }
  }
  ss = warp_sum(ss);
  const float inv = rsqrtf(fadd(fmul(ss, 1.0f / (float)D), eps));
  const int64_t p = pos[t];
  const bf16* cr = cos + p * half;
  const bf16* sr = sin + p * half;
  bf16* dst;
  bf16* vdst = nullptr;
  const bf16* w = is_q ? qw : kw;
  if (is_q) {
    dst = q_out + (t * H + h) * D;
  } else {
    const int64_t base = kv_page[t] * ps_kv + (kv_off[t] * Hkv + h) * D;
    if constexpr (kFp8) {  // the one plane, in bytes
      uint8_t* v8 = reinterpret_cast<uint8_t*>(kv) + base;
#pragma unroll
      for (int j = 0; j < kMaxHeadDim / 32; ++j)
        if (j < per) v8[lane + 32 * j] = lmk_kv_e4m3(f2bf(fmul(x[j], inv)));
      return;
    }
    dst = kv + base;  // K plane (unused with k_eq_v)
    vdst = k_eq_v ? kv + base : kv + base + (int64_t)page_size * Hkv * D;
  }
  if (!is_q) {  // V: unweighted norm of the v head (of the k head with k_eq_v)
    if (k_eq_v) {
#pragma unroll
      for (int j = 0; j < kMaxHeadDim / 32; ++j)
        if (j < per) vdst[lane + 32 * j] = f2bf(fmul(x[j], inv));
      return;
    }
    const bf16* vs = qkv + t * qs + (int64_t)(H + Hkv + h) * D;
    float vv[kMaxHeadDim / 32];
    float vss = 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxHeadDim / 32; ++j) {
      if (j < per) {
        vv[j] = bf2f(vs[lane + 32 * j]);
        vss = fadd(vss, fmul(vv[j], vv[j]));
      }
    }
    vss = warp_sum(vss);
    const float vinv = rsqrtf(fadd(fmul(vss, 1.0f / (float)D), eps));
#pragma unroll
    for (int j = 0; j < kMaxHeadDim / 32; ++j)
      if (j < per) vdst[lane + 32 * j] = f2bf(fmul(vv[j], vinv));
  }
  float n[kMaxHeadDim / 32];
#pragma unroll
  for (int j = 0; j < kMaxHeadDim / 32; ++j)
    if (j < per) n[j] = fmul(fmul(x[j], inv), bf2f(w[lane + 32 * j]));
  const int hj = per / 2;  // the partner of element j is element j + hj
#pragma unroll
  for (int j = 0; j < kMaxHeadDim / 64; ++j) {
    if (j < hj) {
      const int i = lane + 32 * j;  // i < D / 2
      const float c = bf2f(cr[i]), s = bf2f(sr[i]);
      const float x1 = n[j], x2 = n[j + hj];
      dst[i] = f2bf(fsub(fmul(x1, c), fmul(x2, s)));
      dst[i + half] = f2bf(fadd(fmul(x2, c), fmul(x1, s)));
    }
  }
}

}  // namespace

LMK_API int lmk_g4_rmsnorm(const void* x, int64_t x_stride, const void* weight, void* out, int64_t rows, int D,
                           float eps, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const int threads = D >= 4096 ? 1024 : D >= 1024 ? 512 : 256;
  return lmk_launch(g4_rmsnorm_kernel, dim3((unsigned)rows), dim3(threads), D * sizeof(float), stream, (const bf16*)x,
                    x_stride, (const bf16*)weight, (bf16*)out, D, eps);
}

LMK_API int lmk_g4_sandwich(const void* a, int64_t a_stride, const void* residual, const void* w_post,
                            const void* w_next, const void* scalar, void* r_out, void* x_out, int64_t rows, int D,
                            float eps, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const uintptr_t ptrs = (uintptr_t)a | (uintptr_t)residual | (uintptr_t)w_post | (uintptr_t)w_next |
                         (uintptr_t)r_out | (uintptr_t)x_out;
  const int nvec = D / 8;
  if (D % 8 == 0 && a_stride % 8 == 0 && (ptrs & 15) == 0 && nvec <= 4 * 1024) {
    const int threads = nvec >= 1024 ? 1024 : (nvec + 31) / 32 * 32;
    const int nv = (nvec + threads - 1) / threads;
#define LMK_SANDWICH(NV)                                                                                      \
  lmk_launch(g4_sandwich_vec_kernel<NV>, dim3((unsigned)rows), dim3(threads), 0, stream, (const bf16*)a,       \
             a_stride, (const bf16*)residual, (const bf16*)w_post, (const bf16*)w_next, (const bf16*)scalar,    \
             (bf16*)r_out, (bf16*)x_out, D, eps)
    if (nv == 1) return LMK_SANDWICH(1);
    if (nv == 2) return LMK_SANDWICH(2);
    return LMK_SANDWICH(4);
#undef LMK_SANDWICH
  }
  const int threads = D >= 4096 ? 1024 : D >= 1024 ? 512 : 256;
  return lmk_launch(g4_sandwich_kernel, dim3((unsigned)rows), dim3(threads), D * sizeof(float), stream,
                    (const bf16*)a, a_stride, (const bf16*)residual, (const bf16*)w_post, (const bf16*)w_next,
                    (const bf16*)scalar, (bf16*)r_out, (bf16*)x_out, D, eps);
}

LMK_API int lmk_gelu_tanh_mul(const void* gu, int64_t gu_stride, void* out, int64_t T, int I, cudaStream_t stream) {
  const int64_t n = T * I;
  if (n <= 0) return 0;
  return lmk_launch(gelu_tanh_mul_kernel, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, stream, (const bf16*)gu,
                    gu_stride, (bf16*)out, T, I);
}

// `kv_fp8` 1: the KV layer holds E4M3 bytes (k_eq_v only), its page stride in bytes.
LMK_API int lmk_g4_qkv(const void* qkv, int64_t qkv_stride, int k_eq_v, const void* q_weight, const void* k_weight,
                       const void* positions, const void* cos, const void* sin, void* q_out, void* kv_layer,
                       int64_t kv_page_stride, int page_size, int kv_fp8, const void* kv_page, const void* kv_off,
                       int64_t T, int H, int Hkv, int D, float eps, cudaStream_t stream) {
  if (T <= 0) return 0;
  if (D % 64 || D > kMaxHeadDim || kv_fp8 < 0 || kv_fp8 > 1 || (kv_fp8 && !k_eq_v)) return -1;
  const int64_t n = T * (H + Hkv);
#define LMK_G4_QKV(FP8)                                                                                        \
  lmk_launch(lmk_pdl_flag() ? g4_qkv_kernel<true, FP8> : g4_qkv_kernel<false, FP8>, dim3((unsigned)((n + 7) / 8)), \
             dim3(256), 0, stream, (const bf16*)qkv, qkv_stride, k_eq_v, (const bf16*)q_weight,                 \
             (const bf16*)k_weight, (const int64_t*)positions, (const bf16*)cos, (const bf16*)sin, (bf16*)q_out, \
             (bf16*)kv_layer, kv_page_stride, page_size, (const int64_t*)kv_page, (const int64_t*)kv_off, T, H,  \
             Hkv, D, eps)
  if (kv_fp8) return LMK_G4_QKV(true);
  return LMK_G4_QKV(false);
#undef LMK_G4_QKV
}

// For the tests: the FP8 KV's store and read conversions (kv_fp8.cuh) over n values, BF16 to E4M3 bytes
// (`to_fp8` 1) or E4M3 bytes to BF16 (0; n even).
namespace {
__global__ void g4_kv_cast_kernel(const void* __restrict__ in, void* __restrict__ out, int64_t n, int to_fp8) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (to_fp8) {
    if (i < n) static_cast<uint8_t*>(out)[i] = lmk_kv_e4m3(static_cast<const bf16*>(in)[i]);
  } else if (2 * i < n) {
    static_cast<uint32_t*>(out)[i] = lmk_kv_bf16x2(static_cast<const uint16_t*>(in)[i]);
  }
}
}  // namespace

LMK_API int lmk_g4_kv_cast(const void* in, void* out, int64_t n, int to_fp8, cudaStream_t stream) {
  if (n <= 0) return 0;
  if (!to_fp8 && n % 2) return -1;
  const int64_t threads = to_fp8 ? n : n / 2;
  return lmk_launch(g4_kv_cast_kernel, dim3((unsigned)((threads + 255) / 256)), dim3(256), 0, stream, in, out, n,
                    to_fp8);
}

// The quantizing producers. Each returns -1 for a shape or alignment it does not take. The Python wrappers then run
// the BF16 kernel and the standalone quantization, which give the same bits.
LMK_API int lmk_g4_rmsnorm_quant(const void* x, int64_t x_stride, const void* weight, void* q, void* s, int64_t ld,
                                 int64_t rows, int D, float eps, cudaStream_t stream) {
  if (rows <= 0) return 0;
  if (D % kGroup || ld < rows) return -1;
  const int threads = D >= 4096 ? 1024 : D >= 1024 ? 512 : 256;  // as lmk_g4_rmsnorm: same reduction order
  return lmk_launch(g4_rmsnorm_quant_kernel, dim3((unsigned)rows), dim3(threads), D * sizeof(float), stream,
                    (const bf16*)x, x_stride, (const bf16*)weight, (uint8_t*)q, (float*)s, ld, D, eps);
}

LMK_API int lmk_g4_sandwich_quant(const void* a, int64_t a_stride, const void* residual, const void* w_post,
                                  const void* w_next, const void* scalar, void* r_out, void* q, void* s, int64_t ld,
                                  int64_t rows, int D, float eps, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const uintptr_t ptrs = (uintptr_t)a | (uintptr_t)residual | (uintptr_t)w_post | (uintptr_t)w_next |
                         (uintptr_t)r_out | (uintptr_t)q;
  const int nvec = D / 8;
  // only where lmk_g4_sandwich takes its vector kernel (same threads, so same reduction order)
  if (D % kGroup || a_stride % 8 || (ptrs & 15) || nvec > 4 * 1024 || ld < rows) return -1;
  const int threads = nvec >= 1024 ? 1024 : (nvec + 31) / 32 * 32;
  const int nv = (nvec + threads - 1) / threads;
#define LMK_SANDWICH_Q(NV)                                                                                    \
  lmk_launch(g4_sandwich_quant_kernel<NV>, dim3((unsigned)rows), dim3(threads), 0, stream, (const bf16*)a,     \
             a_stride, (const bf16*)residual, (const bf16*)w_post, (const bf16*)w_next, (const bf16*)scalar,    \
             (bf16*)r_out, (uint8_t*)q, (float*)s, ld, D, eps)
  if (nv == 1) return LMK_SANDWICH_Q(1);
  if (nv == 2) return LMK_SANDWICH_Q(2);
  return LMK_SANDWICH_Q(4);
#undef LMK_SANDWICH_Q
}

LMK_API int lmk_gelu_tanh_mul_quant(const void* gu, int64_t gu_stride, void* q, void* s, int64_t ld, int64_t T, int I,
                                    cudaStream_t stream) {
  if (T <= 0) return 0;
  if (I % kGroup || gu_stride % 4 || (uintptr_t)gu % 8 || ld < T) return -1;
  constexpr int kWarps = 8;
  const int64_t items = T * (I / kGroup);
  return lmk_launch(gelu_tanh_mul_quant_kernel, dim3((unsigned)((items + kWarps - 1) / kWarps)), dim3(kWarps * 32), 0,
                    stream, (const bf16*)gu, gu_stride, (uint8_t*)q, (float*)s, T, I, ld);
}

// Shared helpers for llm-mini-infer's CUDA kernels (docs/architecture.md, "Kernels"). Plain CUDA, no torch headers.
// nvcc alone builds the library (llm_mini_infer/kernels/build.py). Callers use ctypes with raw device pointers and
// their own stream.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define LMK_API extern "C" __attribute__((visibility("default")))

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float bf2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2bf(float x) { return __float2bfloat16_rn(x); }

// Rounded fp32 operations. The torch reference runs one op per kernel, so it contracts nothing into an FMA. Kernels
// that must reproduce its rounding use these instead of `*`, `+`, `-`.
__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float fdiv(float a, float b) { return __fdiv_rn(a, b); }

// torch's CUDA formulas (each op rounded to fp32): sigmoid = 1 / (1 + exp(-x)), silu = x / (1 + exp(-x)).
__device__ __forceinline__ float t_sigmoid(float x) { return fdiv(1.0f, fadd(1.0f, expf(-x))); }
__device__ __forceinline__ float t_silu(float x) { return fdiv(x, fadd(1.0f, expf(-x))); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024). Every thread gets the result. `smem` holds 32 floats.
__device__ __forceinline__ float block_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v);
  __syncthreads();  // a previous call may still read smem
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  v = lane < nw ? smem[lane] : 0.0f;
  return warp_sum(v);
}

// The fixed split of our batch-invariant attention kernels (csrc/attn512.cu, csrc/attn_window.cu; kernels/attn512.py
// `split_size`, `query_splits`): a query at absolute position p splits its keys [0, p] at multiples of keys · 2^m, the
// least m >= 0 that keeps its p + 1 keys within LMK_MAX_SPLITS splits (the merges' lanes). A query whose p + 1 keys are
// at most `whole` takes them whole.
constexpr int LMK_MAX_SPLITS = 32;

__device__ __forceinline__ int lmk_split_size(int p, int keys) {
  long long s = keys;
  while ((long long)p + 1 > s * LMK_MAX_SPLITS) s *= 2;
  return (int)s;
}

// Query p's keys in split j, [lo, hi) (empty when lo >= hi): taken whole in split 0 while p + 1 <= whole.
__device__ __forceinline__ void lmk_split_range(int p, int j, int keys, int whole, int& lo, int& hi) {
  if (p + 1 <= whole) {
    lo = 0;
    hi = j == 0 ? p + 1 : 0;
    return;
  }
  const int s = lmk_split_size(p, keys);
  lo = j * s;
  hi = min(lo + s, p + 1);
}

// The last position whose query splits as p's does (the same size, or whole).
__device__ __forceinline__ int lmk_split_class_end(int p, int keys, int whole) {
  return p + 1 <= whole ? whole - 1 : lmk_split_size(p, keys) * LMK_MAX_SPLITS - 1;
}

// The union [lo, hi) of split j's ranges over the consecutive positions [p0, p1] (one size class at a time, each
// class's largest range at its last position); lo = hi = 0 when empty.
__device__ __forceinline__ void lmk_split_union(int p0, int p1, int j, int keys, int whole, int& lo, int& hi) {
  lo = 0x7fffffff;
  hi = 0;
  for (int p = p0; p <= p1;) {
    const int e = min(lmk_split_class_end(p, keys, whole), p1);
    int a, b;
    lmk_split_range(e, j, keys, whole, a, b);
    if (a < b) {
      lo = min(lo, a);
      hi = max(hi, b);
    }
    p = e + 1;
  }
  if (lo >= hi) lo = hi = 0;
}

static inline int lmk_status() { return (int)cudaGetLastError(); }

// Programmatic dependent launch (PDL; docs/architecture.md, "CUDA graphs"). While the library's flag is on
// (`lmk_set_pdl`, csrc/launch.cu), `lmk_launch` adds the programmatic-stream-serialization attribute. The kernel may
// then start while the previous kernel on the stream finishes. In a CUDA graph, the edge between them becomes a
// programmatic one.
// Every kernel launched that way calls `lmk_pdl_wait()` in every CTA before it reads or writes memory that a
// predecessor may touch. Only constant inputs (the weights) may be read before it. The wait returns when the previous
// kernel has completed and its writes are visible, so every bit equals the serialized launch's. Without the attribute
// (flag off, the default), the wait returns at once.
// `lmk_pdl_trigger()` lets the next kernel launch before this one ends. That kernel then waits in its own
// `lmk_pdl_wait`. Without a trigger, the next kernel launches as this one's CTAs exit.
// The trigger lets only a successor that has the attribute start early. Our kernels and the CUTLASS GEMMs carry the
// attribute only while the flag is on. FlashInfer's attention carries it by default (SM 9.0+). So the kernels before
// FlashInfer (the attention inputs: Gemma 4's `g4_qkv` and Qwen's `qk_norm_rope`) trigger only while the flag is on
// (an instantiation each way, `kTrigger`). With the flag off, FlashInfer starts at their end, as after a serialized
// launch.
__device__ __forceinline__ void lmk_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void lmk_pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// The library's PDL flag: one per process (the static of an inline function is shared by every translation unit).
inline int& lmk_pdl_flag() {
  static int on = 0;
  return on;
}

// kernel<<<grid, block, smem, stream>>>(args...), with the PDL attribute while the flag is on. Returns a cudaError_t.
template <typename... Params, typename... Args>
static inline int lmk_launch(void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                             Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  if (lmk_pdl_flag()) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  return (int)cudaLaunchKernelEx(&cfg, kernel, static_cast<Params>(args)...);
}

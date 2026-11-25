// FP8 E4M3 KV values: Gemma 4's full layers with `dev.cache.kv_cache_dtype fp8` (the reference is `kv_fp8` and
// `kv_read` in kernels/gemma_ops.py). A store rounds the BF16 value that a BF16 pool would hold to nearest even and
// saturates (`.satfinite`: an overflow or an infinity becomes ±448, a NaN stays a NaN). A read is exact: every E4M3
// value is an FP16 value and a BF16 value.
#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "common.cuh"

__device__ __forceinline__ uint8_t lmk_kv_e4m3(bf16 x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(bf2f(x), __NV_SATFINITE, __NV_E4M3));
}

// Two E4M3 bytes (the first in the low byte) as two packed BF16 values (the first in the low half).
__device__ __forceinline__ uint32_t lmk_kv_bf16x2(uint32_t two) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xffffu), __NV_E4M3);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
  return *reinterpret_cast<const uint32_t*>(&b);
}

// Paged attention for Gemma 4's full-attention layers: head dim 512, GQA, causal, with a given scale. The kernel
// derives K from the one stored plane (`attention_k_eq_v`). The pool holds V = k / rms(k) per position. K = RoPE(V ⊙
// k_norm.weight) at the key's position (reference: `k_from_v` in kernels/gemma_ops.py). FlashInfer 0.7.2 has a
// head-dim-512 path on SM120, at the cost of a separate V plane. This kernel reads half the KV of a separate K and V
// pool (models/gemma4/model.py).
//
// **Work.** A work item is (row, tile, KV head, split). The high 16 bits of its split word hold the split count n of
// its group. A tile is 16 query rows of one KV head: the flattened (position, group head) pairs [16 t, 16 t + 16) of
// the row's queries, position-major. G is the number of query heads per KV head. A decode row's tile holds its G heads,
// and a prefill tile holds 16 / G consecutive positions.
//
// A split is a range of the tile's keys. With `split_keys` 0 (the wave split) split s is [s × chunk, (s + 1) × chunk),
// chunk = ceil(limit / n) rounded up to 16, where limit is the tile's causal end. With `split_keys` > 0 (the fixed
// split) each query row has its own size from its absolute position p (`split_size`: a decode row's from `split_keys`,
// a prefill row's from `prefill_keys` once its p + 1 keys pass `whole`, below which it takes them whole), and split s
// holds its keys [s × size, (s + 1) × size). The CTA walks the hull of its rows' ranges and masks each row to its own,
// so a query's result depends on its keys alone, not on the split count, its tile, its chunk or the batch
// (kernels/attn512.py, the fixed split).
//
// A group of one split writes the tile's output. A group of more splits writes fp32 partials (O unnormalised, m, l),
// which `attn512_merge` combines. The host builds the item and group tables (kernels/attn512.py). Lengths come from the
// device row table, so a CUDA graph replays with new lengths. An item or group whose row is −1 is padding, and its CTA
// exits.
//
// **CTA.** 4 warps. Warp w owns the QK dims [64 w, 64 w + 64) and [256 + 64 w, 256 + 64 w + 64), so each RoPE pair (d,
// d + 256) stays in one thread. It also owns the output dims [128 w, 128 w + 128). Each tile has 16 keys,
// double-buffered with cp.async in 16-byte chunks, XOR-swizzled by key against bank conflicts. Each warp computes its
// partial S = Q Kᵀ (mma.sync m16n8k16). It builds K in registers from the V plane: × k_norm.weight, rotated where the
// frequency is non-zero, rounded to BF16.
//
// The four partials sum through shared memory in a fixed order, so every warp gets the same S. Then come the online
// softmax (fp32, exp2) and P V on the warp's output dims. Keys past the range are zero-filled (a stale page never
// reaches the product) and masked to −inf. A row whose split holds none of its keys keeps m = −inf and l = 0, and the
// merge skips it.
//
// **FP8 pools** (`FP8`: `cache.kv_cache_dtype fp8`). The plane holds E4M3 bytes (kv_fp8.cuh). A tile of 16 keys is
// then 8 KiB. It lands by cp.async in one of two unswizzled 8 KiB stages, and each tile is converted once, exactly,
// into the swizzled BF16 tile that the BF16 path loads directly (FlashInfer 0.7.2's FP8 prefill does the same). Both
// fit in the BF16 path's two 16 KiB stages: the landing stages share the second, the BF16 tile is the first (the Q tile
// aliases it). Shared memory, and so the CTAs an SM holds, stay the same. Everything after the conversion is the BF16
// path's code, so an FP8 pool gives the bits of a BF16 pool holding the same values. The conversion's barrier replaces
// the one that ends a tile on the BF16 path, which the FP8 path does not need (see the loop).
//
// Programmatic dependent launch (common.cuh): each kernel here waits for its predecessor as its first statement, then
// lets the next kernel launch. The launcher adds the attribute while the library's flag is on (the runner's
// `dev.runner.cuda.pdl`).
#include "common.cuh"
#include "kv_fp8.cuh"

namespace {

constexpr int D = 512;
constexpr int HALF = 256;
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr int ROWS = 16;  // query rows per tile
constexpr int KT = 16;    // keys per KV tile
constexpr int CHUNKS = D / 8;  // 16-byte chunks per key row
constexpr int TILE_BYTES = KT * D * 2;
constexpr int CHUNKS8 = D / 16;     // 16-byte chunks per key row of an FP8 pool
constexpr int LAND_BYTES = KT * D;  // an FP8 tile
constexpr float LOG2E = 1.4426950408889634f;
constexpr int MAX_SPLITS = LMK_MAX_SPLITS;  // the merge's lanes (kernels/attn512.py MAX_SPLITS)

struct Row {
  int slot, qstart, nq, kvlen;
};
struct Item {
  int row, tile, kvh, split;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Byte offset of 16-byte chunk c of row r in a [rows][CHUNKS] tile, XOR-swizzled by r % 8.
__device__ __forceinline__ int swz(int r, int c) { return (r * CHUNKS + (c ^ (r & 7))) * 16; }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  const int n = valid ? 16 : 0;  // 0: zero-fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float2 unpack(uint32_t v) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
}
__device__ __forceinline__ uint32_t pack(float x, float y) {
  __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&h);
}

// The kernel stages the rotating dims' cos and sin per tile. Gemma 4's full layers rotate the first 64 dims of each
// 256-dim half (proportional RoPE). All of them sit in warp 0's dims, and warp 0's scattered table reads would be the
// critical path of every tile (the kernel without rotation takes half the time). A row of 72 elements (64 staged plus a
// pad) puts warp 0's reads of 8 keys at once on distinct banks. Wider rotations read the tables directly.
constexpr int ROT_STAGED = 64;
constexpr int CS_STRIDE = 72;
constexpr int CS_CHUNKS = ROT_STAGED * 2 / 16;  // 16-byte chunks of a staged row

struct Smem {
  // two stages of V-plane tiles [KT keys][D]; the Q tile aliases stage 0. FP8 pools: stage 0 is the converted BF16
  // tile, stage 1 the two FP8 landing stages [KT keys][D bytes]
  alignas(128) unsigned char kv[2][TILE_BYTES];
  float s[WARPS][ROWS][KT];                     // partial scores
  alignas(16) bf16 cs[2][2][KT][CS_STRIDE];     // two stages of cos and sin of the tile's keys, dims [0, ROT_STAGED)
};

// K for one B fragment pair: u0 (dims d0 and d0 + 1 of the low half) and u1 (the same dims + 256) of one key, times the
// weight, rotated where d < rot. Returns packed BF16 for the low and the high k-step.
__device__ __forceinline__ void k_pair(uint32_t u0, uint32_t u1, float w0a, float w0b, float w1a, float w1b, int d0,
                                       int rot, const bf16* cos_row, const bf16* sin_row, uint32_t& k0, uint32_t& k1) {
  const float2 a = unpack(u0), b = unpack(u1);
  const float x0 = fmul(a.x, w0a), x1 = fmul(a.y, w0b);  // low half: dims d0, d0 + 1
  const float y0 = fmul(b.x, w1a), y1 = fmul(b.y, w1b);  // high half: dims d0 + 256, d0 + 257
  if (d0 < rot) {
    const float c0 = bf2f(cos_row[d0]), c1 = bf2f(cos_row[d0 + 1]);
    const float s0 = bf2f(sin_row[d0]), s1 = bf2f(sin_row[d0 + 1]);
    k0 = pack(fsub(fmul(x0, c0), fmul(y0, s0)), fsub(fmul(x1, c1), fmul(y1, s1)));
    k1 = pack(fadd(fmul(y0, c0), fmul(x0, s0)), fadd(fmul(y1, c1), fmul(x1, s1)));
  } else {
    k0 = pack(x0, x1);
    k1 = pack(y0, y1);
  }
}

template <bool SPLIT, bool FP8>
__global__ void __launch_bounds__(THREADS, 2)
    attn512_kernel(const bf16* __restrict__ q, int H, int G, int Hkv, const bf16* __restrict__ kv, int64_t page_stride,
                   int page_size, const int* __restrict__ page_table, int64_t pt_stride, const bf16* __restrict__ kw,
                   const bf16* __restrict__ cos, const bf16* __restrict__ sin, int rot, float scale,
                   const Row* __restrict__ rows, const Item* __restrict__ items, const int* __restrict__ order,
                   int split_keys, int n_dec, int whole, int prefill_keys, bf16* __restrict__ out,
                   float* __restrict__ o_part, float* __restrict__ ml_part) {
  lmk_pdl_wait();  // PDL: read nothing before the predecessor finishes (common.cuh)
  lmk_pdl_trigger();
  extern __shared__ unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int idx = order ? order[blockIdx.x] : (int)blockIdx.x;  // the launch order (kernels/attn512.py `plan`)
  const Item it = items[idx];
  if (it.row < 0) return;  // padding in a graph's item table
  const int n_splits = SPLIT ? max(1, it.split >> 16) : 1;  // the split count of this item's group
  const int split = it.split & 0xffff;
  const Row rw = rows[it.row];
  const int q0 = rw.kvlen - rw.nq;  // position of the row's first query
  // Rows of this tile: flattened f = 16 t + i maps to query position f / G and head kvh * G + f % G.
  const int f0 = it.tile * ROWS;
  const int p_last = min((f0 + ROWS - 1) / G, rw.nq - 1);
  const int limit = min(rw.kvlen, q0 + p_last + 1);  // the keys this tile may see
  int k_begin = 0, k_end = limit;
  // the fixed split: this row's base and whole (a decode row's, or a prefill row's), the tile's rows' ranges' hull
  const bool fixed = SPLIT && n_splits > 1 && split_keys > 0;
  const int keys = it.row < n_dec ? split_keys : prefill_keys, whole_r = it.row < n_dec ? 0 : whole;
  if (fixed) {  // a split past the tile's keys is empty: m = −inf, l = 0, and the merge skips it
    lmk_split_hull(q0 + f0 / G, q0 + p_last, split, keys, whole_r, k_begin, k_end);
  } else if (n_splits > 1) {  // the wave split
    const int chunk = ((limit + n_splits - 1) / n_splits + KT - 1) / KT * KT;
    k_begin = split * chunk;
    k_end = min(limit, k_begin + chunk);
  }
  const int64_t item = idx;  // its partials' slot

  // -- the Q tile: rows of 512 dims into stage 0 (swizzled), then A fragments per warp --
  for (int c = tid; c < ROWS * CHUNKS; c += THREADS) {
    const int r = c / CHUNKS, ch = c % CHUNKS;
    const int f = f0 + r, p = f / G;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (p < rw.nq) {
      const int head = it.kvh * G + f % G;
      v = *reinterpret_cast<const uint4*>(q + ((int64_t)(rw.qstart + p) * H + head) * D + ch * 8);
    }
    *reinterpret_cast<uint4*>(sm.kv[0] + swz(r, ch)) = v;
  }
  __syncthreads();
  uint32_t qa[8][4];  // k-steps 0-3: the low half's dims 64 w + 16 j; k-steps 4-7: the same dims + 256
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int dim = (j < 4 ? 0 : HALF) + 64 * warp + 16 * (j & 3);
    const int r = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int ch = dim / 8 + (lane >> 4);
    ldsm_x4(smem_u32(sm.kv[0] + swz(r, ch)), qa[j][0], qa[j][1], qa[j][2], qa[j][3]);
  }
  // This thread's k_norm weights: per low k-step j, dims (64 w + 16 j) + 2 t4 + {0, 1} and + 8 + {0, 1}; and + 256.
  float wl[4][4], wh[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int d = 64 * warp + 16 * j + 2 * t4;
    wl[j][0] = bf2f(kw[d]);
    wl[j][1] = bf2f(kw[d + 1]);
    wl[j][2] = bf2f(kw[d + 8]);
    wl[j][3] = bf2f(kw[d + 9]);
    wh[j][0] = bf2f(kw[d + HALF]);
    wh[j][1] = bf2f(kw[d + HALF + 1]);
    wh[j][2] = bf2f(kw[d + HALF + 8]);
    wh[j][3] = bf2f(kw[d + HALF + 9]);
  }
  __syncthreads();  // stage 0 now holds KV

  // This thread's rows are g and g + 8, with their absolute positions, validity and keys [rlo, rhi) in this split: the
  // tile's range cut at the row's causal end (the fixed split: the row's own range), empty past the row's queries, so
  // one unsigned comparison masks a key.
  int rpos[2], rlo[2], rhi[2];
  bool rok[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int f = f0 + g + 8 * h, p = f / G;
    rok[h] = p < rw.nq;
    rpos[h] = q0 + p;
    rlo[h] = k_begin;
    rhi[h] = min(k_end, rpos[h] + 1);
    if (fixed && rok[h]) lmk_split_range(rpos[h], split, keys, whole_r, rlo[h], rhi[h]);
    if (!rok[h] || rhi[h] < rlo[h]) rhi[h] = rlo[h];
  }
  float o[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
  const float sl2 = scale * LOG2E;

  const int n_tiles = k_end > k_begin ? (k_end - k_begin + KT - 1) / KT : 0;
  const int* pt = page_table + (int64_t)rw.slot * pt_stride;
  const bool staged = rot > 0 && rot <= ROT_STAGED;  // cos and sin come from shared memory (see ROT_STAGED)
  const bool page_tiles = page_size % KT == 0;  // a tile (its first key a multiple of KT) lies in one page
  auto load_tile = [&](int ti, int stage) {
    const int kbase = k_begin + ti * KT;
    if constexpr (FP8) {  // the same offsets in bytes (the pool view's elements), into an unswizzled landing stage
      const unsigned char* kv8 = reinterpret_cast<const unsigned char*>(kv);
      const unsigned char* tbase = page_tiles ? kv8 + (int64_t)pt[kbase / page_size] * page_stride +
                                                    (int64_t)(kbase % page_size) * Hkv * D : kv8;
      unsigned char* land = sm.kv[1] + stage * LAND_BYTES;
      for (int c = tid; c < KT * CHUNKS8; c += THREADS) {
        const int r = c / CHUNKS8, ch = c % CHUNKS8;
        const int key = kbase + r;
        const bool ok = key < k_end;
        const unsigned char* src = kv8;
        if (ok) {
          src = page_tiles ? tbase + ((int64_t)r * Hkv + it.kvh) * D + ch * 16
                           : kv8 + (int64_t)pt[key / page_size] * page_stride +
                                 ((int64_t)(key % page_size) * Hkv + it.kvh) * D + ch * 16;
        }
        cp_async16(smem_u32(land + r * D + ch * 16), src, ok);  // zero-fill: E4M3 0x00 is +0
      }
    } else {
      const bf16* tbase = page_tiles ? kv + (int64_t)pt[kbase / page_size] * page_stride +
                                           (int64_t)(kbase % page_size) * Hkv * D : kv;  // the tile's page, found once
      for (int c = tid; c < KT * CHUNKS; c += THREADS) {
        const int r = c / CHUNKS, ch = c % CHUNKS;
        const int key = kbase + r;
        const bool ok = key < k_end;
        const bf16* src = kv;
        if (ok) {
          src = page_tiles ? tbase + ((int64_t)r * Hkv + it.kvh) * D + ch * 8
                           : kv + (int64_t)pt[key / page_size] * page_stride +
                                 ((int64_t)(key % page_size) * Hkv + it.kvh) * D + ch * 8;
        }
        cp_async16(smem_u32(sm.kv[stage] + swz(r, ch)), src, ok);
      }
    }
    if (staged) {  // keys past k_end load zeros: their scores are masked and their V rows are zero
      for (int c = tid; c < KT * 2 * CS_CHUNKS; c += THREADS) {
        const int r = c / (2 * CS_CHUNKS), t = (c / CS_CHUNKS) % 2, ch = c % CS_CHUNKS;
        const int key = kbase + r;
        const bool ok = key < k_end;
        const bf16* src = (t ? sin : cos) + (ok ? (int64_t)key * HALF + ch * 8 : 0);
        cp_async16(smem_u32(&sm.cs[stage][t][r][ch * 8]), src, ok);
      }
    }
    cp_commit();
  };
  if (n_tiles) load_tile(0, 0);
  for (int ti = 0; ti < n_tiles; ++ti) {
    const int stage = ti & 1;
    if (ti + 1 < n_tiles) {
      load_tile(ti + 1, stage ^ 1);
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    const unsigned char* tile;
    if constexpr (FP8) {
      // The landed FP8 tile into the BF16 tile, each thread 8 dims (8 bytes, one swizzled 16-byte chunk) at a time;
      // the previous tile's readers passed the barrier that ended its iteration
      const unsigned char* land = sm.kv[1] + stage * LAND_BYTES;
      for (int c = tid; c < KT * CHUNKS; c += THREADS) {
        const int r = c / CHUNKS, ch = c % CHUNKS;
        const uint2 b = *reinterpret_cast<const uint2*>(land + r * D + ch * 8);
        const uint4 v = make_uint4(lmk_kv_bf16x2(b.x), lmk_kv_bf16x2(b.x >> 16), lmk_kv_bf16x2(b.y),
                                   lmk_kv_bf16x2(b.y >> 16));
        *reinterpret_cast<uint4*>(sm.kv[0] + swz(r, ch)) = v;
      }
      __syncthreads();
      tile = sm.kv[0];
    } else {
      tile = sm.kv[stage];
    }
    const int kbase = k_begin + ti * KT;
    // -- partial S over this warp's dims --
    float s[2][4];
#pragma unroll
    for (int n = 0; n < 2; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
    // This thread's keys (B fragments): kbase + g (n-tile 0) and kbase + 8 + g (n-tile 1).
    const bf16* crow[2];
    const bf16* srow[2];
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      if (staged) {
        crow[n] = sm.cs[stage][0][8 * n + g];
        srow[n] = sm.cs[stage][1][8 * n + g];
      } else {
        const int key = min(kbase + 8 * n + g, max(k_end - 1, 0));
        crow[n] = cos + (int64_t)key * HALF;
        srow[n] = sin + (int64_t)key * HALF;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int dl = 64 * warp + 16 * j;  // first dim of the low k-step; its RoPE pair is dl + 256
      uint32_t ul[4], uh[4];
      {
        const int r = (lane & 7) + 8 * (lane >> 4);
        const int chl = dl / 8 + ((lane >> 3) & 1);
        ldsm_x4(smem_u32(tile + swz(r, chl)), ul[0], ul[1], ul[2], ul[3]);
        ldsm_x4(smem_u32(tile + swz(r, chl + HALF / 8)), uh[0], uh[1], uh[2], uh[3]);
      }
      // ul[0], ul[1]: n-tile 0's b0 and b1 (dims dl + 2 t4 and + 8); ul[2], ul[3]: n-tile 1's
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        uint32_t kl0, kh0, kl1, kh1;
        const int d = dl + 2 * t4;
        k_pair(ul[2 * n], uh[2 * n], wl[j][0], wl[j][1], wh[j][0], wh[j][1], d, rot, crow[n], srow[n], kl0, kh0);
        k_pair(ul[2 * n + 1], uh[2 * n + 1], wl[j][2], wl[j][3], wh[j][2], wh[j][3], d + 8, rot, crow[n], srow[n],
               kl1, kh1);
        mma(s[n], qa[j], kl0, kl1);
        mma(s[n], qa[j + 4], kh0, kh1);
      }
    }
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      sm.s[warp][g][8 * n + 2 * t4] = s[n][0];
      sm.s[warp][g][8 * n + 2 * t4 + 1] = s[n][1];
      sm.s[warp][g + 8][8 * n + 2 * t4] = s[n][2];
      sm.s[warp][g + 8][8 * n + 2 * t4 + 1] = s[n][3];
    }
    __syncthreads();
    // -- the whole S (the same sum in every warp), masked --
    float sc[2][4];
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int r = g + 8 * (e >> 1), col = 8 * n + 2 * t4 + (e & 1);
        float v = sm.s[0][r][col];
        v = fadd(v, sm.s[1][r][col]);
        v = fadd(v, sm.s[2][r][col]);
        v = fadd(v, sm.s[3][r][col]);
        const int key = kbase + col;
        const int h = e >> 1;
        const bool ok = (unsigned)(key - rlo[h]) < (unsigned)(rhi[h] - rlo[h]);  // rlo <= key < rhi
        sc[n][e] = ok ? v * sl2 : -INFINITY;
      }
    }
    // -- online softmax: rows g (e 0, 1) and g + 8 (e 2, 3); a quad of lanes shares a row --
    float mx[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float v = fmaxf(fmaxf(sc[0][2 * h], sc[0][2 * h + 1]), fmaxf(sc[1][2 * h], sc[1][2 * h + 1]));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
      mx[h] = fmaxf(m[h], v);
    }
    float alpha[2], ref[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      ref[h] = mx[h] == -INFINITY ? 0.0f : mx[h];
      alpha[h] = exp2f(m[h] - ref[h]);  // 0 when m = −inf
      m[h] = mx[h];
    }
    float p[2][4], rs[2] = {0.0f, 0.0f};
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        p[n][e] = exp2f(sc[n][e] - ref[e >> 1]);
        rs[e >> 1] += p[n][e];
      }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
      rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
      l[h] = l[h] * alpha[h] + rs[h];
    }
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      o[n][0] *= alpha[0];
      o[n][1] *= alpha[0];
      o[n][2] *= alpha[1];
      o[n][3] *= alpha[1];
    }
    // P as an A fragment (16 rows × 16 keys): the C layout of the two n-tiles is the A layout
    uint32_t pa[4] = {pack(p[0][0], p[0][1]), pack(p[0][2], p[0][3]), pack(p[1][0], p[1][1]),
                      pack(p[1][2], p[1][3])};
    // -- P V on this warp's output dims [128 w, 128 w + 128) --
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) {  // pairs of n-tiles of 8 dims
      const int dim = 128 * warp + 16 * nb;
      uint32_t v0, v1, v2, v3;
      // .trans loads (keys 0-7, dims dim..+8), (keys 8-15, same dims), (keys 0-7, dims +8..+16), (keys 8-15, same dims)
      const int r = (lane & 7) + 8 * ((lane >> 3) & 1);
      const int ch = dim / 8 + (lane >> 4);
      ldsm_x4_t(smem_u32(tile + swz(r, ch)), v0, v1, v2, v3);
      mma(o[2 * nb], pa, v0, v1);
      mma(o[2 * nb + 1], pa, v2, v3);
    }
    // The next tile rewrites the stage and the score buffer. FP8 pools need no barrier here: the next load writes the
    // other landing stage and cos/sin stage, read before this tile's conversion barrier and score barrier, and the
    // conversion and the scores follow the next tile's first barrier
    if constexpr (!FP8) __syncthreads();
  }

  // -- epilogue --
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = g + 8 * h;
    const int f = f0 + r, pidx = f / G;
    if (!rok[h]) continue;
    const int head = it.kvh * G + f % G;
    if (n_splits > 1) {
      float* op = o_part + (item * ROWS + r) * D;
#pragma unroll
      for (int n = 0; n < 16; ++n) {
        const int d = 128 * warp + 8 * n + 2 * t4;
        *reinterpret_cast<float2*>(op + d) = make_float2(o[n][2 * h], o[n][2 * h + 1]);
      }
      if (warp == 0 && t4 == 0) {
        ml_part[(item * ROWS + r) * 2] = m[h];
        ml_part[(item * ROWS + r) * 2 + 1] = l[h];
      }
    } else {
      const float inv = l[h] > 0.0f ? 1.0f / l[h] : 0.0f;
      bf16* dst = out + ((int64_t)(rw.qstart + pidx) * H + head) * D;
#pragma unroll
      for (int n = 0; n < 16; ++n) {
        const int d = 128 * warp + 8 * n + 2 * t4;
        *reinterpret_cast<__nv_bfloat162*>(dst + d) = __floats2bfloat162_rn(o[n][2 * h] * inv, o[n][2 * h + 1] * inv);
      }
    }
  }
}

// Combine the splits of each (row, tile, KV head) group: items [first, first + n) from the group table (first, n). One
// block handles one (group, query row of the tile), with 128 threads of 4 dims. The first warp turns the splits' (m, l)
// into weights in shared memory. Then every thread sums its dims over the splits with independent loads. The block
// exits for a group of one split (its CTA wrote the output), a padding group, or a row past the row's queries. One
// block that walks a group's 16 rows and its splits serially would take 28 µs per layer at batch 1, more than the
// attention itself.
constexpr int MAX_MERGE_SPLITS = MAX_SPLITS;
__global__ void attn512_merge_kernel(int H, int G, const Row* __restrict__ rows, const Item* __restrict__ items,
                                     const int2* __restrict__ groups, const float* __restrict__ o_part,
                                     const float* __restrict__ ml_part, bf16* __restrict__ out) {
  lmk_pdl_wait();  // PDL: read nothing before the predecessor finishes (common.cuh)
  lmk_pdl_trigger();
  __shared__ float wts[MAX_MERGE_SPLITS];
  __shared__ float inv_l;
  const int2 grp = groups[blockIdx.x / ROWS];
  const int n_splits = grp.y;
  if (grp.x < 0 || n_splits <= 1) return;
  const int r = blockIdx.x % ROWS;
  const int64_t i0 = grp.x;
  const Item it = items[i0];
  const Row rw = rows[it.row];
  const int f = it.tile * ROWS + r, p = f / G;
  if (p >= rw.nq) return;
  if (threadIdx.x < 32) {
    const int s = threadIdx.x;
    float m = -INFINITY, l = 0.0f;
    if (s < n_splits) {
      const float* ml = ml_part + ((i0 + s) * ROWS + r) * 2;
      m = ml[0];
      l = ml[1];
    }
    const bool live = s < n_splits && l > 0.0f;
    float mx = live ? m : -INFINITY;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const float w = live ? exp2f(m - mx) : 0.0f;
    if (s < n_splits) wts[s] = w;
    const float L = warp_sum(w * l);
    if (s == 0) inv_l = L > 0.0f ? 1.0f / L : 0.0f;
  }
  __syncthreads();
  const int d = threadIdx.x * 4;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
  for (int s = 0; s < n_splits; ++s) {
    const float w = wts[s];
    if (w == 0.0f) continue;
    const float4 v = *reinterpret_cast<const float4*>(o_part + ((i0 + s) * ROWS + r) * D + d);
    acc.x += w * v.x;
    acc.y += w * v.y;
    acc.z += w * v.z;
    acc.w += w * v.w;
  }
  const float inv = inv_l;
  const int head = it.kvh * G + f % G;
  bf16* dst = out + ((int64_t)(rw.qstart + p) * H + head) * D + d;
  *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(acc.x * inv, acc.y * inv);
  *reinterpret_cast<__nv_bfloat162*>(dst + 2) = __floats2bfloat162_rn(acc.z * inv, acc.w * inv);
}

}  // namespace

// rows: int32 [n_rows, 4] (slot, first token in q/out, n_query, KV length after the step). items: int32 [n_items, 4]
// (row, tile, KV head, split | n << 16), grouped by (row, tile, KV head) with a group's n splits consecutive (row −1:
// padding). groups: int32 [n_groups, 2] (first item, n) of each group (first −1: padding). q and out are [T, H, 512].
// kv is a layer's V plane [pages, page_size, Hkv, 512], in BF16 or, with `kv_fp8` 1, E4M3 bytes (its page stride
// then counts bytes). cos and sin are [max_pos, 256] (the kernel reads only dims < rot).
//
// `split` 0: every group has one split (the kernel does not read the items' n; no merge, `groups` unused). `split` 1:
// o_part [n_items, 16, 512] fp32 and ml_part [n_items, 16, 2] fp32 hold the partials of groups of more than one split
// (n <= 32), and the merge combines them. `split_keys` 0: a group's n splits divide its tile's keys evenly.
// `split_keys` a multiple of 16: the fixed split (see Work above), the rows [0, n_dec) decode rows sized from
// `split_keys`, the rest prefill rows taking their keys whole up to `whole` keys and sized from `prefill_keys` (a
// multiple of 16) past it. A step's launches (kernels/attn512.py) pass each their own item and group tables, and
// `order` (or null): the item each block takes, so that a launch can walk every group's first split before any
// group's second (the CTAs at once then read the same keys, as an unsplit launch's do); an item's partials stay at its
// own index.
//
// The kernel stages each tile's cos and sin rows in shared memory when rot <= 64, and looks up a tile's page once when
// the page size is a multiple of 16. Other shapes read both per key and give the same bits.
namespace {
template <bool SPLIT, bool FP8>
int attn512_launch(const void* q, int H, int G, int Hkv, const void* kv, int64_t page_stride, int page_size,
                   const void* page_table, int64_t pt_stride, const void* k_weight, const void* cos, const void* sin,
                   int rot, float scale, const void* rows, const void* items, int n_items, int split_keys, int n_dec,
                   int whole, int prefill_keys, const void* order, void* out, void* o_part, void* ml_part,
                   cudaStream_t stream) {
  const int smem = (int)sizeof(Smem);
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(attn512_kernel<SPLIT, FP8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    init = true;
  }
  return lmk_launch(attn512_kernel<SPLIT, FP8>, dim3(n_items), dim3(THREADS), smem, stream, (const bf16*)q, H, G, Hkv,
                    (const bf16*)kv, page_stride, page_size, (const int*)page_table, pt_stride, (const bf16*)k_weight,
                    (const bf16*)cos, (const bf16*)sin, rot, scale, (const Row*)rows, (const Item*)items,
                    (const int*)order, SPLIT ? split_keys : 0, SPLIT ? n_dec : 0, SPLIT ? whole : 0,
                    SPLIT ? prefill_keys : 0, (bf16*)out, SPLIT ? (float*)o_part : (float*)nullptr,
                    SPLIT ? (float*)ml_part : (float*)nullptr);
}
}  // namespace

LMK_API int lmk_attn512(const void* q, int H, int Hkv, const void* kv, int64_t page_stride, int page_size, int kv_fp8,
                        const void* page_table, int64_t pt_stride, const void* k_weight, const void* cos,
                        const void* sin, int rot, float scale, const void* rows, const void* items, int n_items,
                        const void* groups, int n_groups, int split, int split_keys, int n_dec, int whole,
                        int prefill_keys, const void* order, void* out, void* o_part, void* ml_part,
                        cudaStream_t stream) {
  if (n_items <= 0) return 0;
  const int G = H / Hkv;
  if (H % Hkv || ROWS % G || rot > HALF || rot % 2 || split < 0 || split > 1 || (split && n_groups <= 0) ||
      split_keys < 0 || split_keys % KT || (split_keys && (prefill_keys < KT || prefill_keys % KT || whole < 0)) ||
      kv_fp8 < 0 || kv_fp8 > 1)
    return -1;
  auto launch = !split ? (kv_fp8 ? attn512_launch<false, true> : attn512_launch<false, false>)
                       : (kv_fp8 ? attn512_launch<true, true> : attn512_launch<true, false>);
  const int st = launch(q, H, G, Hkv, kv, page_stride, page_size, page_table, pt_stride, k_weight, cos, sin, rot, scale,
                        rows, items, n_items, split_keys, n_dec, whole, prefill_keys, order, out, o_part, ml_part,
                        stream);
  if (st || !split) return st;
  return lmk_launch(attn512_merge_kernel, dim3(n_groups * ROWS), dim3(128), 0, stream, H, G, (const Row*)rows,
                    (const Item*)items, (const int2*)groups, (const float*)o_part, (const float*)ml_part, (bf16*)out);
}

// Returns the number of split-kernel CTAs (over BF16 pools, or FP8 ones with `kv_fp8` 1) that fit on one SM at once
// (shared memory and registers allow it), or −1. The wave-aware split fills one wave of them.
LMK_API int lmk_attn512_occupancy(int kv_fp8) {
  int n = 0;
  const int smem = (int)sizeof(Smem);
  auto kernel = kv_fp8 ? attn512_kernel<true, true> : attn512_kernel<true, false>;
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, THREADS, smem) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  return n;
}

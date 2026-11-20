// Paged sliding-window attention for Gemma 4's sliding layers, for reproducible answers (kernels/attn_window.py).
// Head dim 256, GQA, causal. A query at position p sees the keys [p − window_left, p]. K and V use FlashInfer's paged
// layout [pages, 2, page_size, Hkv, 256]. The kernels read it through the slot's window page table by absolute page
// (`wpage_table[slot, key / page_size]`, the runner's window table). The reference is
// `models.gemma4.backend.TorchSlidingAttention`.
//
// **Why our own.** FlashInfer's prefill kernel starts a CTA's keys at the window start of its tile's first query.
// It also picks the query tile and the split chunk size per batch. So a query's last bits depend on the rows beside
// it and on its offset in an MTP verify window. Here a query's keys come in KV tiles at absolute multiples of 16
// positions, and in splits at absolute multiples of `split_keys`. The result depends on the query's keys alone (the
// module docstring of kernels/attn_window.py).
//
// **Decode rows** (`attn_window_kernel`: the few queries of a decode or MTP verify row). A work item is (row, tile,
// KV head, j). The high 16 bits of the j word hold the split count n of the item's group. A tile is 16 query rows
// of one KV head: the flattened (position, group head) pairs [16 t, 16 t + 16), position-major. G is the number of
// query heads per KV head (2 on Gemma 4, so a tile spans 8 positions). The tile's keys are [lo, hi): lo is the
// window start of its first query, hi the causal end of its last.
// A group of one split (n = 1) takes all the keys, in KV tiles from lo rounded down to 16, and writes the tile's
// output. A group of n > 1 splits takes split s = lo / split_keys + j: the keys [s × split_keys, (s + 1) × split_keys)
// of [lo, hi), empty past hi. It writes fp32 partials (O unnormalised, m, l) that `attn_window_merge_kernel` sums in
// split order. Lengths come from the device row table. A CUDA graph replays them with new lengths (MTP: rows
// planned for an upper bound get their true lengths written on the device). An item whose row is −1 is padding: its
// CTA exits.
//
// **Decode CTA.** 4 warps. Warp w owns the dims [64 w, 64 w + 64) of the QK product and of the output. Per KV tile
// of 16 keys (K and V double-buffered with cp.async, 16-byte chunks XOR-swizzled by key), each warp computes its
// partial S = Q Kᵀ (mma.sync m16n8k16). The four partials are summed through shared memory in a fixed order, so
// every warp gets the same S. Then come the online softmax (fp32, exp2) and P V on the warp's output dims.
// A key outside the query's window, past the query or past the split is masked to −inf. A KV tile that none of a
// query's keys reach leaves its state as it was: a tile before the first key keeps m = −inf, l = 0, O = 0, and a
// tile after the last scales by exactly 1 and adds exactly 0. So the extra tiles a CTA visits do not change the
// query's bits.
//
// **Full attention's sizes** (`attn_window_kernel<true>`, Qwen3.8's full layers, no window): a decode query at absolute
// position p splits its keys at multiples of its own size, `split_keys` · 2^m (common.cuh `lmk_split_size`), so a tile
// whose rows straddle a size boundary walks the union of their ranges and masks each row to its own.
//
// **Prefill rows** (`attn_window_prefill_kernel`: the many queries of a prompt; whole, no split). A tile is 64 query
// rows (32 positions on Gemma 4). It reads the window's keys once for 4 times the queries. The decode CTA re-reads
// them every 16 rows, and the L2 bandwidth bounds it (a 139-token prompt tail took 96 µs a layer). The CTA has 8
// warps in two groups of 4. Warp w owns the query rows [16 (w % 4), + 16) and the KV tiles of absolute index
// ≡ w / 4 (mod 2). With one group, a single prompt's 80 CTAs would leave the latency of the warps exposed.
// Each warp computes the whole S of its rows over the 256 dims (Q in shared memory, A fragments read per k-step),
// their softmax, and P V into all 256 output dims (fp32, 128 registers a thread). Each group's K and V tiles of 16
// keys are single-buffered. Each loads while the other is in use. Its page is read a tile ahead. A thread keeps
// its row's sum as its own share and adds the quad's shares at the end. It rescales its output only when its max
// moves (otherwise the factor is exactly 1: the same bits). At the end the odd group's state merges into the even
// group's, in that order. The masking rule is the same, and a query's tiles fall in the same group whatever the
// batch, so its bits depend on its keys alone. An image-span query (`kend`) also sees the later keys of its span
// (keys [p − window_left, span end)) in the same pass. The tile's keys run to the largest key end of its queries.
//
// **The prefill split** (`attn_window_prefill_kernel<true>`, full attention only): a prefill query whose p + 1 keys pass
// `whole` splits them at multiples of its own size from `split_keys` (common.cuh), each split one item writing the
// tile's even and odd groups' merged, unnormalised state as fp32 partials, which `attn_window_merge_kernel<PROWS>`
// combines in split order. A query with at most `whole` keys is written by the unsplit kernel (`<false>`), whose code
// stays the unsplit path's; the split pass leaves it alone, even in a tile that straddles `whole`.
//
// Programmatic dependent launch (common.cuh): each kernel here waits for the kernel before it as its first
// statement, then lets the next one launch.
#include "common.cuh"

namespace {

constexpr int D = 256;
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr int ROWS = 16;       // query rows per tile
constexpr int KT = 16;         // keys per KV tile
constexpr int CHUNKS = D / 8;  // 16-byte chunks per key row
constexpr int TILE_BYTES = KT * D * 2;
constexpr float LOG2E = 1.4426950408889634f;
constexpr int MAX_MERGE_SPLITS = LMK_MAX_SPLITS;  // the merge's splits at most (common.cuh)

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
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ex2(float x) {  // 2^x (MUFU.EX2; −inf: +0)
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack(float x, float y) {
  __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&h);
}

struct Smem {
  alignas(128) unsigned char kv[2][2][TILE_BYTES];  // two stages of (K, V) tiles [KT keys][D]; Q aliases K of stage 0
  float s[WARPS][ROWS][KT];                         // partial scores
};

template <bool CLASSES>
__global__ void __launch_bounds__(THREADS, 2)
    attn_window_kernel(const bf16* __restrict__ q, int H, int G, int Hkv, const bf16* __restrict__ kv,
                       int64_t page_stride, int64_t plane_stride, int page_size, const int* __restrict__ wpt,
                       int64_t wpt_stride, int window_left, float scale, const Row* __restrict__ rows,
                       const Item* __restrict__ items, int split_keys, bf16* __restrict__ out,
                       float* __restrict__ o_part, float* __restrict__ ml_part) {
  lmk_pdl_wait();  // PDL: read nothing a predecessor writes before this (common.cuh)
  lmk_pdl_trigger();
  extern __shared__ unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const Item it = items[blockIdx.x];
  if (it.row < 0) return;  // padding in a graph's item table
  const int n_splits = max(1, it.split >> 16);  // the split count of this item's group
  const int j = it.split & 0xffff;
  const Row rw = rows[it.row];
  const int q0 = rw.kvlen - rw.nq;  // position of the row's first query
  // flattened row f = 16 t + i: query position f / G, head kvh * G + f % G
  const int f0 = it.tile * ROWS;
  const int p_first = q0 + f0 / G;
  const int p_last = q0 + min((f0 + ROWS - 1) / G, rw.nq - 1);
  const int lo = max(0, p_first - window_left);  // the tile's keys: [lo, hi)
  const int hi = min(rw.kvlen, p_last + 1);
  int k_begin = lo & ~(KT - 1), k_end = hi;  // KV tiles at absolute multiples of KT
  if (n_splits > 1) {  // split s: empty past hi (m = −inf, l = 0; the merge skips it)
    if constexpr (CLASSES) {  // full attention only (lo = 0): the union of the tile's rows' own split j (common.cuh)
      lmk_split_union(p_first, p_last, j, split_keys, 0, k_begin, k_end);
    } else {
      const int s = lo / split_keys + j;
      k_begin = max(k_begin, s * split_keys);
      k_end = min(hi, (s + 1) * split_keys);
    }
  }
  const int64_t item = blockIdx.x;

  // -- the Q tile: rows of 256 dims into the K buffer of stage 0 (swizzled), then A fragments per warp ------------
  for (int c = tid; c < ROWS * CHUNKS; c += THREADS) {
    const int r = c / CHUNKS, ch = c % CHUNKS;
    const int f = f0 + r, p = f / G;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (p < rw.nq) {
      const int head = it.kvh * G + f % G;
      v = *reinterpret_cast<const uint4*>(q + ((int64_t)(rw.qstart + p) * H + head) * D + ch * 8);
    }
    *reinterpret_cast<uint4*>(sm.kv[0][0] + swz(r, ch)) = v;
  }
  __syncthreads();
  uint32_t qa[4][4];  // one A fragment per k-step: dims 64 w + 16 jj
#pragma unroll
  for (int jj = 0; jj < 4; ++jj) {
    const int dim = 64 * warp + 16 * jj;
    const int r = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int ch = dim / 8 + (lane >> 4);
    ldsm_x4(smem_u32(sm.kv[0][0] + swz(r, ch)), qa[jj][0], qa[jj][1], qa[jj][2], qa[jj][3]);
  }
  __syncthreads();  // stage 0 now holds KV

  // this thread's rows g and g + 8: absolute position, window start, validity (full attention's sizes: its key end)
  int rpos[2], rlo[2], rhi[2];
  bool rok[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int f = f0 + g + 8 * h, p = f / G;
    rok[h] = p < rw.nq;
    rpos[h] = q0 + p;
    rlo[h] = max(0, rpos[h] - window_left);
    rhi[h] = k_end;
    if constexpr (CLASSES) {  // [rlo, rhi): the tile's keys cut at the row's causal end, or its own split's range
      rhi[h] = min(k_end, rpos[h] + 1);
      if (n_splits > 1 && rok[h]) {
        int a, b;
        lmk_split_range(rpos[h], j, split_keys, 0, a, b);
        rlo[h] = max(rlo[h], a);
        rhi[h] = min(rhi[h], b);
      }
      if (!rok[h] || rhi[h] < rlo[h]) rhi[h] = rlo[h];
    }
  }
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
  const float sl2 = scale * LOG2E;

  const int n_tiles = k_end > k_begin ? (k_end - k_begin + KT - 1) / KT : 0;
  const int* pt = wpt + (int64_t)rw.slot * wpt_stride;
  const bool page_tiles = page_size % KT == 0;  // a tile (first key a multiple of KT) lies in one page
  auto load_tile = [&](int ti, int stage) {
    const int kbase = k_begin + ti * KT;
    const int64_t pbase = page_tiles ? (int64_t)pt[kbase / page_size] * page_stride +
                                           (int64_t)(kbase % page_size) * Hkv * D : 0;  // the tile's page, read once
    for (int c = tid; c < 2 * KT * CHUNKS; c += THREADS) {
      const int plane = c / (KT * CHUNKS), r = (c / CHUNKS) % KT, ch = c % CHUNKS;
      const int key = kbase + r;
      const bool ok = key < k_end;
      const bf16* src = kv;
      if (ok) {
        const int64_t at = page_tiles ? pbase + (int64_t)r * Hkv * D
                                      : (int64_t)pt[key / page_size] * page_stride +
                                            (int64_t)(key % page_size) * Hkv * D;
        src = kv + at + plane * plane_stride + (int64_t)it.kvh * D + ch * 8;
      }
      cp_async16(smem_u32(sm.kv[stage][plane] + swz(r, ch)), src, ok);
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
    const unsigned char* kt = sm.kv[stage][0];
    const unsigned char* vt = sm.kv[stage][1];
    const int kbase = k_begin + ti * KT;
    // -- partial S over this warp's dims: B fragments come straight from the K tile ------------------------------
    float s[2][4];
#pragma unroll
    for (int n = 0; n < 2; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int dl = 64 * warp + 16 * jj;
      uint32_t u[4];  // u[0], u[1]: b0, b1 of keys 0-7 (dims dl + 2 t4, + 8); u[2], u[3]: the same for keys 8-15
      const int r = (lane & 7) + 8 * (lane >> 4);
      const int ch = dl / 8 + ((lane >> 3) & 1);
      ldsm_x4(smem_u32(kt + swz(r, ch)), u[0], u[1], u[2], u[3]);
      mma(s[0], qa[jj], u[0], u[1]);
      mma(s[1], qa[jj], u[2], u[3]);
    }
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      sm.s[warp][g][8 * n + 2 * t4] = s[n][0];
      sm.s[warp][g][8 * n + 2 * t4 + 1] = s[n][1];
      sm.s[warp][g + 8][8 * n + 2 * t4] = s[n][2];
      sm.s[warp][g + 8][8 * n + 2 * t4 + 1] = s[n][3];
    }
    __syncthreads();
    // -- the whole S (the same sum in every warp), masked ------------------------------------------------------
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
        bool ok;
        if constexpr (CLASSES) {
          ok = (unsigned)(key - rlo[h]) < (unsigned)(rhi[h] - rlo[h]);  // rlo <= key < rhi
        } else {
          ok = rok[h] && key < k_end && key <= rpos[h] && key >= rlo[h];
        }
        sc[n][e] = ok ? v * sl2 : -INFINITY;
      }
    }
    // -- online softmax: rows g (e 0, 1) and g + 8 (e 2, 3); the four lanes of a quad share a row -----------------
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
      alpha[h] = exp2f(m[h] - ref[h]);  // m = −inf: 0
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
    for (int n = 0; n < 8; ++n) {
      o[n][0] *= alpha[0];
      o[n][1] *= alpha[0];
      o[n][2] *= alpha[1];
      o[n][3] *= alpha[1];
    }
    // P as an A fragment (16 rows × 16 keys): the C layout of the two n-tiles is the A layout
    uint32_t pa[4] = {pack(p[0][0], p[0][1]), pack(p[0][2], p[0][3]), pack(p[1][0], p[1][1]),
                      pack(p[1][2], p[1][3])};
    // -- P V on this warp's output dims [64 w, 64 w + 64) ------------------------------------------------------
#pragma unroll
    for (int nb = 0; nb < 4; ++nb) {  // pairs of n-tiles of 8 dims
      const int dim = 64 * warp + 16 * nb;
      uint32_t v0, v1, v2, v3;
      // .trans: matrices (keys 0-7, dims dim..+8), (keys 8-15, dims dim..), (keys 0-7, dims +8), (keys 8-15, +8)
      const int r = (lane & 7) + 8 * ((lane >> 3) & 1);
      const int ch = dim / 8 + (lane >> 4);
      ldsm_x4_t(smem_u32(vt + swz(r, ch)), v0, v1, v2, v3);
      mma(o[2 * nb], pa, v0, v1);
      mma(o[2 * nb + 1], pa, v2, v3);
    }
    __syncthreads();  // the next tile rewrites the stage and the score buffer
  }

  // -- epilogue -------------------------------------------------------------------------------------------------
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = g + 8 * h;
    const int f = f0 + r, pidx = f / G;
    if (!rok[h]) continue;
    const int head = it.kvh * G + f % G;
    if (n_splits > 1) {
      float* op = o_part + (item * ROWS + r) * D;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int d = 64 * warp + 8 * n + 2 * t4;
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
      for (int n = 0; n < 8; ++n) {
        const int d = 64 * warp + 8 * n + 2 * t4;
        *reinterpret_cast<__nv_bfloat162*>(dst + d) = __floats2bfloat162_rn(o[n][2 * h] * inv, o[n][2 * h + 1] * inv);
      }
    }
  }
}

constexpr int PROWS = WARPS * ROWS;  // a prefill tile's query rows
constexpr int PWARPS = 2 * WARPS;    // two groups of 4 warps: even and odd KV tiles
constexpr int PTHREADS = PWARPS * 32;
constexpr int MO_STRIDE = D + 8;     // a row of the merge's fp32 output (padded: at most 2-way bank conflicts)

struct PSmem {
  union {
    struct {
      alignas(128) unsigned char q[PROWS * D * 2];  // [64 rows][D], swizzled
      alignas(128) unsigned char kv[2][2][TILE_BYTES];  // [group][K, V]: [KT keys][D]
    } loop;
    struct {
      float o[PROWS * MO_STRIDE];  // the odd group's output, unnormalised
      float m[PROWS], l[PROWS][4];  // the max of its rows and the shares of their sums per thread
    } merge;
  };
  int hi[PWARPS];
};

// A prefill row's tile of 64 query rows (module comment above): item (row, tile of 64, KV head, unused; SPLIT: split
// j | n << 16). `kend` (or null): int32 per token of q, the span end of an image-span query (0 elsewhere). The two
// groups run in step. SPLIT (full attention, no spans): the tile's queries past `whole` keys, in their split j, into
// the partials o_part [items, 64, D] and ml_part [items, 64, 2].
template <bool SPLIT>
__global__ void __launch_bounds__(PTHREADS, 1)
    attn_window_prefill_kernel(const bf16* __restrict__ q, int H, int G, int Hkv, const bf16* __restrict__ kv,
                               int64_t page_stride, int64_t plane_stride, int page_size, const int* __restrict__ wpt,
                               int64_t wpt_stride, int window_left, float scale, const Row* __restrict__ rows,
                               const Item* __restrict__ items, const int* __restrict__ kend, bf16* __restrict__ out,
                               int whole, int split_keys, const int* __restrict__ order, float* __restrict__ o_part,
                               float* __restrict__ ml_part) {
  lmk_pdl_wait();  // PDL: read nothing a predecessor writes before this (common.cuh)
  lmk_pdl_trigger();
  extern __shared__ unsigned char smem_raw[];
  PSmem& sm = *reinterpret_cast<PSmem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int grp = warp >> 2, wq = warp & 3, gtid = tid & 127;
  [[maybe_unused]] int idx = blockIdx.x;  // SPLIT: the item, from the launch order (as attn512's)
  if constexpr (SPLIT) {
    if (order) idx = order[blockIdx.x];
  }
  const Item it = items[SPLIT ? idx : blockIdx.x];
  if (it.row < 0) return;
  const Row rw = rows[it.row];
  const int q0 = rw.kvlen - rw.nq;
  const int f0 = it.tile * PROWS;  // flattened f -> query position f / G, head kvh * G + f % G
  [[maybe_unused]] const int j = it.split & 0xffff;  // SPLIT: this item's split
  [[maybe_unused]] int s_lo = 0, s_hi = 0;  // SPLIT: the union of the split queries' ranges in split j
  if constexpr (SPLIT) {
    const int p0 = max(q0 + f0 / G, whole), p1 = q0 + min((f0 + PROWS - 1) / G, rw.nq - 1);
    if (p0 <= p1) lmk_split_union(p0, p1, j, split_keys, whole, s_lo, s_hi);
  }
  for (int c = tid; c < PROWS * CHUNKS; c += PTHREADS) {  // the Q tile is the first cp.async group
    const int r = c / CHUNKS, ch = c % CHUNKS;
    const int f = f0 + r, p = f / G;
    const bool ok = p < rw.nq;
    const bf16* src = ok ? q + ((int64_t)(rw.qstart + p) * H + it.kvh * G + f % G) * D + ch * 8 : q;
    cp_async16(smem_u32(sm.loop.q + swz(r, ch)), src, ok);
  }
  cp_commit();
  // this thread's rows 16 wq + g and + 8: first key and key end (for a span query, its span's end)
  int rlo[2], rhi[2], hi = 0;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int f = f0 + 16 * wq + g + 8 * h, p = f / G;
    rlo[h] = max(0, q0 + p - window_left);
    rhi[h] = p < rw.nq ? max(q0 + p + 1, kend ? kend[rw.qstart + p] : 0) : 0;  // 0: no keys (past the row's queries)
    if constexpr (SPLIT) {  // its keys in split j; none for a query taken whole (the unsplit kernel writes it)
      if (p < rw.nq && q0 + p + 1 > whole) {
        int a, b;
        lmk_split_range(q0 + p, j, split_keys, whole, a, b);
        rlo[h] = max(rlo[h], a);
        rhi[h] = min(rhi[h], b);
      } else {
        rhi[h] = 0;
      }
    }
    hi = max(hi, rhi[h]);
  }
#pragma unroll
  for (int x = 1; x < 32; x <<= 1) hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, x));
  if (lane == 0) sm.hi[warp] = hi;
  __syncthreads();
  hi = max(max(sm.hi[0], sm.hi[1]), max(sm.hi[2], sm.hi[3]));  // the tile's keys: [lo, hi)
  int k_begin = max(0, q0 + f0 / G - window_left) & ~(KT - 1);
  if constexpr (SPLIT) {
    k_begin = s_lo;
    hi = s_hi;
  }
  const int n_tiles = hi > k_begin ? (hi - k_begin + KT - 1) / KT : 0;
  // this group's tiles: ti = first, first + 2, ... (absolute index k_begin / KT + ti ≡ grp mod 2)
  const int first = ((k_begin / KT) & 1) == grp ? 0 : 1;
  const int n_mine = n_tiles > first ? (n_tiles - first + 1) / 2 : 0;
  const int n_iter = (n_tiles + 1) / 2;
  const int* pt = wpt + (int64_t)rw.slot * wpt_stride;
  // A KV tile lies in one page (the launcher checks page_size % KT == 0). This returns its offset in the pool. The
  // page is read a tile ahead of the loads.
  auto tile_base = [&](int i) -> int64_t {
    const int kbase = k_begin + (first + 2 * i) * KT;
    return i < n_mine ? (int64_t)pt[kbase / page_size] * page_stride + (int64_t)(kbase % page_size) * Hkv * D +
                            (int64_t)it.kvh * D
                      : 0;
  };
  unsigned char* kt = sm.loop.kv[grp][0];
  unsigned char* vt = sm.loop.kv[grp][1];
  auto load_tile = [&](int i, int64_t base, int plane, unsigned char* dst) {  // one plane of a tile, for one group
    if (i < n_mine) {
      const int kbase = k_begin + (first + 2 * i) * KT;
      for (int c = gtid; c < KT * CHUNKS; c += THREADS) {
        const int r = c / CHUNKS, ch = c % CHUNKS;
        const bool ok = kbase + r < hi;
        const bf16* src = ok ? kv + base + plane * plane_stride + (int64_t)r * Hkv * D + ch * 8 : kv;
        cp_async16(smem_u32(dst + swz(r, ch)), src, ok);
      }
    }
    cp_commit();  // empty past the group's tiles: every thread commits the same number of groups
  };
  float o[D / 8][4];  // n-tiles of 8 dims
#pragma unroll
  for (int n = 0; n < D / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f}, al[2];
  const float sl2 = scale * LOG2E;
  int64_t base = tile_base(0);
  load_tile(0, base, 0, kt);
  load_tile(0, base, 1, vt);
  base = tile_base(1);  // the next tile's offset
  // In flight at the top of step i: (Q,) K(i), V(i). K(i + 1) loads during the softmax and P V, V(i + 1) during the
  // next S.
  for (int i = 0; i < n_iter; ++i) {
    const bool act = i < n_mine;  // warp-uniform
    const int kbase = k_begin + (first + 2 * i) * KT;
    cp_wait<1>();  // wait for K(i) (and Q)
    __syncthreads();
    float s[2][4];
#pragma unroll
    for (int n = 0; n < 2; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
    if (act) {
#pragma unroll
      for (int jj = 0; jj < D / 16; ++jj) {
        uint32_t a[4], u[4];
        ldsm_x4(smem_u32(sm.loop.q + swz(16 * wq + (lane & 7) + 8 * ((lane >> 3) & 1), 2 * jj + (lane >> 4))), a[0],
                a[1], a[2], a[3]);
        ldsm_x4(smem_u32(kt + swz((lane & 7) + 8 * (lane >> 4), 2 * jj + ((lane >> 3) & 1))), u[0], u[1], u[2], u[3]);
        mma(s[0], a, u[0], u[1]);
        mma(s[1], a, u[2], u[3]);
      }
    }
    __syncthreads();  // every warp has read K(i)
    load_tile(i + 1, base, 0, kt);
    const int64_t base2 = tile_base(i + 2);
    uint32_t pa[4] = {0u, 0u, 0u, 0u};
    if (act) {
      float sc[2][4];
#pragma unroll
      for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int key = kbase + 8 * n + 2 * t4 + (e & 1), h = e >> 1;
          sc[n][e] = key >= rlo[h] && key < rhi[h] ? s[n][e] * sl2 : -INFINITY;
        }
      }
      float mx[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float v = fmaxf(fmaxf(sc[0][2 * h], sc[0][2 * h + 1]), fmaxf(sc[1][2 * h], sc[1][2 * h + 1]));
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
        mx[h] = fmaxf(m[h], v);
      }
      // A row whose max moved rescales its sum and output. Otherwise the factor is exactly 1 and the rescale is
      // skipped (the same bits). The sum is this thread's share of its row's keys; the epilogue adds the quad's shares.
      float ref[2];
      bool moved = false;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        ref[h] = mx[h] == -INFINITY ? 0.0f : mx[h];
        al[h] = 1.0f;
        if (mx[h] != m[h]) {
          al[h] = ex2(m[h] - ref[h]);  // m = −inf: 0
          l[h] *= al[h];
          moved = true;
        }
        m[h] = mx[h];
      }
      if (__any_sync(0xffffffffu, moved)) {
#pragma unroll
        for (int n = 0; n < D / 8; ++n) {
          o[n][0] *= al[0];
          o[n][1] *= al[0];
          o[n][2] *= al[1];
          o[n][3] *= al[1];
        }
      }
      float p[2][4];
#pragma unroll
      for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          p[n][e] = ex2(sc[n][e] - ref[e >> 1]);
          l[e >> 1] += p[n][e];
        }
      }
      pa[0] = pack(p[0][0], p[0][1]);
      pa[1] = pack(p[0][2], p[0][3]);
      pa[2] = pack(p[1][0], p[1][1]);
      pa[3] = pack(p[1][2], p[1][3]);
    }
    cp_wait<1>();  // wait for V(i); K(i + 1) may still be in flight
    __syncthreads();
    if (act) {
#pragma unroll
      for (int nb = 0; nb < D / 16; ++nb) {  // pairs of n-tiles of 8 dims, as in `attn_window_kernel`
        uint32_t v0, v1, v2, v3;
        ldsm_x4_t(smem_u32(vt + swz((lane & 7) + 8 * ((lane >> 3) & 1), 2 * nb + (lane >> 4))), v0, v1, v2, v3);
        mma(o[2 * nb], pa, v0, v1);
        mma(o[2 * nb + 1], pa, v2, v3);
      }
    }
    __syncthreads();  // every warp has read V(i)
    load_tile(i + 1, base, 1, vt);
    base = base2;
  }
  cp_wait<0>();
  __syncthreads();  // the loop's buffers become the merge's
  if (grp == 1) {  // the odd tiles' state, for the even group's warps on the same rows
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = 16 * wq + g + 8 * h;
#pragma unroll
      for (int n = 0; n < D / 8; ++n)
        *reinterpret_cast<float2*>(sm.merge.o + r * MO_STRIDE + 8 * n + 2 * t4) =
            make_float2(o[n][2 * h], o[n][2 * h + 1]);
      if (t4 == 0) sm.merge.m[r] = m[h];
      sm.merge.l[r][t4] = l[h];
    }
  }
  __syncthreads();
  if (grp == 1) return;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = 16 * wq + g + 8 * h;
    const int f = f0 + r, pidx = f / G;
    const float m1 = sm.merge.m[r];
    const float mx = fmaxf(m[h], m1);
    const float ref = mx == -INFINITY ? 0.0f : mx;
    const float s0 = ex2(m[h] - ref), s1 = ex2(m1 - ref);
    float lt = l[h] * s0 + sm.merge.l[r][t4] * s1;
    lt += __shfl_xor_sync(0xffffffffu, lt, 1);  // the quad's shares, in a fixed order
    lt += __shfl_xor_sync(0xffffffffu, lt, 2);
    if (pidx >= rw.nq) continue;
    if constexpr (SPLIT) {  // a split query's state, unnormalised, for the merge; a query taken whole: nothing
      if (q0 + pidx + 1 <= whole) continue;
      const int64_t item = idx;  // its partials' slot
      float* op = o_part + (item * PROWS + r) * D;
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        const int d = 8 * n + 2 * t4;
        const float2 v = *reinterpret_cast<const float2*>(sm.merge.o + r * MO_STRIDE + d);
        *reinterpret_cast<float2*>(op + d) = make_float2(o[n][2 * h] * s0 + v.x * s1, o[n][2 * h + 1] * s0 + v.y * s1);
      }
      if (t4 == 0) {
        ml_part[(item * PROWS + r) * 2] = mx;
        ml_part[(item * PROWS + r) * 2 + 1] = lt;
      }
      continue;
    }
    const float inv = lt > 0.0f ? 1.0f / lt : 0.0f;
    bf16* dst = out + ((int64_t)(rw.qstart + pidx) * H + it.kvh * G + f % G) * D;
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      const int d = 8 * n + 2 * t4;
      const float2 v = *reinterpret_cast<const float2*>(sm.merge.o + r * MO_STRIDE + d);
      *reinterpret_cast<__nv_bfloat162*>(dst + d) =
          __floats2bfloat162_rn((o[n][2 * h] * s0 + v.x * s1) * inv, (o[n][2 * h + 1] * s0 + v.y * s1) * inv);
    }
  }
}

// Combine the splits of each (row, tile, KV head) group: the items [first, first + n) given by the group table
// (first, n), in split order. One block per (group, query row of the tile), 64 threads × 4 dims. One thread turns the
// (m, l) of the splits into weights and their sum, in split order. This fixes the sum's order by the absolute
// positions of the splits, whichever lanes they would use. Then every thread sums its dims over the splits. A group
// of one split (its CTA wrote it), a padding group and a row past the row's queries exit. A split that none of the
// row's keys reached (l = 0) weighs 0.
// TR: the tile's query rows (ROWS for decode rows, PROWS for the prefill split, whose queries with at most `whole`
// keys the unsplit kernel wrote).
template <int TR>
__global__ void attn_window_merge_kernel(int H, int G, const Row* __restrict__ rows, const Item* __restrict__ items,
                                         const int2* __restrict__ groups, const float* __restrict__ o_part,
                                         const float* __restrict__ ml_part, bf16* __restrict__ out, int whole) {
  lmk_pdl_wait();  // PDL: read nothing a predecessor writes before this (common.cuh)
  lmk_pdl_trigger();
  __shared__ float wts[MAX_MERGE_SPLITS];
  __shared__ float inv_l;
  const int2 grp = groups[blockIdx.x / TR];
  const int n_splits = grp.y;
  if (grp.x < 0 || n_splits <= 1) return;
  const int r = blockIdx.x % TR;
  const int64_t i0 = grp.x;
  const Item it = items[i0];
  const Row rw = rows[it.row];
  const int f = it.tile * TR + r, p = f / G;
  if (p >= rw.nq) return;
  if constexpr (TR != ROWS) {
    if (rw.kvlen - rw.nq + p + 1 <= whole) return;
  }
  if (threadIdx.x == 0) {
    float mx = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
      const float* ml = ml_part + ((i0 + s) * TR + r) * 2;
      if (ml[1] > 0.0f) mx = fmaxf(mx, ml[0]);
    }
    float L = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
      const float* ml = ml_part + ((i0 + s) * TR + r) * 2;
      const float w = ml[1] > 0.0f ? exp2f(ml[0] - mx) : 0.0f;
      wts[s] = w;
      L += w * ml[1];
    }
    inv_l = L > 0.0f ? 1.0f / L : 0.0f;
  }
  __syncthreads();
  const int d = threadIdx.x * 4;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int s = 0; s < n_splits; ++s) {
    const float w = wts[s];
    if (w == 0.0f) continue;
    const float4 v = *reinterpret_cast<const float4*>(o_part + ((i0 + s) * TR + r) * D + d);
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

// Decode rows. rows: int32 [n_rows, 4] (slot, first token in q/out, n_query, KV length after the step). items: int32
// [n_items, 4] (row, tile, KV head, j | n << 16), grouped by (row, tile, KV head), the n splits of a group
// consecutive (row −1: padding). groups: int32 [n_groups, 2] (first item, n) per group (first −1: padding).
// q, out: [T, H, 256]. kv: the pages of a sliding layer, [pages, 2, page_size, Hkv, 256] (`page_stride` elements a
// page, `plane_stride` from K to V). wpage_table: int32 [slots, pages of a row] by absolute page. o_part
// [n_items, 16, 256] fp32 and ml_part [n_items, 16, 2] fp32 hold the partials of groups of more than one split
// (n <= 32) for the merge. `split_keys`: keys per split, a multiple of 16.
LMK_API int lmk_attn_window(const void* q, int H, int Hkv, const void* kv, int64_t page_stride, int64_t plane_stride,
                            int page_size, const void* wpage_table, int64_t wpt_stride, int window_left, float scale,
                            const void* rows, const void* items, int n_items, const void* groups, int n_groups,
                            int split_keys, void* out, void* o_part, void* ml_part, cudaStream_t stream) {
  if (n_items <= 0) return 0;
  const int G = H / Hkv;
  if (H % Hkv || ROWS % G || window_left < 0 || n_groups <= 0 || split_keys <= 0 || split_keys % KT) return -1;
  const int smem = (int)sizeof(Smem);
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(attn_window_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    init = true;
  }
  const int st = lmk_launch(attn_window_kernel<false>, dim3(n_items), dim3(THREADS), smem, stream, (const bf16*)q, H, G,
                            Hkv, (const bf16*)kv, page_stride, plane_stride, page_size, (const int*)wpage_table,
                            wpt_stride, window_left, scale, (const Row*)rows, (const Item*)items, split_keys,
                            (bf16*)out, (float*)o_part, (float*)ml_part);
  if (st) return st;
  return lmk_launch(attn_window_merge_kernel<ROWS>, dim3(n_groups * ROWS), dim3(D / 4), 0, stream, H, G,
                    (const Row*)rows, (const Item*)items, (const int2*)groups, (const float*)o_part,
                    (const float*)ml_part, (bf16*)out, 0);
}

// Prefill rows (module comment above). items: int32 [n_items, 4] (row, tile of 64 query rows, KV head, 0). rows, q,
// out, kv and the window page table: as above. kend: null, or int32 [tokens of q] with the span end of each
// image-span query (0 elsewhere).
LMK_API int lmk_attn_window_prefill(const void* q, int H, int Hkv, const void* kv, int64_t page_stride,
                                    int64_t plane_stride, int page_size, const void* wpage_table, int64_t wpt_stride,
                                    int window_left, float scale, const void* rows, const void* items, int n_items,
                                    const void* kend, void* out, cudaStream_t stream) {
  if (n_items <= 0) return 0;
  const int G = H / Hkv;
  if (H % Hkv || ROWS % G || window_left < 0 || page_size % KT) return -1;
  const int smem = (int)sizeof(PSmem);
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(attn_window_prefill_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    init = true;
  }
  return lmk_launch(attn_window_prefill_kernel<false>, dim3(n_items), dim3(PTHREADS), smem, stream, (const bf16*)q, H,
                    G, Hkv, (const bf16*)kv, page_stride, plane_stride, page_size, (const int*)wpage_table,
                    wpt_stride, window_left, scale, (const Row*)rows, (const Item*)items, (const int*)kend,
                    (bf16*)out, 0, 0, (const int*)nullptr, (float*)nullptr, (float*)nullptr);
}

// Full causal attention on the kernels above (Qwen3.8's full layers under batch invariance; kernels/attn_window.py
// `FullAttention`). There is no window: window_left exceeds every position, so a query at p sees the keys [0, p] and
// a tile's keys start at 0. Any GQA group G works. A tile's 16 (decode) or 64 (prefill) rows are (position, head)
// pairs in that order, whatever divides them. With G = 6 the heads of a position may straddle two tiles. Each row is
// its own query, and its bits depend on its keys alone. The page table is the slot's own (by absolute page). A
// decode query's splits come from its own position (common.cuh: sizes from `split_keys`, at most the merge's 32), its
// group's count n from the most its rows may take (kernels/attn_window.py). The arguments are those of
// `lmk_attn_window` and `lmk_attn_window_prefill`, without the window. `lmk_attn_full_prefill_split`: the prefill
// split's items (row, 64-row tile, KV head, j | n << 16) and their groups, for the queries past `whole` keys (sizes
// from `split_keys`), into o_part [n_items, 64, 256] fp32 and ml_part [n_items, 64, 2], merged in split order.
constexpr int kNoWindow = 0x7fffffff;  // window_left: p − kNoWindow is below 0 for every position p

LMK_API int lmk_attn_full(const void* q, int H, int Hkv, const void* kv, int64_t page_stride, int64_t plane_stride,
                          int page_size, const void* page_table, int64_t pt_stride, float scale, const void* rows,
                          const void* items, int n_items, const void* groups, int n_groups, int split_keys, void* out,
                          void* o_part, void* ml_part, cudaStream_t stream) {
  if (n_items <= 0) return 0;
  if (H % Hkv || n_groups <= 0 || split_keys <= 0 || split_keys % KT) return -1;
  const int G = H / Hkv, smem = (int)sizeof(Smem);
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(attn_window_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    init = true;
  }
  const int st = lmk_launch(attn_window_kernel<true>, dim3(n_items), dim3(THREADS), smem, stream, (const bf16*)q, H, G,
                            Hkv, (const bf16*)kv, page_stride, plane_stride, page_size, (const int*)page_table,
                            pt_stride, kNoWindow, scale, (const Row*)rows, (const Item*)items, split_keys,
                            (bf16*)out, (float*)o_part, (float*)ml_part);
  if (st) return st;
  return lmk_launch(attn_window_merge_kernel<ROWS>, dim3(n_groups * ROWS), dim3(D / 4), 0, stream, H, G,
                    (const Row*)rows, (const Item*)items, (const int2*)groups, (const float*)o_part,
                    (const float*)ml_part, (bf16*)out, 0);
}

LMK_API int lmk_attn_full_prefill(const void* q, int H, int Hkv, const void* kv, int64_t page_stride,
                                  int64_t plane_stride, int page_size, const void* page_table, int64_t pt_stride,
                                  float scale, const void* rows, const void* items, int n_items, void* out,
                                  cudaStream_t stream) {
  if (n_items <= 0) return 0;
  if (H % Hkv || page_size % KT) return -1;
  const int G = H / Hkv, smem = (int)sizeof(PSmem);
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(attn_window_prefill_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    init = true;
  }
  return lmk_launch(attn_window_prefill_kernel<false>, dim3(n_items), dim3(PTHREADS), smem, stream, (const bf16*)q, H,
                    G, Hkv, (const bf16*)kv, page_stride, plane_stride, page_size, (const int*)page_table, pt_stride,
                    kNoWindow, scale, (const Row*)rows, (const Item*)items, (const int*)nullptr, (bf16*)out, 0, 0,
                    (const int*)nullptr, (float*)nullptr, (float*)nullptr);
}

LMK_API int lmk_attn_full_prefill_split(const void* q, int H, int Hkv, const void* kv, int64_t page_stride,
                                        int64_t plane_stride, int page_size, const void* page_table, int64_t pt_stride,
                                        float scale, const void* rows, const void* items, int n_items,
                                        const void* groups, int n_groups, int whole, int split_keys,
                                        const void* order, void* out, void* o_part, void* ml_part,
                                        cudaStream_t stream) {
  if (n_items <= 0) return 0;
  if (H % Hkv || page_size % KT || n_groups <= 0 || whole < 0 || split_keys < KT || split_keys % KT) return -1;
  const int G = H / Hkv, smem = (int)sizeof(PSmem);
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(attn_window_prefill_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    init = true;
  }
  const int st = lmk_launch(attn_window_prefill_kernel<true>, dim3(n_items), dim3(PTHREADS), smem, stream,
                            (const bf16*)q, H, G, Hkv, (const bf16*)kv, page_stride, plane_stride, page_size,
                            (const int*)page_table, pt_stride, kNoWindow, scale, (const Row*)rows, (const Item*)items,
                            (const int*)nullptr, (bf16*)out, whole, split_keys, (const int*)order, (float*)o_part,
                            (float*)ml_part);
  if (st) return st;
  return lmk_launch(attn_window_merge_kernel<PROWS>, dim3(n_groups * PROWS), dim3(D / 4), 0, stream, H, G,
                    (const Row*)rows, (const Item*)items, (const int2*)groups, (const float*)o_part,
                    (const float*)ml_part, (bf16*)out, whole);
}

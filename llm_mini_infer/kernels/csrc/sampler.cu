// The counter-based sampler as one kernel (docs/architecture.md, "Sampling"). One CTA per sample row reproduces the
// torch reference (runner/cuda/sampler.py) token for token:
//
//   greedy (temperature < 1e-5): argmax of the logits, lowest id on ties.
//   otherwise x = logits / T (fp32). The kept set is the intersection of
//     min-p:  x >= max(x) + log(min_p)                                              (fp32, as torch)
//     top-k:  rank < k in the order (x desc, id asc) among the min-p survivors
//     top-p:  S <= top_p * Z, with e = exp(x - max(x)) in fp32. Z sums e in fp64 over the top-k set.
//             S sums e in fp64 over the tokens before each one in that order (exclusive).
//   The token is the lowest-id argmax of x + g over the kept set. g = -log(-log(u)) is the Gumbel noise of
//   (seed, position, id): u comes from splitmix64(key + id * phi), key = splitmix64(seed ^ splitmix64(position)).
//
// With `ids` (int32 [V], ascending), logit column v stands for token ids[v]. This is a reduced vocabulary (the MTP
// drafter's head). The noise and the written token are those of ids[v]. Ties still go to the lowest id, because the
// ids ascend with the column.
//
// With `pad` != 0, the ascending ids come first and entries < 0 follow. Two heads use it: the grammar-sparse head over
// the union of the step's masks, and the drafter's head over the grammar's universe. Both pad to a fixed width for the
// graphs. A padding column is never a candidate, as if its logit were -inf. It is not in the max, the min-p count, the
// top-k / top-p order and sums, or the Gumbel-max. Without `pad`, every id is >= 0. Without a mask, the candidate test
// then loads nothing more (the drafter's frequency-ranked head).
//
// With a grammar mask (`mask` [*, mask_ld] int32 words, `mask_row[r]` the row's mask, or -1 for no mask), only tokens
// whose bit is set are candidates (bit t % 32 of word t / 32 for token t). Everything above runs over them alone, as if
// the other logits were -inf. With `ids`, the test uses the column's token, ids[v]. The non-finite check still reads
// every column, padding included (a NaN hidden state makes every column NaN). A row with no allowed column (the mask
// excludes all of them, or all are padding) gets token 0 and status bit 1 (MASK_EMPTY).
//
// A row with a non-finite logit gets token 0 and status bit 0 (NONFINITE), which wins over MASK_EMPTY. Only the
// summation order of S and Z differs from torch. S and Z sum in fp64. A Z over more than kCap tokens sums per-thread
// compensated fp32 first, then fp64. A token can differ only when S lands within about 1e-13 of top_p * Z.
//
// Candidates come in batches of up to kCap tokens in that order. A radix select finds each batch's last token. The
// batch is gathered (ties at its boundary by id) and sorted in shared memory. The running top-p sum continues across
// batches until it passes top_p * Z or the top-k set ends. A top-k of at most kCap, the common case, is one batch.
//
// Penalties and the logit bias (runner/cuda/penalties.py; with `terms`, else a kernel without them). A row whose slot
// has terms reads each logit l of token t as vLLM's v2 runner adjusts it, in fp32: y = l + bias[t]; when t is in the
// prompt or the output so far, y = y > 0 ? y * (1 / rep) : y * rep; then y -= freq * count[t] and y -= pres when
// count[t] > 0; y clamped to the finite floats. Everything above runs on y (greedy included), and x = y / T is clamped
// too. The non-finite check still reads l. The slot's word of t (`cnt`): bit 31 in the prompt, bit 30 biased (its value
// by a binary search of the slot's ascending bias ids), bits 0-29 its output count; a verify row (`w_tok`: the token
// of its sample token is -1 - w at window position w) adds the w drafts before it (`drafts`), each once. With `commit`
// (no MTP: the sample is committed), a penalised row's token counts. Rows without terms run the code above unchanged.
//
// Programmatic dependent launch (common.cuh): the sampler waits for the previous kernel as its first statement, then
// lets the next kernel launch. The launcher adds the attribute while the library's flag is on (the runner's
// `dev.runner.cuda.pdl`).
#include <cfloat>

#include "common.cuh"

namespace {

constexpr int kT = 1024;  // threads per row
constexpr int kNW = kT / 32;
constexpr int kCap = 2048;  // candidates per batch
constexpr int kEqCap = 1024;  // boundary ties ranked in shared memory (more: an ordered pass over the row)
constexpr float kGreedyEps = 1e-5f;
constexpr uint64_t kPhi = 0x9E3779B97F4A7C15ull;
constexpr int kMaxDrafts = 8;  // a verify window's drafts at most (MTP's k)
constexpr int32_t kBiased = 1 << 30;  // a `cnt` word's bias bit (runner/cuda/penalties.py)
constexpr int32_t kCount = (1 << 30) - 1;  // its output count

__device__ __forceinline__ float clampf(float y) { return fminf(fmaxf(y, -FLT_MAX), FLT_MAX); }

// The terms' arguments (the file header; runner/cuda/penalties.py's tables).
struct TermArgs {
  const int32_t* terms;  // per slot: bit 0 penalties, bit 1 a bias; null: no terms
  const float* rep;
  const float* pres;
  const float* freq;
  int32_t* cnt;  // [slots, cnt_ld]
  int64_t cnt_ld;
  const int32_t* bias_n;
  const int32_t* bias_id;  // [slots, bias_ld], ascending
  const float* bias_val;
  int64_t bias_ld;
  const int64_t* drafts;  // [slots, k] (MTP), or null
  int k;
  const int64_t* w_tok;  // the static block's tokens (MTP), or null
  const int64_t* w_idx;  // each row's sample token
  int commit;
};

// One row's terms (`TermArgs` at its slot).
struct Adj {
  const int32_t* e;  // the slot's words
  const int32_t* bid;
  const float* bval;
  int bn;
  bool pen;
  float rep, inv, pres, freq;
  int nw;
  int32_t win[kMaxDrafts];

  __device__ __forceinline__ float bias(int32_t t) const {
    int lo = 0, hi = bn - 1;  // t is among the ids (its bias bit)
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (bid[mid] < t) lo = mid + 1; else hi = mid;
    }
    return bval[lo];
  }
  // l as the terms adjust it for token t (t < 0: padding, never a candidate)
  __device__ __forceinline__ float apply(float l, int32_t t) const {
    if (t < 0) return l;
    const int32_t w = e[t];
    float y = l;
    if (w & kBiased) y = fadd(y, bias(t));
    if (pen) {
      int c = w & kCount;
      for (int i = 0; i < nw; ++i) c += win[i] == t;
      if (w < 0 || c > 0) y = y > 0.0f ? fmul(y, inv) : fmul(y, rep);
      if (c > 0) y = fsub(fsub(y, fmul(freq, (float)c)), pres);
    }
    return clampf(y);
  }
};

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += kPhi;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__device__ __forceinline__ float gumbel(uint64_t key, uint32_t id) {
  const uint64_t h = splitmix64(key + (uint64_t)id * kPhi);
  // 24 bits: u = ((h >> 40) + 0.5) / 2^24, clamped to 1 - 2^-24 (the largest float below 1). The top value rounds to
  // 1 in fp32 and would give noise +inf. It draws 16.64 instead, one token in 2^24 (the next value draws 15.94).
  const float u = fminf(fmul(fadd((float)(h >> 40), 0.5f), 1.0f / 16777216.0f), 0x1.fffffep-1f);
  return -logf(-logf(u));
}

// Monotone map of a float to an unsigned key (larger float, larger key).
__device__ __forceinline__ uint32_t okey(float x) {
  const uint32_t b = __float_as_uint(x);
  return b ^ ((b & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}

struct Shared {
  uint64_t cand[kCap];  // (~key << 32 | id): ascending = (x desc, id asc)
  uint32_t eq[kEqCap];
  uint32_t hist[256];
  double dred[kNW];
  double dscan[kNW];
  double dtotal;
  float fred[kNW];
  int ired[kNW];
  uint32_t ured[kNW];
  int n_cand, n_eq;
  uint32_t sel_prefix;
  int sel_rem;
  uint32_t sel_count_eq;
};

// Whether column v may be sampled: not padding, and its token allowed by the row's grammar mask `mk` (null: every
// token). `tk` holds each column's token id when the test needs one (padded ids, or ids under a mask). Otherwise `tk`
// is null and column v is tested as token v with no extra load (a full vocabulary, or the drafter's head).
__device__ __forceinline__ bool col_allowed(const int32_t* mk, const int32_t* tk, int v) {
  if (tk == nullptr) return mk == nullptr || ((mk[v >> 5] >> (v & 31)) & 1);
  const int32_t t = tk[v];
  return t >= 0 && (mk == nullptr || ((mk[t >> 5] >> (t & 31)) & 1));
}

template <bool kTerms>
struct Row {
  const float* l;
  int V;
  float T;
  float thr;  // min-p threshold on x (-inf: off)
  bool after;  // only tokens strictly after (a_key, a_id) in the order count
  uint32_t a_key, a_id;
  const int32_t* mk;  // the row's grammar mask, or null (every token allowed)
  const int32_t* tk;  // the columns' token ids for the allowed test, or null (col_allowed)
  const int32_t* ids;  // the columns' token ids (null: column v is token v), for the terms
  Adj a;  // the terms (kTerms)

  __device__ __forceinline__ float y(int v) const {
    if (kTerms) return a.apply(l[v], ids != nullptr ? ids[v] : v);
    return l[v];
  }
  __device__ __forceinline__ float x(int v) const {
    if (kTerms) return clampf(fdiv(y(v), T));
    return fdiv(l[v], T);
  }
  __device__ __forceinline__ bool allowed(int v) const { return col_allowed(mk, tk, v); }
  // Whether token v (with key k) is an allowed min-p survivor after the previous batch's last token.
  __device__ __forceinline__ bool live(uint32_t k, float xv, uint32_t v) const {
    if (!(xv >= thr) || !allowed(v)) return false;
    return !after || k < a_key || (k == a_key && v > a_id);
  }
};

__device__ __forceinline__ bool better(float s, uint32_t id, float bs, uint32_t bid) {
  return s > bs || (s == bs && id < bid);
}

// Block-wide best (score, id), lowest id on ties.
__device__ void block_best(float& s, uint32_t& id, Shared& sh) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float s2 = __shfl_xor_sync(0xffffffffu, s, o);
    const uint32_t i2 = __shfl_xor_sync(0xffffffffu, id, o);
    if (better(s2, i2, s, id)) {
      s = s2;
      id = i2;
    }
  }
  __syncthreads();
  if (lane == 0) {
    sh.fred[w] = s;
    sh.ured[w] = id;
  }
  __syncthreads();
  s = lane < kNW ? sh.fred[lane] : -INFINITY;
  id = lane < kNW ? sh.ured[lane] : 0xFFFFFFFFu;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float s2 = __shfl_xor_sync(0xffffffffu, s, o);
    const uint32_t i2 = __shfl_xor_sync(0xffffffffu, id, o);
    if (better(s2, i2, s, id)) {
      s = s2;
      id = i2;
    }
  }
}

__device__ int block_sum_int(int v, Shared& sh) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if (lane == 0) sh.ired[w] = v;
  __syncthreads();
  v = lane < kNW ? sh.ired[lane] : 0;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ double block_sum_double(double v, Shared& sh) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if (lane == 0) sh.dred[w] = v;
  __syncthreads();
  v = lane < kNW ? sh.dred[lane] : 0.0;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ int block_min_int(int v, Shared& sh) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
  __syncthreads();
  if (lane == 0) sh.ired[w] = v;
  __syncthreads();
  v = lane < kNW ? sh.ired[lane] : 0x7FFFFFFF;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Exclusive block scan of one double per thread, in thread order: returns this thread's offset and sets `total`.
__device__ double block_scan_double(double v, double& total, Shared& sh) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  double inc = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const double y = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += y;
  }
  double exc = __shfl_up_sync(0xffffffffu, inc, 1);
  if (lane == 0) exc = 0.0;
  __syncthreads();
  if (lane == 31) sh.dred[w] = inc;  // warp totals
  __syncthreads();
  if (w == 0) {
    const double t = lane < kNW ? sh.dred[lane] : 0.0;
    double ti = t;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const double y = __shfl_up_sync(0xffffffffu, ti, o);
      if (lane >= o) ti += y;
    }
    double te = __shfl_up_sync(0xffffffffu, ti, 1);
    if (lane == 0) te = 0.0;
    if (lane < kNW) sh.dscan[lane] = te;  // exclusive offsets of the warps
    if (lane == kNW - 1) sh.dtotal = ti;
  }
  __syncthreads();
  total = sh.dtotal;
  const double off = sh.dscan[w] + exc;
  __syncthreads();
  return off;
}

// Warp-aggregated histogram increment of `bin` (a lane with no candidate in this pass passes bin 256).
__device__ __forceinline__ void hist_add(uint32_t* hist, uint32_t bin) {
  const unsigned peers = __match_any_sync(0xffffffffu, bin);
  if (bin < 256 && (threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(&hist[bin], __popc(peers));
}

// The cnt-th largest key (1-based) among live tokens: sets sh.sel_prefix (the key), sh.sel_rem (how many tokens with
// exactly that key are among the first cnt) and sh.sel_count_eq (how many live tokens have that key).
template <class R>
__device__ void radix_select(const R& r, int cnt, Shared& sh) {
  uint32_t prefix = 0;
  int rem = cnt;
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    const uint32_t mask = pass == 0 ? 0u : 0xFFFFFFFFu << (32 - 8 * pass);
    for (int i = threadIdx.x; i < 256; i += kT) sh.hist[i] = 0;
    __syncthreads();
    for (int base = 0; base < r.V; base += kT) {
      const int v = base + threadIdx.x;
      uint32_t bin = 256;
      if (v < r.V) {
        const float xv = r.x(v);
        const uint32_t k = okey(xv);
        if (r.live(k, xv, v) && (k & mask) == prefix) bin = (k >> shift) & 255u;
      }
      hist_add(sh.hist, bin);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int above = 0, b = 255;
      for (; b > 0; --b) {
        if (above + (int)sh.hist[b] >= rem) break;
        above += sh.hist[b];
      }
      sh.sel_prefix = prefix | ((uint32_t)b << shift);
      sh.sel_rem = rem - above;
      sh.sel_count_eq = sh.hist[b];
    }
    __syncthreads();
    prefix = sh.sel_prefix;
    rem = sh.sel_rem;
  }
}

// Gather the batch that `radix_select` chose into sh.cand: live tokens with key > K, and the first `need` (by id) with
// key == K. Returns the number gathered (unsorted).
template <class R>
__device__ int gather(const R& r, Shared& sh) {
  const uint32_t K = sh.sel_prefix;
  const int need = sh.sel_rem;
  const bool all_eq = (int)sh.sel_count_eq == need;
  const bool small_eq = sh.sel_count_eq <= kEqCap;
  if (threadIdx.x == 0) {
    sh.n_cand = 0;
    sh.n_eq = 0;
  }
  __syncthreads();
  for (int v = threadIdx.x; v < r.V; v += kT) {
    const float xv = r.x(v);
    const uint32_t k = okey(xv);
    if (!r.live(k, xv, v) || k < K) continue;
    if (k > K || all_eq) {
      sh.cand[atomicAdd(&sh.n_cand, 1)] = ((uint64_t)(~k) << 32) | (uint32_t)v;
    } else if (small_eq) {
      sh.eq[atomicAdd(&sh.n_eq, 1)] = (uint32_t)v;
    }
  }
  __syncthreads();
  if (!all_eq) {
    if (small_eq) {
      const int n_eq = sh.n_eq;
      for (int i = threadIdx.x; i < n_eq; i += kT) {
        const uint32_t id = sh.eq[i];
        int rank = 0;
        for (int j = 0; j < n_eq; ++j) rank += sh.eq[j] < id;
        if (rank < need) sh.cand[atomicAdd(&sh.n_cand, 1)] = ((uint64_t)(~K) << 32) | id;
      }
    } else {  // many ties at the boundary: an ordered pass over the row, a chunk of kT ids at a time
      int seen = 0;
      for (int base = 0; base < r.V && seen < need; base += kT) {
        const int v = base + threadIdx.x;
        bool eq = false;
        if (v < r.V) {
          const float xv = r.x(v);
          const uint32_t k = okey(xv);
          eq = r.live(k, xv, v) && k == K;
        }
        const unsigned ball = __ballot_sync(0xffffffffu, eq);
        const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
        __syncthreads();
        if (lane == 0) sh.ired[w] = __popc(ball);
        __syncthreads();
        int before = 0, total = 0;
        for (int i = 0; i < kNW; ++i) {
          before += i < w ? sh.ired[i] : 0;
          total += sh.ired[i];
        }
        const int rank = seen + before + __popc(ball & ((1u << lane) - 1));
        if (eq && rank < need) sh.cand[atomicAdd(&sh.n_cand, 1)] = ((uint64_t)(~K) << 32) | (uint32_t)v;
        seen += total;
      }
    }
  }
  __syncthreads();
  return sh.n_cand;
}

__device__ void bitonic_sort(uint64_t* a, int n) {
  int p2 = 1;
  while (p2 < n) p2 <<= 1;
  for (int i = n + threadIdx.x; i < p2; i += kT) a[i] = ~0ull;
  __syncthreads();
  for (int k = 2; k <= p2; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < p2; i += kT) {
        const int ixj = i ^ j;
        if (ixj > i) {
          const uint64_t x = a[i], y = a[ixj];
          const bool up = (i & k) == 0;
          if ((x > y) == up) {
            a[i] = y;
            a[ixj] = x;
          }
        }
      }
      __syncthreads();
    }
  }
}

// fp32 terms summed with Neumaier compensation per thread, then in fp64 across the block.
struct CSum {
  float s = 0.0f, c = 0.0f;
  __device__ __forceinline__ void add(float e) {
    const float t = s + e;
    c += fabsf(s) >= fabsf(e) ? (s - t) + e : (e - t) + s;
    s = t;
  }
  __device__ __forceinline__ double value() const { return (double)s + (double)c; }
};

// One row (the kernel's CTA), with its terms (`kTerms`) or without.
template <bool kTerms>
__device__ __forceinline__ void sample_row(const float* l, int V, int64_t slot, int64_t qpos, float T,
                                           const float* t_top_p, const int64_t* t_top_k, const float* t_min_p,
                                           const int64_t* t_seed, const int32_t* mk, const int32_t* tk,
                                           const int32_t* ids, const Adj& a, Shared& sh, int64_t& token, int& bad,
                                           bool& empty) {
  auto tid = [ids](uint32_t v) -> uint32_t { return ids != nullptr ? (uint32_t)ids[v] : v; };
  // Pass 1: the max over the allowed columns (lowest id on ties) and non-finite logits (all of them, padding too).
  float best = -INFINITY;
  uint32_t best_id = 0xFFFFFFFFu;
  bad = 0;
  for (int v = threadIdx.x; v < V; v += kT) {
    float x = l[v];
    bad |= !isfinite(x);
    if (kTerms) x = a.apply(x, ids != nullptr ? ids[v] : v);
    // a thread visits ids in increasing order: the first of equal values stays
    if (x > best && col_allowed(mk, tk, v)) {
      best = x;
      best_id = v;
    }
  }
  bad = block_sum_int(bad, sh);
  block_best(best, best_id, sh);
  // no allowed column (only possible where the test can refuse one: a mask, or padding)
  empty = (mk != nullptr || tk != nullptr) && best_id == 0xFFFFFFFFu;
  if (bad || empty) {
    token = 0;
  } else if (!(T >= kGreedyEps)) {
    token = best_id;
  } else {
    const uint64_t key = splitmix64((uint64_t)t_seed[slot] ^ splitmix64((uint64_t)(qpos + 1)));
    const float m = kTerms ? clampf(fdiv(best, T)) : fdiv(best, T);
    const float mp = t_min_p[slot];
    const float p = t_top_p[slot];
    const int64_t k = t_top_k[slot];
    Row<kTerms> row{l, V, T, mp > 0.0f ? fadd(m, logf(mp)) : -INFINITY, false, 0u, 0u, mk, tk, ids, a};
    float bs = -INFINITY;
    uint32_t bid = 0xFFFFFFFFu;
    if (k <= 0 && p >= 1.0f) {  // plain: Gumbel-max over the min-p survivors
      for (int v = threadIdx.x; v < V; v += kT) {
        const float xv = row.x(v);
        if (xv >= row.thr && row.allowed(v)) {
          const float s = fadd(xv, gumbel(key, tid(v)));
          if (better(s, v, bs, bid)) {
            bs = s;
            bid = v;
          }
        }
      }
      block_best(bs, bid, sh);
    } else {
      // n_valid: the column count, or the min-p survivor count when min-p is on. Under a mask or padding, the columns
      // outnumber the allowed tokens, and a select for more tokens than remain gathers them all (it runs out at key 0).
      int n_valid = V;
      if (mp > 0.0f) {
        int c = 0;
        for (int v = threadIdx.x; v < V; v += kT) c += row.x(v) >= row.thr && row.allowed(v);
        n_valid = block_sum_int(c, sh);
      }
      const int k_eff = k > 0 && k < n_valid ? (int)k : n_valid;
      const bool top_p = p < 1.0f;
      double target = 0.0;
      bool z_from_batch = false;
      if (top_p) {
        if (k_eff <= kCap) {
          z_from_batch = true;  // the first batch is the whole top-k set
        } else {
          // Z over the top-k set: tokens above its last key, plus `need` tokens at that key (all with the same e).
          uint32_t K = 0;
          int need = 0;
          if (k_eff < n_valid) {
            radix_select(row, k_eff, sh);
            K = sh.sel_prefix;
            // Fewer than k_eff allowed tokens (a mask or padding, without min-p): the select ran out at K = 0. Every
            // allowed token is above it, so there is no boundary key. A boundary term would read the float of key 0,
            // a NaN, and top-p would keep the whole set.
            need = sh.sel_rem <= (int)sh.sel_count_eq ? sh.sel_rem : 0;
          }
          CSum cs;
          for (int v = threadIdx.x; v < V; v += kT) {
            const float xv = row.x(v);
            if (xv >= row.thr && row.allowed(v) && (k_eff == n_valid || okey(xv) > K)) cs.add(expf(fsub(xv, m)));
          }
          double z = block_sum_double(cs.value(), sh);
          if (need > 0) {
            // the boundary key's value: any token with key K has x = the float of K
            const uint32_t b = K ^ ((K & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu);
            z += (double)need * (double)expf(fsub(__uint_as_float(b), m));
          }
          target = (double)p * z;
        }
      }
      double run = 0.0;  // the top-p sum over the batches done
      int done = 0;
      bool cut = false;
      while (done < k_eff && !cut) {
        const int cnt = min(kCap, k_eff - done);
        radix_select(row, cnt, sh);
        const int n = gather(row, sh);
        bitonic_sort(sh.cand, n);
        // two candidates per thread, in order
        const int j0 = 2 * threadIdx.x;
        float x0 = -INFINITY, x1 = -INFINITY;
        uint32_t i0 = 0, i1 = 0;
        float e0 = 0.0f, e1 = 0.0f;
        if (j0 < n) {
          i0 = (uint32_t)sh.cand[j0];
          x0 = row.x(i0);
          e0 = expf(fsub(x0, m));
        }
        if (j0 + 1 < n) {
          i1 = (uint32_t)sh.cand[j0 + 1];
          x1 = row.x(i1);
          e1 = expf(fsub(x1, m));
        }
        int keep = n;
        if (top_p) {
          double total;
          const double off = block_scan_double((double)e0 + (double)e1, total, sh);
          if (z_from_batch) target = (double)p * total;  // this batch is the whole top-k set
          const double s0 = run + off, s1 = run + off + (double)e0;  // S of the thread's two candidates
          int first_out = n;  // the first candidate with S > target (S grows along the order)
          if (j0 < n && s0 > target) {
            first_out = j0;
          } else if (j0 + 1 < n && s1 > target) {
            first_out = j0 + 1;
          }
          keep = block_min_int(first_out, sh);
          run += total;
          cut = keep < n || run > target;  // a later batch starts at S = run: nothing of it would be kept
        }
        if (j0 < keep) {
          const float s = fadd(x0, gumbel(key, tid(i0)));
          if (better(s, i0, bs, bid)) {
            bs = s;
            bid = i0;
          }
        }
        if (j0 + 1 < keep) {
          const float s = fadd(x1, gumbel(key, tid(i1)));
          if (better(s, i1, bs, bid)) {
            bs = s;
            bid = i1;
          }
        }
        done += n;
        if (n > 0) {  // the next batch starts after this one's last token
          const uint64_t last = sh.cand[n - 1];
          row.after = true;
          row.a_key = ~(uint32_t)(last >> 32);
          row.a_id = (uint32_t)last;
        }
        __syncthreads();
        if (n == 0) break;
      }
      block_best(bs, bid, sh);
    }
    token = bid;
  }
  if (!bad && !empty && ids != nullptr) token = ids[token];  // an allowed column: never padding
}

template <bool kHasTerms>
__global__ void __launch_bounds__(kT) sample_kernel(const float* __restrict__ logits, int64_t ld, int V,
                                                     const int64_t* __restrict__ row_slot,
                                                     const int64_t* __restrict__ qpos,
                                                     const float* __restrict__ t_temp, const float* __restrict__ t_top_p,
                                                     const int64_t* __restrict__ t_top_k,
                                                     const float* __restrict__ t_min_p,
                                                     const int64_t* __restrict__ t_seed, int64_t* __restrict__ out,
                                                     int64_t* __restrict__ status, int64_t* __restrict__ pending,
                                                     const int32_t* __restrict__ ids, int pad,
                                                     const int32_t* __restrict__ mask, int64_t mask_ld,
                                                     const int64_t* __restrict__ mask_row, TermArgs ta) {
  lmk_pdl_wait();  // PDL: nothing a predecessor writes is read before it (common.cuh)
  lmk_pdl_trigger();
  __shared__ Shared sh;
  const int r = blockIdx.x;
  const int64_t slot = row_slot[r];
  const float* l = logits + (int64_t)r * ld;
  const float T = t_temp[slot];
  const int32_t* mk = nullptr;
  if (mask != nullptr && mask_row[r] >= 0) mk = mask + mask_row[r] * mask_ld;
  // the ids the allowed test reads: to skip padding, or to test the mask bits of the columns' tokens
  const int32_t* tk = ids != nullptr && (pad != 0 || mk != nullptr) ? ids : nullptr;
  int64_t token;
  int bad;
  bool empty;
  const int terms = kHasTerms ? ta.terms[slot] : 0;
  Adj a{};
  if (kHasTerms && terms != 0) {
    a.e = ta.cnt + slot * ta.cnt_ld;
    a.bid = ta.bias_id + slot * ta.bias_ld;
    a.bval = ta.bias_val + slot * ta.bias_ld;
    a.bn = ta.bias_n[slot];
    a.pen = (terms & 1) != 0;
    a.rep = ta.rep[slot];
    a.inv = fdiv(1.0f, a.rep);  // rounded once, as vLLM's 1.0 / scale
    a.pres = ta.pres[slot];
    a.freq = ta.freq[slot];
    a.nw = 0;
    if (ta.w_tok != nullptr) {  // a verify row's sample token at window position w: the drafts before it
      const int64_t tw = ta.w_tok[ta.w_idx[r]];
      const int w = tw < 0 ? (int)(-1 - tw) : 0;
      a.nw = min(w, min(ta.k, kMaxDrafts));
      for (int i = 0; i < a.nw; ++i) a.win[i] = (int32_t)ta.drafts[slot * ta.k + i];
    }
    sample_row<true>(l, V, slot, qpos[r], T, t_top_p, t_top_k, t_min_p, t_seed, mk, tk, ids, a, sh, token, bad,
                     empty);
  } else {
    sample_row<false>(l, V, slot, qpos[r], T, t_top_p, t_top_k, t_min_p, t_seed, mk, tk, ids, a, sh, token, bad,
                      empty);
  }
  if (threadIdx.x == 0) {
    out[r] = token;
    if (status != nullptr) status[r] = bad ? 1 : (empty ? 2 : 0);
    if (pending != nullptr) pending[slot] = token;
    if (kHasTerms && ta.commit && (terms & 1) && !bad && !empty) atomicAdd(ta.cnt + slot * ta.cnt_ld + token, 1);
  }
}

}  // namespace

// logits [n, V] fp32 at row stride ld. Per row r, row_slot[r] indexes the runner's per-slot tables (temperature,
// top_p, top_k int64, min_p, seed int64). qpos[r] is the position of the row's last query token; the sampled token
// takes qpos + 1. The kernel writes out[r] (int64), status[r] (int64, may be null) and pending[row_slot[r]] (may be
// null). ids (int32 [V], ascending) and mask (int32 [*, mask_ld]) may be null. pad is a flag (nonzero: padding after
// the ids). mask_row (int64 [n]) is read only with a mask. The terms (`TermArgs`' fields, in order; `terms` null: none,
// the kernel without them): per slot terms, rep, pres, freq (int32, fp32 x 3), cnt (int32 [*, cnt_ld]), bias_n, bias_id
// (int32 [*, bias_ld]), bias_val (fp32), drafts (int64 [*, k]) and per row w_tok, w_idx (int64; null: no window), and
// commit. See the file header.
LMK_API int lmk_sample(const void* logits, int64_t ld, int V, int n, const void* row_slot, const void* qpos,
                       const void* temperature, const void* top_p, const void* top_k, const void* min_p,
                       const void* seed, void* out, void* status, void* pending, const void* ids, int pad,
                       const void* mask, int64_t mask_ld, const void* mask_row, const void* terms, const void* rep,
                       const void* pres, const void* freq, void* cnt, int64_t cnt_ld, const void* bias_n,
                       const void* bias_id, const void* bias_val, int64_t bias_ld, const void* drafts, int k,
                       const void* w_tok, const void* w_idx, int commit, cudaStream_t stream) {
  if (n <= 0) return 0;
  if (k > kMaxDrafts) return (int)cudaErrorInvalidValue;
  const TermArgs ta{(const int32_t*)terms, (const float*)rep, (const float*)pres, (const float*)freq, (int32_t*)cnt,
                    cnt_ld, (const int32_t*)bias_n, (const int32_t*)bias_id, (const float*)bias_val, bias_ld,
                    (const int64_t*)drafts, k, (const int64_t*)w_tok, (const int64_t*)w_idx, commit};
  return lmk_launch(terms != nullptr ? sample_kernel<true> : sample_kernel<false>, dim3(n), dim3(kT), 0, stream,
                    (const float*)logits, ld, V, (const int64_t*)row_slot, (const int64_t*)qpos,
                    (const float*)temperature, (const float*)top_p, (const int64_t*)top_k, (const float*)min_p,
                    (const int64_t*)seed, (int64_t*)out, (int64_t*)status, (int64_t*)pending, (const int32_t*)ids, pad,
                    (const int32_t*)mask, mask_ld, (const int64_t*)mask_row, ta);
}

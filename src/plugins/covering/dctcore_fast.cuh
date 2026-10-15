// dctcore_fast.cuh — SET-SIDE GATHER: the exact per-candidate support count
// computed from the small side of the coverage relation.
//
// ball_gather(c, role) counts the words x in B(c) whose cnt satisfies a
// predicate (deficient / critical / literal).  The support walk does that by
// visiting all |B| words of every candidate's ball: nb * |B| random cnt
// reads.  Coverage is symmetric and, for the four shipped families, a pure
// function of the two words' digits:
//
//     x in B(c)  <=>  dist(x, c) <= R      (Hamming / Lee / torus L-inf /
//                                           open-grid L-inf)
//
// so the same count is  #{ z in Z : dist(z, c) <= R }  with Z the set of
// words satisfying the predicate.  Near a cover Z is tiny (the deficient
// words), and Z costs ONE sequential pass over cnt to list.  Then
//     walk     : nb * |B| random 2-byte reads (32-byte sectors)
//     set side : |X| sequential 2-byte reads + nb * |Z| digit compares
// and the binding picks the cheaper (dct_hc.cu: gather_set_path).  Results
// are identical integers either way; the walk remains the fallback whenever
// Z overflows its buffer or the support table is not provably the metric
// ball of the problem (checked once at init: every table entry inside the
// ball and the entry count equal to the ball's cardinality — together with
// the v1 precondition that entries are distinct, the table IS the ball).
//
// Digits are packed one nibble per axis (ax_j <= 16, n <= 16: 64 bits).

#pragma once
#include "dctcore_core.cuh"
#include "dctcore_mu.cuh"
#include "dctcore_lee.cuh"     // PROB_LEE
#include "dctcore_grid.cuh"    // PROB_GRID_LINF

__device__ __forceinline__ uint64_t pack_digits(long long x) {
  uint64_t p = 0;
  for (int i = 0; i < c_n; i++) {
    long long q = x / c_ax[i];
    p |= (uint64_t)(x - q * c_ax[i]) << (4 * i);
    x = q;
  }
  return p;
}

// warp-aggregated slot claim: one global atomic per warp instead of per hit
__device__ __forceinline__ long long claim_slot(bool hit,
                                                unsigned long long *ctr) {
  unsigned m = __ballot_sync(0xffffffffu, hit);
  if (!m) return -1;
  int lane = threadIdx.x & 31;
  int leader = __ffs(m) - 1;
  unsigned long long base = 0;
  if (lane == leader) base = atomicAdd(ctr, (unsigned long long)__popc(m));
  base = __shfl_sync(0xffffffffu, base, leader);
  if (!hit) return -1;
  return (long long)base + __popc(m & ((1u << lane) - 1u));
}

// block-wide sum (all threads must call; result valid in thread 0)
__device__ __forceinline__ long long block_sum_ll(long long v) {
  __shared__ long long red[32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) red[wid] = v;
  __syncthreads();
  v = (threadIdx.x < (blockDim.x >> 5)) ? red[lane] : 0;
  if (wid == 0)
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;                               // valid in thread 0
}

// Has the slot counter already passed cap?  Warp-uniform (lane 0 reads,
// broadcasts).  Once the output is full, further hits are only COUNTED, in
// a per-thread tally added once per block — the total stays exact (every
// hit is either a claim or a tally) while the single-address atomic stops
// being the bottleneck (thr = 1 on a near-cover cell: ~|X| hits).
__device__ __forceinline__ bool slots_full(const unsigned long long *ctr,
                                           long long cap) {
  unsigned long long v = 0;
  if ((threadIdx.x & 31) == 0) v = *(volatile const unsigned long long *)ctr;
  v = __shfl_sync(0xffffffffu, v, 0);
  return (long long)v >= cap;
}

// List the words whose cnt satisfies the predicate, as packed digits.
// *count receives the true number found (may exceed cap; only cap stored).
// Reads cnt eight counters at a time (16-byte loads); every thread of every
// warp runs the same trip count so the warp-wide ballot is well defined.
template <int MODE>
__global__ void k_compact_pred(const uint16_t *cnt, long long N, int target,
                               uint64_t *out, long long cap,
                               unsigned long long *count) {
  long long nvec = N >> 3;
  long long gs = (long long)gridDim.x * blockDim.x;
  long long t0 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  // One vector per trip.  (Four loads in flight per thread, as in the
  // extraction kernel, measured 3-5x SLOWER here once the set has 1e5+
  // words: 1251 vs 297 us at |Z| = 1.2e5 on K8(9,4); kept simple.)
  // Trips rounded up to whole grid strides so every lane votes every time.
  long long trips = (nvec + gs - 1) / gs;
  const uint4 *v = (const uint4 *)cnt;
  long long tally = 0;
  for (long long k = 0; k < trips; k++) {
    long long i = t0 + k * gs;
    uint4 q = make_uint4(0, 0, 0, 0);
    bool in = i < nvec;
    if (in) q = v[i];
    uint32_t w[4] = {q.x, q.y, q.z, q.w};
    unsigned hits = 0;
#pragma unroll
    for (int h = 0; h < 8; h++) {
      int c = (int)((w[h >> 1] >> (16 * (h & 1))) & 0xFFFFu);
      if (in && cnt_pred(MODE, c, target)) hits |= 1u << h;
    }
    // near a cover almost no warp has a hit: one vote, no per-word ballots
    if (!__any_sync(0xffffffffu, hits != 0)) continue;
    if (slots_full(count, cap)) { tally += __popc(hits); continue; }
#pragma unroll
    for (int h = 0; h < 8; h++) {
      long long s = claim_slot((hits >> h) & 1u, count);
      if (s >= 0 && s < cap) out[s] = pack_digits(i * 8 + h);
    }
  }
  // tail (N not a multiple of 8): one warp of block 0
  if (blockIdx.x == 0 && threadIdx.x < 32) {
    long long i = nvec * 8 + threadIdx.x;
    bool hit = i < N && cnt_pred(MODE, (int)cnt[i < N ? i : 0], target);
    long long s = claim_slot(hit, count);
    if (s >= 0 && s < cap) out[s] = pack_digits(i);
  }
  tally = block_sum_ll(tally);
  if (threadIdx.x == 0 && tally)
    atomicAdd(count, (unsigned long long)tally);
}

// does centre c (packed) cover word z (packed)?  The four shipped metrics.
template <int PROB>
__device__ __forceinline__ bool covers_packed(uint64_t c, uint64_t z) {
  if (PROB == PROB_HAMMING) {
    uint64_t x = c ^ z;
    x |= x >> 1;
    x |= x >> 2;
    x &= 0x1111111111111111ULL;
    return __popcll(x) <= c_R;
  }
  int acc = 0;
  for (int i = 0; i < c_n; i++) {
    int a = (int)((c >> (4 * i)) & 15u), b = (int)((z >> (4 * i)) & 15u);
    int d = a > b ? a - b : b - a;
    if (PROB == PROB_GRID_LINF) {
      if (d > c_R) return false;
    } else {
      int e = c_ax[i] - d;
      if (e < d) d = e;
      if (PROB == PROB_TORUS_LINF) {
        if (d > c_R) return false;
      } else {                          // Lee: sum of circular distances
        acc += d;
        if (acc > c_R) return false;
      }
    }
  }
  return true;
}

// out[wi] = #{ z in Z : z in B(words[wi]) }; split blocks per word share Z.
template <int PROB>
__global__ void k_gather_set(const long long *words, int split,
                             const uint64_t *zd, long long nz, int32_t *out) {
  __shared__ uint64_t cd;
  __shared__ int32_t red[256];
  int wi = blockIdx.x / split, seg = blockIdx.x % split;
  if (threadIdx.x == 0) cd = pack_digits(words[wi]);
  __syncthreads();
  uint64_t c = cd;
  int32_t acc = 0;
  long long step = (long long)split * blockDim.x;
  for (long long t = (long long)seg * blockDim.x + threadIdx.x; t < nz;
       t += step)
    acc += covers_packed<PROB>(c, zd[t]);
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (split == 1) out[wi] = red[0];
    else atomicAdd(&out[wi], red[0]);
  }
}

// ---------------------------------------------------------------------------
// Whole-field reductions with 16-byte loads (8 cnt counters / 4 map values
// per load).  Same integers as the core's k_count_eq / k_count_deficit /
// k_max_i32 / k_hist / k_extract(64); the extraction claims slots with one
// atomic per warp instead of one per hit (slot order was already
// unspecified: the contract is an unordered set).


// MODE MUP_EQ: sum [cnt == target];  MODE MUP_DEF: sum max(0, mu - cnt)
template <int MODE>
__global__ void k_cnt_reduce(const uint16_t *cnt, long long N, int target,
                             long long *out) {
  long long nvec = N >> 3;
  long long gs = (long long)gridDim.x * blockDim.x;
  const uint4 *v = (const uint4 *)cnt;
  long long acc = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < nvec; i += gs) {
    uint4 q = v[i];
    uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int h = 0; h < 8; h++) {
      int c = (int)((w[h >> 1] >> (16 * (h & 1))) & 0xFFFFu);
      if (MODE == MUP_EQ) acc += (c == target);
      else { int d = c_mu - c; if (d > 0) acc += d; }
    }
  }
  if (blockIdx.x == 0 && threadIdx.x < (N & 7)) {
    int c = (int)cnt[nvec * 8 + threadIdx.x];
    if (MODE == MUP_EQ) acc += (c == target);
    else { int d = c_mu - c; if (d > 0) acc += d; }
  }
  acc = block_sum_ll(acc);
  if (threadIdx.x == 0 && acc)
    atomicAdd((unsigned long long *)out, (unsigned long long)acc);
}

__global__ void k_max_i32v(const int32_t *A, long long N, long long *out) {
  long long nvec = N >> 2;
  long long gs = (long long)gridDim.x * blockDim.x;
  const int4 *v = (const int4 *)A;
  int32_t m = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < nvec; i += gs) {
    int4 q = v[i];
    m = max(m, max(max(q.x, q.y), max(q.z, q.w)));
  }
  if (blockIdx.x == 0 && threadIdx.x < (N & 3)) m = max(m, A[nvec * 4 + threadIdx.x]);
  for (int o = 16; o > 0; o >>= 1) m = max(m, __shfl_down_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0 && m > 0) atomicMax((int *)out, m);
}

__device__ __forceinline__ void hist_one(int32_t v, int nbins, int32_t vmax,
                                         int32_t *sh) {
  if (v > 0) {
    int b = (int)((long long)(v - 1) * nbins / vmax);
    if (b >= nbins) b = nbins - 1;
    atomicAdd(&sh[b], 1);
  }
}

__global__ void k_hist_v(const int32_t *A, long long N, int nbins,
                         int32_t vmax, int32_t *hist) {
  extern __shared__ int32_t sh[];
  for (int i = threadIdx.x; i < nbins; i += blockDim.x) sh[i] = 0;
  __syncthreads();
  long long nvec = N >> 2;
  long long gs = (long long)gridDim.x * blockDim.x;
  const int4 *v = (const int4 *)A;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < nvec; i += gs) {
    int4 q = v[i];
    hist_one(q.x, nbins, vmax, sh);
    hist_one(q.y, nbins, vmax, sh);
    hist_one(q.z, nbins, vmax, sh);
    hist_one(q.w, nbins, vmax, sh);
  }
  if (blockIdx.x == 0 && threadIdx.x < (N & 3))
    hist_one(A[nvec * 4 + threadIdx.x], nbins, vmax, sh);
  __syncthreads();
  for (int i = threadIdx.x; i < nbins; i += blockDim.x)
    if (sh[i]) atomicAdd(&hist[i], sh[i]);
}

// IDX = uint32_t (spaces < 2^32) or int64_t.  Uniform trip count per warp
// (the ballot in claim_slot needs every lane).
template <typename IDX>
__global__ void k_extract_v(const int32_t *A, long long N, int32_t thr,
                            IDX *oidx, int32_t *oval, int cap,
                            long long *ocount) {
  long long nvec = N >> 2;
  long long gs = (long long)gridDim.x * blockDim.x;
  long long t0 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int U = 4;                  // loads in flight per thread
  long long trips = (nvec + gs * U - 1) / (gs * U);
  const int4 *v = (const int4 *)A;
  unsigned long long *ctr = (unsigned long long *)ocount;
  long long tally = 0;
  for (long long k = 0; k < trips; k++) {
    int4 qs[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
      long long i = t0 + (k * U + u) * gs;
      qs[u] = i < nvec ? v[i] : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      long long i = t0 + (k * U + u) * gs;
      bool in = i < nvec;
      int4 q = qs[u];
      int32_t e[4] = {q.x, q.y, q.z, q.w};
      bool any = in && (e[0] >= thr || e[1] >= thr || e[2] >= thr ||
                        e[3] >= thr);
      if (!__any_sync(0xffffffffu, any)) continue;
      if (slots_full(ctr, cap)) {
        if (in)
          tally += (e[0] >= thr) + (e[1] >= thr) + (e[2] >= thr) +
                   (e[3] >= thr);
        continue;
      }
#pragma unroll
      for (int h = 0; h < 4; h++) {
        bool hit = in && e[h] >= thr;
        long long s = claim_slot(hit, ctr);
        if (s >= 0 && s < cap) { oidx[s] = (IDX)(i * 4 + h); oval[s] = e[h]; }
      }
    }
  }
  if (blockIdx.x == 0 && threadIdx.x < 32) {
    long long i = nvec * 4 + threadIdx.x;
    bool hit = i < N && A[i < N ? i : 0] >= thr;
    long long s = claim_slot(hit, ctr);
    if (s >= 0 && s < cap) { oidx[s] = (IDX)i; oval[s] = A[i]; }
  }
  tally = block_sum_ll(tally);
  if (threadIdx.x == 0 && tally) atomicAdd(ctr, (unsigned long long)tally);
}

// ---------------------------------------------------------------------------
// Support walks that report what they changed or found (v0.3 optional ABI:
// dct_ball_update_delta, dct_swap_eval).  WALK selects the table walk:
// 0 = v1 bytes (circulant), 1 = ox nibbles (circulant), 2 = v1 bytes with
// open-board rejection (grid_linf).  Every block walks one contiguous
// segment pattern of one word's support set with a block-uniform loop
// bound, so warp-wide ballots are legal inside.
struct WalkShared {
  long long contrib[MAXN * CONTRIB_STRIDE];
  uint8_t ok[MAXN * CONTRIB_STRIDE];
  uint8_t sdig[MAXN];
};

template <int WALK>
__device__ __forceinline__ void walk_setup(long long w, WalkShared &s) {
  if (WALK == 0) build_contrib(w, s.contrib, s.sdig);
  else if (WALK == 1) build_contrib(w, nullptr, s.sdig);
  else build_contrib_grid(w, s.contrib, s.ok, s.sdig);
}

template <int WALK>
__device__ __forceinline__ long long walk_word(long long w, long long t,
                                               const WalkShared &s,
                                               const uint64_t *pat,
                                               const uint16_t *msk2,
                                               const uint64_t *offs2) {
  if (WALK == 0) return pat_word(w, pat[t], s.contrib);
  if (WALK == 1) return tbl2_word(w, msk2[t], offs2[t], s.sdig);
  return pat_word_grid(w, pat[t], s.contrib, s.ok);
}

// cnt += delta over each word's support set, counting the deficit
// transitions exactly from the atomics' old values: an increment from
// c < mu lowers sum max(0, mu - cnt) by one, a decrement from c <= mu raises
// it by one.  *trans += number of transitions (the host applies the sign).
template <int WALK>
__global__ void k_ball_update_dd(const long long *words, int split, int delta,
                                 uint16_t *cnt, const uint64_t *pat,
                                 const uint16_t *msk2, const uint64_t *offs2,
                                 long long balllen, long long *trans) {
  __shared__ WalkShared s;
  int wi = blockIdx.x / split, seg = blockIdx.x % split;
  long long w = words[wi];
  walk_setup<WALK>(w, s);
  long long acc = 0;
  long long step = (long long)split * blockDim.x;
  for (long long t = (long long)seg * blockDim.x + threadIdx.x; t < balllen;
       t += step) {
    long long x = walk_word<WALK>(w, t, s, pat, msk2, offs2);
    if (x < 0) continue;
    uint32_t *cell = (uint32_t *)((uintptr_t)(&cnt[x]) & ~(uintptr_t)3);
    int sh = (x & 1) ? 16 : 0;
    uint32_t add = 1u << sh;
    uint32_t old = delta > 0 ? atomicAdd(cell, add) : atomicSub(cell, add);
    int c = (int)((old >> sh) & 0xFFFFu);
    acc += delta > 0 ? (c < c_mu) : (c <= c_mu);
  }
  acc = block_sum_ll(acc);
  if (threadIdx.x == 0 && acc)
    atomicAdd((unsigned long long *)trans, (unsigned long long)acc);
}

// For each drop word w_i: loss[i] = #{x in B(w_i) : 1 <= cnt <= mu} (the
// exact deficit increase of removing one copy of w_i) and the list
// L_i = {x in B(w_i) : cnt == mu} (the words that become deficient), as
// (packed digits, i) entries in one shared buffer; *lcount = true total.
template <int WALK>
__global__ void k_drop_walk(const long long *drops, int split,
                            const uint16_t *cnt, const uint64_t *pat,
                            const uint16_t *msk2, const uint64_t *offs2,
                            long long balllen, uint32_t *loss, uint64_t *ld,
                            int32_t *ldrop, long long lcap,
                            unsigned long long *lcount) {
  __shared__ WalkShared s;
  int wi = blockIdx.x / split, seg = blockIdx.x % split;
  long long w = drops[wi];
  walk_setup<WALK>(w, s);
  long long acc = 0;
  long long step = (long long)split * blockDim.x;
  for (long long b = (long long)seg * blockDim.x; b < balllen; b += step) {
    long long t = b + threadIdx.x;
    long long x = t < balllen ? walk_word<WALK>(w, t, s, pat, msk2, offs2)
                              : -1;
    int c = x >= 0 ? (int)cnt[x] : -1;
    acc += (c >= 1 && c <= c_mu);
    long long slot = claim_slot(c == c_mu, lcount);
    if (slot >= 0 && slot < lcap) { ld[slot] = pack_digits(x); ldrop[slot] = wi; }
  }
  acc = block_sum_ll(acc);
  if (threadIdx.x == 0 && acc) atomicAdd(&loss[wi], (uint32_t)acc);
}

// out[j] += #{ (x, i) in L : i == cdrop[j], x in B(cand[j]) }  — the words of
// cand j's ball that the removal of its drop word makes deficient.  Skipped
// (the host falls back) when L overflowed its buffer.
template <int PROB>
__global__ void k_swap_corr(const long long *cand, const int32_t *cdrop,
                            int split, const uint64_t *ld,
                            const int32_t *ldrop, long long lcap,
                            const unsigned long long *lcount, int32_t *out) {
  __shared__ uint64_t cd;
  __shared__ int32_t red[256];
  long long n = (long long)*lcount;
  if (n > lcap) return;
  int j = blockIdx.x / split, seg = blockIdx.x % split;
  if (threadIdx.x == 0) cd = pack_digits(cand[j]);
  __syncthreads();
  uint64_t c = cd;
  int i = cdrop[j];
  int32_t acc = 0;
  long long step = (long long)split * blockDim.x;
  for (long long t = (long long)seg * blockDim.x + threadIdx.x; t < n;
       t += step)
    acc += (ldrop[t] == i) && covers_packed<PROB>(c, ld[t]);
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
    if (threadIdx.x < s2) red[threadIdx.x] += red[threadIdx.x + s2];
    __syncthreads();
  }
  if (threadIdx.x == 0 && red[0]) atomicAdd(&out[j], red[0]);
}

// Large-Z form of k_gather_set: a block holds TC candidates (packed digits
// in shared memory) and streams a grid-stride slice of Z once, comparing
// every z with all TC candidates (register accumulators).  Z is read once
// per candidate tile instead of once per candidate.  out[] must be zeroed.
template <int PROB, int TC>
__global__ void k_gather_set_tiled(const long long *words, int nb,
                                   const uint64_t *zd, long long nz,
                                   int32_t *out) {
  __shared__ uint64_t cds[TC];
  __shared__ int32_t cnts[TC];
  int c0 = blockIdx.y * TC;
  int nc = nb - c0 < TC ? nb - c0 : TC;
  for (int i = threadIdx.x; i < TC; i += blockDim.x) {
    cds[i] = i < nc ? pack_digits(words[c0 + i]) : 0;
    cnts[i] = 0;
  }
  __syncthreads();
  int32_t acc[TC];
#pragma unroll
  for (int i = 0; i < TC; i++) acc[i] = 0;
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < nz;
       t += gs) {
    uint64_t z = zd[t];
#pragma unroll
    for (int i = 0; i < TC; i++) acc[i] += covers_packed<PROB>(cds[i], z);
  }
#pragma unroll
  for (int i = 0; i < TC; i++) {
    int32_t v = acc[i];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0 && v) atomicAdd(&cnts[i], v);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < nc; i += blockDim.x)
    if (cnts[i]) atomicAdd(&out[c0 + i], cnts[i]);
}

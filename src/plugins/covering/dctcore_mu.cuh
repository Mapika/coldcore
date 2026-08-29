// dctcore_mu.cuh — MU-FOLD MULTIPLE COVERING: the state-core extension that
// makes every problem family solve K^mu instead of K.
//
// A mu-fold cover of X requires cnt(x) >= mu for EVERY x (mu = 1 is the
// ordinary covering problem).  The search core never learns about mu: it is
// driven by three quantities whose mu = 1 forms it already uses, so the whole
// generalization is a change of PREDICATE on the cnt plane:
//
//   deficit    D  = sum_x max(0, mu - cnt(x))          (0 <=> feasible)
//   gain(x)      = # words in B(x) with cnt < mu       (= -delta D of adding x)
//   loss(c)      = # words in B(c) with 1 <= cnt <= mu (= +delta D of removing c)
//
// Both marginals are exact: adding x lowers the deficit of every deficient
// word of B(x) by one, and removing a codeword c raises the deficit of every
// word of B(c) that sits at level <= mu (words of B(c) always have cnt >= 1,
// so the "1 <=" clause only excludes words no codeword covers, which no
// removal can touch).  At mu = 1 the predicates collapse to [cnt==0] and
// [cnt==1] — the exact fields the vendored core already computes, which is
// why nothing here is on the mu = 1 code path.
//
// Everything below is ADDITIVE: dctcore_core.cuh stays byte-identical to
// upstream (see VENDORED.md); these kernels duplicate the three cnt-reading
// ones with a templated predicate instead.

#pragma once
#include "dctcore_core.cuh"

// predicate selectors (template arguments; the branch compiles away)
#define MUP_EQ 0    // literal cnt == target        (mu-blind, legacy)
#define MUP_DEF 1   // deficient: cnt < mu          (gain / uncovered)
#define MUP_CRIT 2  // critical:  1 <= cnt <= mu    (loss / peel safety)

static int g_mu = 1;
__constant__ int c_mu;

__device__ __forceinline__ bool cnt_pred(int mode, int c, int target) {
  if (mode == MUP_DEF) return c < c_mu;
  if (mode == MUP_CRIT) return c >= 1 && c <= c_mu;
  return c == target;
}

// A0 = indicator(predicate(cnt)) — the transform input for gain (MUP_DEF)
// and loss (MUP_CRIT) maps.
template <int MODE>
__global__ void k_init_from_cnt_mu(int32_t *A0, const uint16_t *cnt,
                                   long long N) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs)
    A0[i] = cnt_pred(MODE, (int)cnt[i], 0) ? 1 : 0;
}

// total deficit sum_x max(0, mu - cnt(x)) in one scan (mu = 1: the uncovered
// count).  int64 accumulate: |X| * mu can exceed 2^31.
__global__ void k_count_deficit(const uint16_t *cnt, long long N,
                                long long *out) {
  __shared__ long long red[256];
  long long gs = (long long)gridDim.x * blockDim.x;
  long long acc = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs) {
    int d = c_mu - (int)cnt[i];
    if (d > 0) acc += d;
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd((unsigned long long *)out,
                                  (unsigned long long)red[0]);
}

// support-set gather under a mu predicate (the k_ball_gather twin).
template <int FMT, int MODE>
__global__ void k_ball_gather_mu(const long long *words, int split,
                                 const uint16_t *cnt, const uint64_t *pat,
                                 const uint16_t *msk2, const uint64_t *offs2,
                                 long long balllen, int32_t *out) {
  __shared__ long long contrib[MAXN * CONTRIB_STRIDE];
  __shared__ uint8_t sdig[MAXN];
  __shared__ int32_t red[256];
  int wi = blockIdx.x / split, seg = blockIdx.x % split;
  long long w = words[wi];
  build_contrib(w, FMT == 0 ? contrib : nullptr, sdig);
  int32_t acc = 0;
  long long step = (long long)split * blockDim.x;
  for (long long t = (long long)seg * blockDim.x + threadIdx.x; t < balllen;
       t += step) {
    long long x = (FMT == 0) ? pat_word(w, pat[t], contrib)
                             : tbl2_word(w, msk2[t], offs2[t], sdig);
    acc += cnt_pred(MODE, (int)cnt[x], 0);
  }
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

// owner-trick losspass under the mu predicate (the k_loss_scan twin).
__global__ void k_loss_scan_mu(const uint16_t *cnt, const uint32_t *owner,
                               uint32_t *loss, long long N) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs)
    if (cnt[i] >= 1 && (int)cnt[i] <= c_mu) atomicAdd(&loss[owner[i]], 1u);
}

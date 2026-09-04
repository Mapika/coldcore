// dctcore_grid.cuh — PROBLEM PLUGIN 4: dominating sets of the OPEN n-D grid
// graph prod_j [0, m_j) under Chebyshev (L-infinity / king-move) balls of
// radius R — a bounded board, NO wraparound.
//
//   cnt(x) = #{u in S : max_j |x_j - u_j| <= R}
//
// This is the torus plugin minus the wrap, and that minus is the point: the
// coverage relation is no longer shift invariant — the per-axis window CLAMPS
// at the boundary, so balls near an edge are smaller than balls in the middle.
// It is therefore the framework's generality test beyond the three circulant
// families, and it touches BOTH plugin seams:
//
//  * the axis pass is still a separable window sum, now clamped:
//        newA0[a] = sum_{b = max(0,a-R)}^{min(Q-1,a+R)} A0[b]
//    (a sliding window over the buffered fiber).  Coverage stays symmetric
//    (|a-b| <= R either way), so the transform is still self-adjoint and the
//    gain/loss maps come from the same pass, exactly as for the torus.
//
//  * the SUPPORT WALK cannot be the core's circulant one.  The packed v1
//    table stores circular shifts, so near a boundary a pattern wraps to the
//    far side of the board — a word that is NOT in the ball.  The table used
//    here is the torus-shaped one (per-axis deltas d with min(d, Q-d) <= R),
//    and the walk resolves each delta against the actual centre digit:
//        offset = (a + d < Q) ? d : d - Q      (the true board displacement)
//    keeping the pattern only if EVERY position lands within |offset| <= R.
//    Balls thus shrink automatically at the edges, and the table stays the
//    single dense max-size table the core allocates and streams.
//
// Guards: 2 <= ax_j <= MAXAX (nibble-packed deltas), n <= MAXPAT (patterns
// change up to n positions, as for the torus).  2R+1 <= ax_j is NOT required:
// an oversized window simply clamps to the whole axis.
//
// Support format: v1 packed bytes only (the ox format's register walk is
// circulant; dct_use_fmt(1) is rejected for this problem).

#pragma once
#include "dctcore_core.cuh"
#include "dctcore_mu.cuh"

#define PROB_GRID_LINF 3  // open-boundary L-infinity grid domination

// clamped window sum over one fiber.  OUT as elsewhere: 0/1 write A0, 2
// clamps into cnt.  Single plane, so OUT 0 and OUT 1 coincide.
template <int OUT>
__global__ void axis_grid(int32_t *A0, long long stride, long long nfib, int Q,
                          int R, uint16_t *cnt) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       f < nfib; f += gs) {
    long long lo = f % stride, hi = f / stride;
    long long base = hi * stride * Q + lo;
    int32_t v[MAXAX];
    for (int a = 0; a < Q; a++) v[a] = A0[base + (long long)a * stride];
    // sliding window: s = sum over [a-R, a+R] intersected with [0, Q)
    int32_t s = 0;
    for (int b = 0; b <= (R < Q - 1 ? R : Q - 1); b++) s += v[b];
    for (int a = 0; a < Q; a++) {
      long long x = base + (long long)a * stride;
      if (OUT == 2) cnt[x] = (uint16_t)(s > 65535 ? 65535 : s);
      else A0[x] = s;
      int add = a + R + 1, drop = a - R;      // window advances to a+1
      if (add < Q) s += v[add];
      if (drop >= 0) s -= v[drop];
    }
  }
}

static int grid_axes(int out_mode, int blocks, int threads) {
  for (int j = 0; j < g_n; j++) {
    int Q = g_ax[j];
    long long stride = g_powq[j];
    long long nfib = g_space / Q;
    int last = (j == g_n - 1);
    int out = last ? out_mode : 0;
    if (out == 2)
      axis_grid<2><<<blocks, threads>>>(g_A[0], stride, nfib, Q, g_R, g_cnt);
    else
      axis_grid<0><<<blocks, threads>>>(g_A[0], stride, nfib, Q, g_R, g_cnt);
  }
  return 0;
}

// ------------------------------------------------------ clamped ball walks
// Per (pos, delta): the signed board displacement of the centre digit, and
// whether that displacement is inside the L-inf radius at all (it is not,
// when the circular delta had to wrap past a boundary).
__device__ __forceinline__ void build_contrib_grid(long long w,
                                                   long long *contrib,
                                                   uint8_t *ok,
                                                   uint8_t *sdig) {
  int tid = threadIdx.x;
  if (tid == 0) {
    long long x = w;
    for (int i = 0; i < c_n; i++) {
      sdig[i] = (uint8_t)(x % c_ax[i]); x /= c_ax[i];
    }
  }
  __syncthreads();
  for (int t = tid; t < c_n * CONTRIB_STRIDE; t += blockDim.x) {
    int p = t / CONTRIB_STRIDE, delta = 1 + t % CONTRIB_STRIDE;
    long long off = 0;
    uint8_t good = 0;
    if (delta < c_ax[p]) {
      int nd = (int)sdig[p] + delta;
      int sh = (nd < c_ax[p]) ? delta : delta - c_ax[p];
      good = (sh <= c_R && sh >= -c_R) ? 1 : 0;
      off = (long long)sh * c_powq[p];
    }
    contrib[t] = off;
    ok[t] = good;
  }
  __syncthreads();
}

// v1 pattern walk with boundary rejection: -1 = this pattern leaves the board
// (or leaves the ball) for this centre.
__device__ __forceinline__ long long pat_word_grid(long long w, uint64_t pat,
                                                   const long long *contrib,
                                                   const uint8_t *ok) {
  long long x = w;
#pragma unroll
  for (int k = 0; k < MAXPAT; k++) {
    unsigned b = (unsigned)(pat >> (8 * k)) & 255u;
    if (!b) break;
    int e = (int)(b >> 4) * CONTRIB_STRIDE + (int)(b & 15u) - 1;
    if (!ok[e]) return -1;
    x += contrib[e];
  }
  return x;
}

template <bool OWNER>
__global__ void k_ball_update_grid(const long long *words, int split, int delta,
                                   uint16_t *cnt, uint32_t *owner,
                                   const uint64_t *pat, long long balllen) {
  __shared__ long long contrib[MAXN * CONTRIB_STRIDE];
  __shared__ uint8_t ok[MAXN * CONTRIB_STRIDE];
  __shared__ uint8_t sdig[MAXN];
  int wi = blockIdx.x / split, seg = blockIdx.x % split;
  long long w = words[wi];
  build_contrib_grid(w, contrib, ok, sdig);
  long long step = (long long)split * blockDim.x;
  for (long long t = (long long)seg * blockDim.x + threadIdx.x; t < balllen;
       t += step) {
    long long x = pat_word_grid(w, pat[t], contrib, ok);
    if (x < 0) continue;
    uint32_t *cell = (uint32_t *)((uintptr_t)(&cnt[x]) & ~(uintptr_t)3);
    uint32_t add = ((x & 1) ? (1u << 16) : 1u);
    if (delta > 0) atomicAdd(cell, add);
    else atomicSub(cell, add);
    if (OWNER) atomicExch(&owner[x], (uint32_t)wi);
  }
}

// MODE: MUP_EQ (literal cnt == target) | MUP_DEF | MUP_CRIT — mu works here
// exactly as for the circulant families (see dctcore_mu.cuh).
template <int MODE>
__global__ void k_ball_gather_grid(const long long *words, int split,
                                   int target, const uint16_t *cnt,
                                   const uint64_t *pat, long long balllen,
                                   int32_t *out) {
  __shared__ long long contrib[MAXN * CONTRIB_STRIDE];
  __shared__ uint8_t ok[MAXN * CONTRIB_STRIDE];
  __shared__ uint8_t sdig[MAXN];
  __shared__ int32_t red[256];
  int wi = blockIdx.x / split, seg = blockIdx.x % split;
  long long w = words[wi];
  build_contrib_grid(w, contrib, ok, sdig);
  int32_t acc = 0;
  long long step = (long long)split * blockDim.x;
  for (long long t = (long long)seg * blockDim.x + threadIdx.x; t < balllen;
       t += step) {
    long long x = pat_word_grid(w, pat[t], contrib, ok);
    if (x >= 0) acc += cnt_pred(MODE, (int)cnt[x], target);
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

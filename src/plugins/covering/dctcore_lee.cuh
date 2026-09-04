// dctcore_lee.cuh — PROBLEM PLUGIN 3: covering codes under the LEE metric
// on mixed-radix product spaces prod_j Z_{ax_j}.
//
// Per coordinate the Lee distance is circular, d_j(a,b) = min(|a-b|,
// ax_j - |a-b|); the total distance is the SUM over coordinates — the
// sum-metric analogue of Hamming, so the same grade-truncated axis DP
// applies, with the circulant per-axis kernel replacing "any other
// digit".  With W = ax_j / 2 the fiber recurrence is
//     newA_d[a] = A_d[a] + sum_{w=1..min(d,W)} K_w(a),
//     K_w(a)    = A_{d-w}[a-w] + A_{d-w}[a+w]     (indices mod ax_j),
// where the two terms coincide — count ONCE — iff ax_j is even and
// w == ax_j/2 (the antipodal digit has multiplicity 1).  After all n
// axes, A_d = number of S-elements at Lee distance exactly d;
// gain/loss/cnt = sum_{d<=R} A_d, as in the Hamming plugin.
//
// Guards: 2 <= ax_j <= MAXAX (nibble-packed support deltas), R <= MAXR
// (layer planes).  Support patterns change at most min(n, R) <= MAXPAT
// positions (every changed position costs Lee weight >= 1).

#pragma once
#include "dctcore_core.cuh"

#define PROB_LEE 2  // covering codes under the Lee (circular sum) metric

// newA_d[a] over one buffered fiber v[d * MAXAX + a] (grade-truncated:
// only grades d-w >= 0 contribute; per-axis weights above W are absent).
__device__ __forceinline__ int32_t lee_grade(const int32_t *v, int a, int d,
                                             int Q, int W) {
  int32_t s = v[d * MAXAX + a];
  int wm = d < W ? d : W;
  for (int w = 1; w <= wm; w++) {
    int am = a - w;
    if (am < 0) am += Q;
    int ap = a + w;
    if (ap >= Q) ap -= Q;
    s += v[(d - w) * MAXAX + am];
    if (ap != am) s += v[(d - w) * MAXAX + ap];
  }
  return s;
}

// OUT semantics as in the Hamming plugin: 0 = write updated layers;
// 1 = write sum_d N_d into A0 (gain/loss map); 2 = clamp into cnt.
template <int OUT>
__global__ void axis_lee(Ptrs P, long long stride, long long nfib, int Q,
                         uint16_t *cnt) {
  long long gs = (long long)gridDim.x * blockDim.x;
  int R1 = c_R + 1, W = Q / 2;
  for (long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       f < nfib; f += gs) {
    long long lo = f % stride, hi = f / stride;
    long long base = hi * stride * Q + lo;
    int32_t v[(MAXR + 1) * MAXAX];
    for (int a = 0; a < Q; a++) {
      long long x = base + (long long)a * stride;
      for (int d = 0; d < R1; d++) v[d * MAXAX + a] = P.a[d][x];
    }
    for (int a = 0; a < Q; a++) {
      long long x = base + (long long)a * stride;
      if (OUT == 0) {
        // descending grade; A_0 is untouched (the w = 0 term keeps it)
        for (int d = R1 - 1; d >= 1; d--)
          P.a[d][x] = lee_grade(v, a, d, Q, W);
      } else {
        int32_t s = v[a];
        for (int d = 1; d < R1; d++) s += lee_grade(v, a, d, Q, W);
        if (OUT == 1) P.a[0][x] = s;
        else cnt[x] = (uint16_t)(s > 65535 ? 65535 : s);
      }
    }
  }
}

// plugin transform driver: axis passes 0..n-1 (A0 already initialized by
// the core; layers 1..R already zeroed).
static int lee_axes(int out_mode, int blocks, int threads) {
  Ptrs P;
  for (int d = 0; d <= g_R; d++) P.a[d] = g_A[d];
  for (int j = 0; j < g_n; j++) {
    int Q = g_ax[j];
    long long stride = g_powq[j];
    long long nfib = g_space / Q;
    int last = (j == g_n - 1);
    int out = last ? out_mode : 0;
    if (out == 0)
      axis_lee<0><<<blocks, threads>>>(P, stride, nfib, Q, g_cnt);
    else if (out == 1)
      axis_lee<1><<<blocks, threads>>>(P, stride, nfib, Q, g_cnt);
    else
      axis_lee<2><<<blocks, threads>>>(P, stride, nfib, Q, g_cnt);
  }
  return 0;
}

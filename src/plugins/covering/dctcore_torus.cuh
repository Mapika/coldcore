// dctcore_torus.cuh — PROBLEM PLUGIN 2: dominating sets on torus grids
// prod_j Z_{m_j} under Chebyshev (L-infinity / king-move) balls of radius R.
//
// cnt(x) = #{u in S : max_j dist_circ(x_j, u_j) <= R} — a separable circular
// window sum: one plane A0, per-axis fiber pass
//     newA0[a] = sum_{e=-R..R} A0[(a+e) mod Q].
// After all axes A0 = cnt exactly.  Gain/loss come from the same pass with
// A0 = [cnt==0] / [cnt==1] (the ball is symmetric, so the transform is
// self-adjoint just like the Hamming plugin).
//
// Guards: 2R+1 <= min axis (window must not self-wrap), axes <= MAXAX.
// The incremental support table is the same packed-pattern format the state
// core walks — patterns have up to n nonzero positions, so n <= MAXPAT for
// this plugin (2D/3D/4D grids in practice).
//
// This plugin is the genericity proof for the framework seams: it reuses the
// state core (fields, walks, reductions, extraction, owner losspass) and the
// search core untouched; only this axis pass is new.

#pragma once
#include "dctcore_core.cuh"

// OUT semantics as in the Hamming plugin; OUT 0 and 1 coincide here (single
// plane): write the window sum into A0.  OUT 2 clamps into cnt.
template <int OUT>
__global__ void axis_torus(int32_t *A0, long long stride, long long nfib,
                           int Q, int R, uint16_t *cnt) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       f < nfib; f += gs) {
    long long lo = f % stride, hi = f / stride;
    long long base = hi * stride * Q + lo;
    int32_t v[MAXAX];
    for (int a = 0; a < Q; a++) v[a] = A0[base + (long long)a * stride];
    for (int a = 0; a < Q; a++) {
      int32_t s = 0;
      for (int e = -R; e <= R; e++) {
        int b = a + e;
        if (b < 0) b += Q;
        else if (b >= Q) b -= Q;
        s += v[b];
      }
      long long x = base + (long long)a * stride;
      if (OUT == 2) cnt[x] = (uint16_t)(s > 65535 ? 65535 : s);
      else A0[x] = s;
    }
  }
}

static int torus_axes(int out_mode, int blocks, int threads) {
  for (int j = 0; j < g_n; j++) {
    int Q = g_ax[j];
    long long stride = g_powq[j];
    long long nfib = g_space / Q;
    int last = (j == g_n - 1);
    int out = last ? out_mode : 0;
    if (out == 2)
      axis_torus<2><<<blocks, threads>>>(g_A[0], stride, nfib, Q, g_R, g_cnt);
    else
      axis_torus<0><<<blocks, threads>>>(g_A[0], stride, nfib, Q, g_R, g_cnt);
  }
  return 0;
}

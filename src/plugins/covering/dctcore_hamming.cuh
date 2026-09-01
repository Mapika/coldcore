// dctcore_hamming.cuh — PROBLEM PLUGIN 1: q-ary (and mixed-radix) covering
// codes under Hamming distance.
//
// Transform: exact distance-count layers.  A_d(x) after axes 0..j = number of
// S-elements matching x on axes > j with Hamming distance d on axes <= j.
// Per-axis fiber recurrence (fiber length Q = ax_j):
//     newA_d[a] = A_d[a] + T_{d-1} - A_{d-1}[a],   T_d = sum_a A_d[a]
// After all n axes, A_d = N_d exactly; gain/loss/cnt = sum_{d<=R} N_d.
// Intermediates bounded by C(j,d)(max_ax-1)^d — int32-safe for all cells in
// scope (guarded in init).
//
// Fast path: homogeneous q with a (Q,R) template instantiation; generic
// kernel covers mixed radix and untemplated cells.

#pragma once
#include "dctcore_core.cuh"

#define MAXQH 10   // register-array cap for the hamming kernels

// OUT: 0 = write updated layers; 1 = write sum_d N_d into A0 (gain/loss map);
//      2 = write min(sum, 65535) into cnt (coverage map).
template <int Q, int R1, int OUT>
__global__ void axis_kernel(Ptrs P, long long stride, long long nfib,
                            uint16_t *cnt) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       f < nfib; f += gs) {
    long long lo = f % stride, hi = f / stride;
    long long base = hi * stride * Q + lo;
    int32_t v[R1][Q];
    int32_t T[R1];
#pragma unroll
    for (int d = 0; d < R1; d++) T[d] = 0;
#pragma unroll
    for (int a = 0; a < Q; a++) {
      long long x = base + (long long)a * stride;
#pragma unroll
      for (int d = 0; d < R1; d++) { v[d][a] = P.a[d][x]; T[d] += v[d][a]; }
    }
#pragma unroll
    for (int a = 0; a < Q; a++) {
      long long x = base + (long long)a * stride;
      if (OUT == 0) {
#pragma unroll
        for (int d = R1 - 1; d >= 1; d--)
          P.a[d][x] = v[d][a] + T[d - 1] - v[d - 1][a];
      } else {
        int32_t s = v[0][a];
#pragma unroll
        for (int d = 1; d < R1; d++) s += v[d][a] + T[d - 1] - v[d - 1][a];
        if (OUT == 1) P.a[0][x] = s;
        else cnt[x] = (uint16_t)(s > 65535 ? 65535 : s);
      }
    }
  }
}

// generic (runtime Q, R) kernel: mixed radix + correctness fallback
template <int OUT>
__global__ void axis_kernel_gen(Ptrs P, long long stride, long long nfib,
                                int Q, uint16_t *cnt) {
  long long gs = (long long)gridDim.x * blockDim.x;
  int R1 = c_R + 1;
  for (long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       f < nfib; f += gs) {
    long long lo = f % stride, hi = f / stride;
    long long base = hi * stride * Q + lo;
    int32_t v[(MAXR + 1) * MAXQH];
    int32_t T[MAXR + 1];
    for (int d = 0; d < R1; d++) T[d] = 0;
    for (int a = 0; a < Q; a++) {
      long long x = base + (long long)a * stride;
      for (int d = 0; d < R1; d++) {
        int32_t y = P.a[d][x]; v[d * MAXQH + a] = y; T[d] += y;
      }
    }
    for (int a = 0; a < Q; a++) {
      long long x = base + (long long)a * stride;
      if (OUT == 0) {
        for (int d = R1 - 1; d >= 1; d--)
          P.a[d][x] = v[d * MAXQH + a] + T[d - 1] - v[(d - 1) * MAXQH + a];
      } else {
        int32_t s = v[0 * MAXQH + a];
        for (int d = 1; d < R1; d++)
          s += v[d * MAXQH + a] + T[d - 1] - v[(d - 1) * MAXQH + a];
        if (OUT == 1) P.a[0][x] = s;
        else cnt[x] = (uint16_t)(s > 65535 ? 65535 : s);
      }
    }
  }
}

// plugin transform driver: axis passes 0..n-1 (A0 already initialized by the
// core; layers 1..R already zeroed).
static int hamming_axes(int out_mode, int blocks, int threads) {
  Ptrs P;
  for (int d = 0; d <= g_R; d++) P.a[d] = g_A[d];
  for (int j = 0; j < g_n; j++) {
    int Q = g_ax[j];
    long long stride = g_powq[j];
    long long nfib = g_space / Q;
    int last = (j == g_n - 1);
    int out = last ? out_mode : 0;
#define DISPATCH(QT, R1)                                                       \
  do {                                                                         \
    if (out == 0)                                                              \
      axis_kernel<QT, R1, 0><<<blocks, threads>>>(P, stride, nfib, g_cnt);     \
    else if (out == 1)                                                         \
      axis_kernel<QT, R1, 1><<<blocks, threads>>>(P, stride, nfib, g_cnt);     \
    else                                                                       \
      axis_kernel<QT, R1, 2><<<blocks, threads>>>(P, stride, nfib, g_cnt);     \
  } while (0)
    if (g_homog && Q == 10 && g_R == 6) DISPATCH(10, 7);
    else if (g_homog && Q == 7 && g_R == 5) DISPATCH(7, 6);
    else if (g_homog && Q == 8 && g_R == 4) DISPATCH(8, 5);
    else if (g_homog && Q == 9 && g_R == 4) DISPATCH(9, 5);
    else if (g_homog && Q == 10 && g_R == 5) DISPATCH(10, 6);
    else if (g_homog && Q == 10 && g_R == 4) DISPATCH(10, 5);
    else if (g_homog && Q == 7 && g_R == 4) DISPATCH(7, 5);
    else if (g_homog && Q == 8 && g_R == 5) DISPATCH(8, 6);
    else if (g_homog && Q == 9 && g_R == 5) DISPATCH(9, 6);
    else if (g_homog && Q == 6 && g_R == 5) DISPATCH(6, 6);
    else if (g_homog && Q == 6 && g_R == 4) DISPATCH(6, 5);
    else {
      if (out == 0)
        axis_kernel_gen<0><<<blocks, threads>>>(P, stride, nfib, Q, g_cnt);
      else if (out == 1)
        axis_kernel_gen<1><<<blocks, threads>>>(P, stride, nfib, Q, g_cnt);
      else
        axis_kernel_gen<2><<<blocks, threads>>>(P, stride, nfib, Q, g_cnt);
    }
#undef DISPATCH
  }
  return 0;
}

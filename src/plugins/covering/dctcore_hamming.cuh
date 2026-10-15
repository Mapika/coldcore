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
#include "dctcore_mu.cuh"   // cnt_pred (fused indicator init)

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

// ---------------------------------------------------------------------------
// v0.3 transform: exact-Q templates for every (Q, R) in scope, zero layers
// skipped, indicator init fused into the first pass.
//
// After axis passes 0..j-1 only layers 0..min(j, R) can be nonzero (a word
// at distance d differs in d of the first j positions), so pass j reads
// lin = min(j, R) + 1 layers and writes layers 1..min(j + 1, R).  Layers
// above that are never read before the pass that first writes them, so the
// R per-transform fill passes are gone, and with them every stale value.
// INIT = 1 computes layer 0 on the fly from the cnt predicate (the gain /
// loss indicators) and writes it once for the later passes, replacing the
// separate init kernel.  Unread layers enter as register zeros, so every
// sum below is the same integer the legacy kernel computes.
//
// Q is exact per pass (mixed radix: one instantiation per axis size), so
// the register arrays are fully unrolled; there is no local-memory
// fallback any more.  Instantiated for Q in 2..MAXQH, R in 1..MAXR.
template <int Q, int R1, int OUT, int INIT>
__global__ void axis_ham(Ptrs P, long long stride, long long nfib, int lin,
                         int lout, uint16_t *cnt, int pmode, int ptarget) {
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
      if (INIT) {
        v[0][a] = cnt_pred(pmode, (int)cnt[x], ptarget) ? 1 : 0;
#pragma unroll
        for (int d = 1; d < R1; d++) v[d][a] = 0;
      } else {
#pragma unroll
        for (int d = 0; d < R1; d++) v[d][a] = (d < lin) ? P.a[d][x] : 0;
      }
#pragma unroll
      for (int d = 0; d < R1; d++) T[d] += v[d][a];
    }
#pragma unroll
    for (int a = 0; a < Q; a++) {
      long long x = base + (long long)a * stride;
      if (OUT == 0) {
        if (INIT) P.a[0][x] = v[0][a];
#pragma unroll
        for (int d = 1; d < R1; d++)
          if (d < lout) P.a[d][x] = v[d][a] + T[d - 1] - v[d - 1][a];
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

// ---------------------------------------------------------------------------
// Two axis passes in one kernel (homogeneous pair: ax_j == ax_{j+1} == Q).
//
// A "column" is one Q x Q tile: the words base + a*s1 + b*s1*Q, a, b < Q,
// with s1 = stride of axis j.  TL consecutive columns are staged in shared
// memory (layer-major S[d][a][b][tl]), axis j runs along a for each (b, tl),
// axis j+1 along b for each (a, tl), and only the result goes back to
// memory: the intermediate layers never leave the SM.  Loads and stores are
// coalesced across tl (consecutive columns are consecutive words when
// s1 >= TL; for s1 == 1 a column block is one contiguous range and is
// streamed linearly).  Integer arithmetic identical to two axis_ham passes.
//
// lin  = layers nonzero on input, lmid = after axis j, lout = after j+1;
// OUT = 0 writes layers (INIT ? 0 : 1) .. lout-1, OUT = 1 / 2 the final
// sum into A0 / cnt.  INIT = 1: layer 0 is the cnt predicate (pass 0 only).
template <int Q, int R1, int OUT, int INIT, int TL>
__global__ void __launch_bounds__(Q * TL, (Q * R1 >= 50) ? 2 : 0)
axis_ham_pair(Ptrs P, long long s1, long long ncols, int lin,
                              int lmid, int lout, uint16_t *cnt, int pmode,
                              int ptarget) {
  extern __shared__ int32_t S[];                 // [R1][Q][Q][TL + 1]
  constexpr int QQ = Q * Q;
  const int r = threadIdx.x / TL, tl = threadIdx.x % TL;   // r < Q
  const long long ngroups = (ncols + TL - 1) / TL;
  // the +1 pad spreads the linear (s1 == 1) staging over the banks
#define SIX(d, a, b, t) S[(((d) * Q + (a)) * Q + (b)) * (TL + 1) + (t)]
  for (long long g = blockIdx.x; g < ngroups; g += gridDim.x) {
    long long c0 = g * TL;
    int ncol = (int)(ncols - c0 < TL ? ncols - c0 : TL);
    bool live = tl < ncol;
    long long base = 0;
    if (s1 != 1 && live) {
      long long c = c0 + tl, lo = c % s1, hi = c / s1;
      base = hi * s1 * QQ + lo;
    }
    // ---- load into S.  Every load of a thread is issued before the
    // first shared store (fully unrolled, predicated), so each thread keeps
    // Q * lin requests in flight.
    if (s1 == 1) {
      // the group is the contiguous range [c0*QQ, (c0+ncol)*QQ): stream it
      // linearly (element e = tid + k * blockDim, Q per thread)
      long long w0 = c0 * QQ;
      int lim = ncol * QQ;
      int32_t buf[R1][Q];
#pragma unroll
      for (int k = 0; k < Q; k++) {
        int e = threadIdx.x + k * Q * TL;
#pragma unroll
        for (int d = 0; d < R1; d++) {
          buf[d][k] = 0;
          if (e < lim) {
            if (INIT) {
              if (d == 0)
                buf[0][k] = cnt_pred(pmode, (int)cnt[w0 + e], ptarget) ? 1 : 0;
            } else if (d < lin) {
              buf[d][k] = P.a[d][w0 + e];
            }
          }
        }
      }
#pragma unroll
      for (int k = 0; k < Q; k++) {
        int e = threadIdx.x + k * Q * TL;
        if (e < lim) {
          int t = e / QQ, wq = e % QQ, a = wq % Q, b = wq / Q;
#pragma unroll
          for (int d = 0; d < R1; d++)
            if (d < (INIT ? 1 : lin)) SIX(d, a, b, t) = buf[d][k];
        }
      }
    } else if (live) {
      // thread (r, tl) loads the fiber a = r, b = 0..Q-1 of its column
      int32_t buf[R1][Q];
#pragma unroll
      for (int b = 0; b < Q; b++) {
        long long x = base + (long long)r * s1 + (long long)b * s1 * Q;
#pragma unroll
        for (int d = 0; d < R1; d++) {
          buf[d][b] = 0;
          if (INIT) {
            if (d == 0) buf[0][b] = cnt_pred(pmode, (int)cnt[x], ptarget) ? 1 : 0;
          } else if (d < lin) {
            buf[d][b] = P.a[d][x];
          }
        }
      }
#pragma unroll
      for (int b = 0; b < Q; b++)
#pragma unroll
        for (int d = 0; d < R1; d++)
          if (d < (INIT ? 1 : lin)) SIX(d, r, b, tl) = buf[d][b];
    }
    __syncthreads();
    // ---- axis j: thread (b = r, tl) runs along a.  Layers are streamed in
    // ascending d holding only layers d-1 and d (new_d = v_d + T_{d-1} -
    // v_{d-1}), so the phase needs 2Q registers instead of R1 * Q.
    {
      int b = r;
      int32_t pv[Q], cv[Q], pT = 0;
#pragma unroll
      for (int a = 0; a < Q; a++) {
        pv[a] = SIX(0, a, b, tl);              // lin >= 1 always
        pT += pv[a];
      }
#pragma unroll
      for (int d = 1; d < R1; d++) {
        if (d < lmid) {
          int32_t cT = 0;
#pragma unroll
          for (int a = 0; a < Q; a++) {
            cv[a] = (d < lin) ? SIX(d, a, b, tl) : 0;
            cT += cv[a];
            SIX(d, a, b, tl) = cv[a] + pT - pv[a];
          }
#pragma unroll
          for (int a = 0; a < Q; a++) pv[a] = cv[a];
          pT = cT;
        }
      }
    }
    __syncthreads();
    // ---- axis j+1: thread (a = r, tl) runs along b, streamed the same way.
    // Each finished layer goes out at once: straight to memory when the
    // columns are strided (coalesced across tl), else back into its own S
    // slot (only this thread reads slot (d, a, *, tl) in this phase) for the
    // linear store below.  OUT != 0 accumulates the layer sum instead.
    // layers stored: OUT 0 -> (INIT ? 0 : 1) .. lout-1; else the sum only
    const int dlo = (OUT == 0) ? (INIT ? 0 : 1) : 0;
    const int dhi = (OUT == 0) ? lout : 1;
    {
      int a = r;
      int32_t pv[Q], cv[Q], sum[Q], pT = 0;
#pragma unroll
      for (int b = 0; b < Q; b++) {
        pv[b] = SIX(0, a, b, tl);
        pT += pv[b];
        sum[b] = pv[b];
      }
      if (OUT == 0 && INIT && s1 != 1 && live)
#pragma unroll
        for (int b = 0; b < Q; b++)
          P.a[0][base + (long long)a * s1 + (long long)b * s1 * Q] = pv[b];
#pragma unroll
      for (int d = 1; d < R1; d++) {
        int32_t cT = 0;
#pragma unroll
        for (int b = 0; b < Q; b++) {
          cv[b] = (d < lmid) ? SIX(d, a, b, tl) : 0;
          cT += cv[b];
          int32_t nv = cv[b] + pT - pv[b];
          if (OUT != 0) {
            sum[b] += nv;
          } else if (d < lout) {
            if (s1 == 1) SIX(d, a, b, tl) = nv;
            else if (live)
              P.a[d][base + (long long)a * s1 + (long long)b * s1 * Q] = nv;
          }
        }
#pragma unroll
        for (int b = 0; b < Q; b++) pv[b] = cv[b];
        pT = cT;
      }
      if (OUT != 0) {
#pragma unroll
        for (int b = 0; b < Q; b++) {
          if (s1 == 1) {
            SIX(0, a, b, tl) = sum[b];
          } else if (live) {
            long long x = base + (long long)a * s1 + (long long)b * s1 * Q;
            if (OUT == 2) cnt[x] = (uint16_t)(sum[b] > 65535 ? 65535 : sum[b]);
            else P.a[0][x] = sum[b];
          }
        }
      }
    }
    if (s1 == 1) {
      // linear, coalesced store of the staged layers
      __syncthreads();
      long long w0 = c0 * QQ;
      int lim = ncol * QQ;
#pragma unroll
      for (int k = 0; k < Q; k++) {
        int e = threadIdx.x + k * Q * TL;
        if (e < lim) {
          int t = e / QQ, wq = e % QQ, a = wq % Q, b = wq / Q;
          long long x = w0 + e;
#pragma unroll
          for (int d = 0; d < R1; d++)
            if (d >= dlo && d < dhi) {
              int32_t val = SIX(d, a, b, t);
              if (OUT == 2) cnt[x] = (uint16_t)(val > 65535 ? 65535 : val);
              else P.a[d][x] = val;
            }
        }
      }
    }
    __syncthreads();
  }
#undef SIX
}

struct HamArgs {
  Ptrs P; long long stride, nfib; int lin, lout, pmode, ptarget;
};

template <int Q, int R1>
static void ham_launch_qr(int out, int init, int blocks, const HamArgs &h) {
#define HL(O, I) axis_ham<Q, R1, O, I><<<blocks, 256>>>(                       \
      h.P, h.stride, h.nfib, h.lin, h.lout, g_cnt, h.pmode, h.ptarget)
  if (init) HL(0, 1);
  else if (out == 0) HL(0, 0);
  else if (out == 1) HL(1, 0);
  else HL(2, 0);
#undef HL
}

// pair launcher: TL columns per block, Q*TL threads, R1*Q*Q*TL ints shared
template <int Q, int R1>
static int ham_pair_qr(int out, int init, int blocks, const HamArgs &h,
                       int lmid, int lout) {
  // 32 columns per block. For Q * R1 >= 50 the kernel is bounded to two
  // resident blocks per SM (<= 102 registers at 320 threads): measured on
  // K10(9,5), 61 ms per transform against 76 ms with 16 columns and no
  // bound, 79 ms with 16 columns and four blocks, 97 ms with 8 columns.
  constexpr int TL = 32;
  size_t shm = (size_t)R1 * Q * Q * (TL + 1) * 4;
  long long ncols = g_space / ((long long)Q * Q);
#define HP(O, I)                                                              \
  do {                                                                        \
    auto kf = axis_ham_pair<Q, R1, O, I, TL>;                                 \
    static bool attr = false;                                                 \
    if (shm > 48 * 1024 && !attr) {                                           \
      cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize,   \
                           (int)shm);                                         \
      attr = true;                                                            \
    }                                                                         \
    kf<<<blocks, Q * TL, shm>>>(h.P, h.stride, ncols, h.lin, lmid, lout,      \
                                g_cnt, h.pmode, h.ptarget);                   \
  } while (0)
  if (init && out == 0) HP(0, 1);
  else if (init) HP(1, 1);          // n == 2: the pair is the whole transform
  else if (out == 0) HP(0, 0);
  else if (out == 1) HP(1, 0);
  else HP(2, 0);
#undef HP
  return 0;
}

template <int Q>
static int ham_pair_q(int R1, int out, int init, int blocks,
                      const HamArgs &h, int lmid, int lout) {
  switch (R1) {
    case 2: return ham_pair_qr<Q, 2>(out, init, blocks, h, lmid, lout);
    case 3: return ham_pair_qr<Q, 3>(out, init, blocks, h, lmid, lout);
    case 4: return ham_pair_qr<Q, 4>(out, init, blocks, h, lmid, lout);
    case 5: return ham_pair_qr<Q, 5>(out, init, blocks, h, lmid, lout);
    case 6: return ham_pair_qr<Q, 6>(out, init, blocks, h, lmid, lout);
    case 7: return ham_pair_qr<Q, 7>(out, init, blocks, h, lmid, lout);
  }
  return -1;
}

static int ham_pair(int Q, int R1, int out, int init, int blocks,
                    const HamArgs &h, int lmid, int lout) {
  switch (Q) {
    case 2: return ham_pair_q<2>(R1, out, init, blocks, h, lmid, lout);
    case 3: return ham_pair_q<3>(R1, out, init, blocks, h, lmid, lout);
    case 4: return ham_pair_q<4>(R1, out, init, blocks, h, lmid, lout);
    case 5: return ham_pair_q<5>(R1, out, init, blocks, h, lmid, lout);
    case 6: return ham_pair_q<6>(R1, out, init, blocks, h, lmid, lout);
    case 7: return ham_pair_q<7>(R1, out, init, blocks, h, lmid, lout);
    case 8: return ham_pair_q<8>(R1, out, init, blocks, h, lmid, lout);
    case 9: return ham_pair_q<9>(R1, out, init, blocks, h, lmid, lout);
    case 10: return ham_pair_q<10>(R1, out, init, blocks, h, lmid, lout);
  }
  return -1;
}

template <int Q>
static int ham_launch_q(int R1, int out, int init, int blocks,
                        const HamArgs &h) {
  switch (R1) {
    case 2: ham_launch_qr<Q, 2>(out, init, blocks, h); return 0;
    case 3: ham_launch_qr<Q, 3>(out, init, blocks, h); return 0;
    case 4: ham_launch_qr<Q, 4>(out, init, blocks, h); return 0;
    case 5: ham_launch_qr<Q, 5>(out, init, blocks, h); return 0;
    case 6: ham_launch_qr<Q, 6>(out, init, blocks, h); return 0;
    case 7: ham_launch_qr<Q, 7>(out, init, blocks, h); return 0;
  }
  return -1;
}

static int ham_launch(int Q, int R1, int out, int init, int blocks,
                      const HamArgs &h) {
  switch (Q) {
    case 2: return ham_launch_q<2>(R1, out, init, blocks, h);
    case 3: return ham_launch_q<3>(R1, out, init, blocks, h);
    case 4: return ham_launch_q<4>(R1, out, init, blocks, h);
    case 5: return ham_launch_q<5>(R1, out, init, blocks, h);
    case 6: return ham_launch_q<6>(R1, out, init, blocks, h);
    case 7: return ham_launch_q<7>(R1, out, init, blocks, h);
    case 8: return ham_launch_q<8>(R1, out, init, blocks, h);
    case 9: return ham_launch_q<9>(R1, out, init, blocks, h);
    case 10: return ham_launch_q<10>(R1, out, init, blocks, h);
  }
  return -1;
}

static int g_timing = -1;           // DCT_TIMING=1: per-pass times on stderr
static int g_pair = 1;              // DCT_PAIR_PASSES=0: one axis per kernel

// v0.3 driver.  init_mode as dct_transform: 0 = A0 already holds the code
// indicator; 1 / 2 = fuse the deficient / critical indicator of cnt into
// pass 0 (pmode / ptarget select the predicate, see dctcore_mu.cuh).
static int hamming_axes2(int out_mode, int fuse, int pmode, int ptarget) {
  if (g_timing < 0) {
    g_timing = getenv("DCT_TIMING") ? 1 : 0;
    const char *pp = getenv("DCT_PAIR_PASSES");
    g_pair = pp ? atoi(pp) : 1;
  }
  HamArgs h;
  for (int d = 0; d <= g_R; d++) h.P.a[d] = g_A[d];
  h.pmode = pmode; h.ptarget = ptarget;
  cudaEvent_t e0, e1;
  if (g_timing) { cudaEventCreate(&e0); cudaEventCreate(&e1); }
  for (int j = 0; j < g_n; j++) {
    int Q = g_ax[j];
    h.stride = g_powq[j];
    h.nfib = g_space / Q;
    h.lin = (j < g_R ? j : g_R) + 1;
    h.lout = (j + 1 < g_R ? j + 1 : g_R) + 1;
    bool pair = g_pair && j + 1 < g_n && g_ax[j + 1] == Q;
    int out = (j + (pair ? 1 : 0) == g_n - 1) ? out_mode : 0;
    int init = (j == 0 && fuse) ? 1 : 0;
    if (g_timing) cudaEventRecord(e0, 0);
    int rc;
    if (pair) {
      int lout2 = (j + 2 < g_R ? j + 2 : g_R) + 1;
      long long groups = (g_space / ((long long)Q * Q) + 31) / 32;
      int blocks = (int)(groups < 4 * g_blocks ? groups : 4 * g_blocks);
      rc = ham_pair(Q, g_R + 1, out, init, blocks, h, h.lout, lout2);
    } else {
      long long nb = (h.nfib + 255) / 256;
      int blocks = (int)(nb < g_blocks ? nb : g_blocks);
      rc = ham_launch(Q, g_R + 1, out, init, blocks, h);
    }
    if (rc) return rc;
    if (g_timing) {
      cudaEventRecord(e1, 0);
      cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      fprintf(stderr, "DCT_TIMING hamming pass %d%s Q=%d lin=%d out=%d "
              "init=%d: %.3f ms\n", j, pair ? "+1" : "", Q, h.lin, out,
              init, ms);
    }
    if (pair) j++;
  }
  if (g_timing) { cudaEventDestroy(e0); cudaEventDestroy(e1); }
  CK(cudaGetLastError());
  return 0;
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

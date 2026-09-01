// dctcore_core.cuh — problem-agnostic STATE CORE of the dctcov engine.
//
// Scope: dense integer fields over a finite product space X = prod_j Z_{ax_j}
// (heterogeneous axis sizes allowed), with automatic HBM-vs-LPDDR placement
// on cache-coherent systems (GH200: ATS plain malloc / managed memory), plus
// the problem-blind machinery every plugin shares:
//
//   * field allocation (layer planes A_0..A_L, cnt plane, optional owner)
//   * fills, indicator init ([cnt==t] -> A0), code scatter
//   * SUPPORT-SET WALKS: exact incremental +/- updates and gathers over a
//     move's coverage set, given as a table of packed shift patterns
//     (byte k = pos<<4 | delta, delta in 1..ax_pos-1, up to 8 bytes/pattern;
//     alternative ox format: mask u16 + nibble offsets u64)
//   * owner-trick losspass (atomicExch owner during mark; loss = one scan)
//   * reductions: count_eq, max, histogram, threshold extraction (32/64-bit)
//
// What the core does NOT know: how the transform that fills the fields
// works.  That is the PROBLEM PLUGIN's job (see PLUGIN.md): a plugin turns
// A0 = indicator(S) into "number of S-elements whose neighborhood covers x"
// for every x at once, via separable per-axis passes.

#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cerrno>
#include <sys/syscall.h>
#include <unistd.h>

#define MAXN 16     // max axes
#define MAXAX 16    // max axis size (nibble-packed deltas)
#define MAXR 6      // max Hamming layer index (plugin 1)
#define MAXPAT 8    // max bytes per packed support pattern (uint64)
#define CONTRIB_STRIDE 15

// memory modes per array
#define MEM_DEV 0      // cudaMalloc (HBM)
#define MEM_ATS 1      // plain malloc, GPU access via ATS/C2C (LPDDR)
#define MEM_MANAGED 2  // cudaMallocManaged + preferred location CPU (LPDDR)

// problems (plugins)
#define PROB_HAMMING 0    // covering codes: Hamming-distance count layers
#define PROB_TORUS_LINF 1 // dominating sets on torus grids, Chebyshev balls

// ---------------------------------------------------------------- context
static int g_prob, g_n, g_R, g_nlayers, g_homog;
static int g_ax[MAXN];
static long long g_space;
static long long g_powq[MAXN + 1];
static int32_t *g_A[MAXR + 1];      // layer planes (g_nlayers of them)
static uint16_t *g_cnt;             // exact coverage multiplicity
static uint32_t *g_owner;           // optional owner tags (owner-trick loss)
static uint64_t *g_ballpat;         // packed support patterns (v1 format)
static uint16_t *g_tbl2_mask;       // ox-format table (optional)
static uint64_t *g_tbl2_offs;
static long long g_balllen;
static long long *g_codeidx;        // device copy of solution word indices
static int g_codeM, g_codecap;
static long long *g_scalar;         // device scalar scratch
static int32_t *g_hist;
static uint32_t *g_extract_idx;
static int64_t *g_extract_idx64;
static int32_t *g_extract_val;
static int g_extract_cap;
static long long *g_d_words;        // persistent staging for support walks
static uint32_t *g_d_loss;
static int32_t *g_d_out;
static int g_layers_mode, g_cnt_mode, g_use_owner;
static int g_fmt;                   // 0 = v1 byte patterns, 1 = ox mask+nibbles
static int g_blocks = 4096;
static long long g_bytes_dev, g_bytes_host;

__constant__ long long c_powq[MAXN + 1];
__constant__ int c_ax[MAXN];
__constant__ int c_n, c_R;
__constant__ long long c_space;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  return -(100 + (int)e); } } while (0)

// alloc/free honoring the memory mode; CPU-first-touch for host modes so the
// pages land in LPDDR and never migrate.
static int alloc_buf(void **p, size_t bytes, int mode) {
  if (mode == MEM_DEV) {
    cudaError_t e = cudaMalloc(p, bytes);
    if (e != cudaSuccess) { fprintf(stderr, "cudaMalloc %zu: %s\n", bytes, cudaGetErrorString(e)); return -1; }
    g_bytes_dev += bytes;
  } else if (mode == MEM_ATS) {
    // page-aligned so the whole range can be mbind-pinned to the CPU NUMA
    // node: on GH200 the kernel's access-counter HMM migration otherwise
    // pulls GPU-hot pageable pages INTO HBM (measured: a 86 GB ATS cell
    // filled HBM to 96.8/97.9 GB and thrashed).  MPOL_BIND forbids that.
    size_t ps = (size_t)sysconf(_SC_PAGESIZE);
    size_t abytes = (bytes + ps - 1) / ps * ps;
    if (posix_memalign(p, ps, abytes)) {
      fprintf(stderr, "posix_memalign %zu failed\n", abytes); return -1;
    }
#ifdef __NR_mbind
    unsigned long nodemask = 1UL;    // CPU LPDDR = NUMA node 0
    if (syscall(__NR_mbind, *p, abytes, 2 /*MPOL_BIND*/, &nodemask, 65UL,
                0UL) != 0)
      fprintf(stderr, "warning: mbind(node0) failed, errno=%d — GH200 HMM "
              "may migrate these pages into HBM\n", errno);
#endif
    memset(*p, 0, bytes);            // first touch on CPU -> LPDDR pages
    g_bytes_host += bytes;
  } else {
    cudaError_t e = cudaMallocManaged(p, bytes);
    if (e != cudaSuccess) { fprintf(stderr, "cudaMallocManaged %zu: %s\n", bytes, cudaGetErrorString(e)); return -1; }
    cudaMemAdvise(*p, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(*p, bytes, cudaMemAdviseSetAccessedBy, 0);
    memset(*p, 0, bytes);            // populate on CPU
    g_bytes_host += bytes;
  }
  return 0;
}

static void free_buf(void *p, int mode) {
  if (!p) return;
  if (mode == MEM_ATS) free(p);
  else cudaFree(p);
}

struct Ptrs { int32_t *a[MAXR + 1]; };

// ---------------------------------------------------------------- fills/init
// (own fill kernels: cudaMemset is not defined for plain-malloc ATS pointers)
__global__ void k_fill32(int32_t *p, int32_t v, long long N) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs)
    p[i] = v;
}

__global__ void k_fill16(uint16_t *p, uint16_t v, long long N) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs)
    p[i] = v;
}

__global__ void k_init_from_cnt(int32_t *A0, const uint16_t *cnt, int target,
                                long long N) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs)
    A0[i] = (cnt[i] == target) ? 1 : 0;
}

__global__ void k_scatter_code(int32_t *A0, const long long *idx, int M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M) atomicAdd(&A0[idx[i]], 1);
}

// ------------------------------------------------------- support-set walks
// split blocks per word; per-(pos,delta) index offsets in shared so each v1
// pattern costs one table load plus <=MAXPAT shared adds (no div/mod in the
// hot loop).  Heterogeneous axes: contrib stride is CONTRIB_STRIDE.
__device__ __forceinline__ void build_contrib(long long w, long long *contrib,
                                              uint8_t *sdig) {
  int tid = threadIdx.x;
  if (tid == 0) {
    long long x = w;
    for (int i = 0; i < c_n; i++) {
      sdig[i] = (uint8_t)(x % c_ax[i]); x /= c_ax[i];
    }
  }
  __syncthreads();
  if (contrib)
    for (int t = tid; t < c_n * CONTRIB_STRIDE; t += blockDim.x) {
      int p = t / CONTRIB_STRIDE, delta = 1 + t % CONTRIB_STRIDE;
      if (delta < c_ax[p]) {
        int nd = sdig[p] + delta;
        contrib[t] = ((nd < c_ax[p]) ? (long long)delta
                                     : (long long)(delta - c_ax[p])) *
                     c_powq[p];
      }
    }
  __syncthreads();
}

// FMT 0: v1 byte-packed (pos<<4|delta) patterns + shared contrib table.
__device__ __forceinline__ long long pat_word(long long w, uint64_t pat,
                                              const long long *contrib) {
  long long x = w;
#pragma unroll
  for (int k = 0; k < MAXPAT; k++) {
    unsigned b = (unsigned)(pat >> (8 * k)) & 255u;
    if (!b) break;
    x += contrib[(b >> 4) * CONTRIB_STRIDE + (b & 15u) - 1];
  }
  return x;
}

// FMT 1 (ox transplant): mask u16 + nibble offsets u64, register-only walk.
__device__ __forceinline__ long long tbl2_word(long long w, unsigned msk,
                                               uint64_t offs,
                                               const uint8_t *sdig) {
  long long x = w;
  while (msk) {
    int i = __ffs((int)msk) - 1;
    msk &= msk - 1;
    int od = (int)sdig[i];
    int nd = od + (int)((offs >> (i << 2)) & 0xF);
    if (nd >= c_ax[i]) nd -= c_ax[i];
    x += (long long)(nd - od) * c_powq[i];
  }
  return x;
}

// delta = +1 / -1 on cnt over the support set of each word (exact; cnt never
// wraps because it only decrements words that are genuinely covered).
// OWNER: additionally tag owner[x] with the word's row id (exact where
// cnt==1 at read time — the ox owner trick).
template <int FMT, bool OWNER>
__global__ void k_ball_update(const long long *words, int split, int delta,
                              uint16_t *cnt, uint32_t *owner,
                              const uint64_t *pat, const uint16_t *msk2,
                              const uint64_t *offs2, long long balllen) {
  __shared__ long long contrib[MAXN * CONTRIB_STRIDE];
  __shared__ uint8_t sdig[MAXN];
  int wi = blockIdx.x / split, seg = blockIdx.x % split;
  long long w = words[wi];
  build_contrib(w, FMT == 0 ? contrib : nullptr, sdig);
  long long step = (long long)split * blockDim.x;
  for (long long t = (long long)seg * blockDim.x + threadIdx.x; t < balllen;
       t += step) {
    long long x = (FMT == 0) ? pat_word(w, pat[t], contrib)
                             : tbl2_word(w, msk2[t], offs2[t], sdig);
    uint32_t *cell = (uint32_t *)((uintptr_t)(&cnt[x]) & ~(uintptr_t)3);
    uint32_t add = ((x & 1) ? (1u << 16) : 1u);
    if (delta > 0) atomicAdd(cell, add);
    else atomicSub(cell, add);
    if (OWNER) atomicExch(&owner[x], (uint32_t)wi);
  }
}

// count words in the support set with cnt == target (0 -> exact placement
// gain, 1 -> exact removal loss).  out[wi] accumulated across split blocks
// (host zeroes out[] first when split > 1).
template <int FMT>
__global__ void k_ball_gather(const long long *words, int split, int target,
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
    acc += (cnt[x] == target);
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

// owner-trick losspass: one linear scan; loss[owner[x]]++ where cnt[x]==1.
__global__ void k_loss_scan(const uint16_t *cnt, const uint32_t *owner,
                            uint32_t *loss, long long N) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs)
    if (cnt[i] == 1) atomicAdd(&loss[owner[i]], 1u);
}

// ---------------------------------------------------------------- reductions
__global__ void k_count_eq(const uint16_t *cnt, int target, long long N,
                           long long *out) {
  __shared__ long long red[256];
  long long gs = (long long)gridDim.x * blockDim.x;
  long long acc = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs)
    acc += (cnt[i] == target);
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd((unsigned long long *)out,
                                  (unsigned long long)red[0]);
}

__global__ void k_max_i32(const int32_t *A, long long N, long long *out) {
  __shared__ int32_t red[256];
  long long gs = (long long)gridDim.x * blockDim.x;
  int32_t acc = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs) { int32_t v = A[i]; if (v > acc) acc = v; }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s && red[threadIdx.x + s] > red[threadIdx.x])
      red[threadIdx.x] = red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicMax((int *)out, red[0]);
}

__global__ void k_hist(const int32_t *A, long long N, int nbins, int32_t vmax,
                       int32_t *hist) {
  extern __shared__ int32_t sh[];
  for (int i = threadIdx.x; i < nbins; i += blockDim.x) sh[i] = 0;
  __syncthreads();
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs) {
    int32_t v = A[i];
    if (v > 0) {
      int b = (int)((long long)(v - 1) * nbins / vmax);
      if (b >= nbins) b = nbins - 1;
      atomicAdd(&sh[b], 1);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < nbins; i += blockDim.x)
    atomicAdd(&hist[i], sh[i]);
}

__global__ void k_extract(const int32_t *A, long long N, int32_t thr,
                          uint32_t *oidx, int32_t *oval, int cap,
                          long long *ocount) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs) {
    int32_t v = A[i];
    if (v >= thr) {
      long long slot = atomicAdd((unsigned long long *)ocount, 1ULL);
      if (slot < cap) { oidx[slot] = (uint32_t)i; oval[slot] = v; }
    }
  }
}

__global__ void k_extract64(const int32_t *A, long long N, int32_t thr,
                            int64_t *oidx, int32_t *oval, int cap,
                            long long *ocount) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs) {
    int32_t v = A[i];
    if (v >= thr) {
      long long slot = atomicAdd((unsigned long long *)ocount, 1ULL);
      if (slot < cap) { oidx[slot] = i; oval[slot] = v; }
    }
  }
}

__global__ void k_read_at(const int32_t *A, const long long *idx, int k,
                          int32_t *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) out[i] = A[idx[i]];
}

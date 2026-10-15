// sparse.cu — coldcore plugin ABI 2: explicit incidence.
//
// A second, independent C ABI next to the separable-transform one
// (dct_*, docs/PLUGIN.md v1).  The problem is a bipartite incidence:
// n_cand candidates, n_univ universe elements, and an (n_cand x d) int32
// matrix whose row c lists the d universe elements candidate c covers.
// The engine keeps
//   cnt[u]   int32 multiplicity of universe element u,
//   map[c]   int32 work field over candidates (the last gain/loss field),
// and answers the same oracle questions the search core asks of the
// product-space plugins (recount, gain/loss field, incremental update,
// exact per-candidate gather, count/max/hist/extract/read_at), each one a
// pass over incidence rows instead of an axis transform.
//
// Placement: cnt, map and the extraction buffers live in HBM; the
// incidence matrix goes to HBM (inc_mode 0) or, when it exceeds the
// caller's budget, to CPU memory pinned to NUMA node 0 (inc_mode 1 —
// posix_memalign + mbind, read by the kernels over C2C/ATS on GH200;
// degrades to managed memory with preferred-location CPU on devices
// without pageable-memory access), exactly the modes of the v1 core.
//
// Conventions (same as ABI 1): singleton context, flat sparse_* symbols,
// int return codes (0 ok, negative engine error, -100-e = CUDA error e,
// message on stderr), caller-allocated outputs, no exceptions.
//
// Build: part of the CMake tree (src/plugins/sparse/CMakeLists.txt) ->
// build/plugins/libcoldcore_sparse.so.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/syscall.h>
#include <unistd.h>

#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "coldcore sparse: CUDA error %s at %s:%d\n", \
          cudaGetErrorString(e), __FILE__, __LINE__); \
  return -(100 + (int)e); } } while (0)

namespace {

enum { MEM_DEV = 0, MEM_HOST = 1, MEM_MANAGED = 2 };

bool g_live = false;
long long g_ncand = 0, g_nuniv = 0;
int g_d = 0, g_mu = 1;
int32_t *g_inc = nullptr;
int g_inc_mode = MEM_DEV;
size_t g_inc_bytes = 0;
// CSR / weighted mode (sparse_init_csr): variable-length rows with
// per-incidence weights w and per-universe weights uw
bool g_csr = false;
long long g_nnz = 0;
int64_t *g_indptr = nullptr;
int32_t *g_w = nullptr, *g_uw = nullptr;
int32_t *g_cnt = nullptr, *g_map = nullptr;
long long *g_scalar = nullptr;
int32_t *g_hist = nullptr;
int64_t *g_xidx = nullptr;
int32_t *g_xval = nullptr;
int g_xcap = 0;
long long *g_words = nullptr;     // device staging for candidate lists
int32_t *g_out = nullptr;         // device staging for per-word outputs
long long g_wcap = 0;
long long g_bytes_dev = 0, g_bytes_host = 0;
int g_sms = 0;

const int HIST_MAX = 4096;
const int THREADS = 256;

int grid_for(long long items) {
  long long b = (items + THREADS - 1) / THREADS;
  long long cap = (long long)g_sms * 32;
  if (b > cap) b = cap;
  if (b < 1) b = 1;
  return (int)b;
}

// ---------------------------------------------------------------- alloc
// mode-tagged allocation with a host->buffer copy; mode may degrade
int alloc_arr(void **dst, size_t bytes, int mode, const void *src) {
  if (mode == MEM_HOST) {
    int coherent = 0;
    if (cudaDeviceGetAttribute(&coherent, cudaDevAttrPageableMemoryAccess, 0)
            != cudaSuccess || !coherent)
      mode = MEM_MANAGED;
  }
  g_inc_mode = mode;
  if (mode == MEM_DEV) {
    CK(cudaMalloc(dst, bytes));
    g_bytes_dev += bytes;
    CK(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
  } else if (mode == MEM_HOST) {
    // page-aligned so the whole range can be mbind-pinned to the CPU NUMA
    // node: GH200's access-counter HMM migration otherwise pulls GPU-hot
    // pages into HBM (see docs/benchmarks.md).  MPOL_BIND forbids that.
    size_t ps = (size_t)sysconf(_SC_PAGESIZE);
    size_t abytes = (bytes + ps - 1) / ps * ps;
    void *p = nullptr;
    if (posix_memalign(&p, ps, abytes)) {
      fprintf(stderr, "coldcore sparse: posix_memalign %zu failed\n", abytes);
      return -3;
    }
#ifdef __NR_mbind
    unsigned long nodemask = 1UL;
    if (syscall(__NR_mbind, p, abytes, 2 /*MPOL_BIND*/, &nodemask, 65UL,
                0UL) != 0)
      fprintf(stderr, "coldcore sparse: warning: mbind(node0) failed, "
              "errno=%d\n", errno);
#endif
    memcpy(p, src, bytes);            // first touch on the CPU -> LPDDR
    *dst = p;
    g_bytes_host += bytes;
  } else {
    CK(cudaMallocManaged(dst, bytes));
    cudaMemAdvise(*dst, bytes, cudaMemAdviseSetPreferredLocation,
                  cudaCpuDeviceId);
    cudaMemAdvise(*dst, bytes, cudaMemAdviseSetAccessedBy, 0);
    memcpy(*dst, src, bytes);
    g_bytes_host += bytes;
  }
  return 0;
}

void free_arr(void *p) {
  if (!p) return;
  if (g_inc_mode == MEM_HOST) free(p);
  else cudaFree(p);
}

int alloc_inc(size_t bytes, int mode, const int32_t *src) {
  int rc = alloc_arr((void **)&g_inc, bytes, mode, src);
  if (rc) return rc;
  g_inc_bytes = bytes;
  return 0;
}

void free_inc() {
  free_arr(g_inc); free_arr(g_indptr); free_arr(g_w);
  g_inc = nullptr; g_indptr = nullptr; g_w = nullptr;
  if (g_uw) { cudaFree(g_uw); g_uw = nullptr; }
  g_csr = false; g_nnz = 0;
}

int ensure_words(long long m) {
  if (m <= g_wcap) return 0;
  long long cap = g_wcap ? g_wcap : 4096;
  while (cap < m) cap *= 2;
  if (g_words) { cudaFree(g_words); cudaFree(g_out); g_words = nullptr; g_out = nullptr; }
  CK(cudaMalloc((void **)&g_words, (size_t)cap * 8));
  CK(cudaMalloc((void **)&g_out, (size_t)cap * 4));
  g_bytes_dev += (cap - g_wcap) * 12;
  g_wcap = cap;
  return 0;
}

// -------------------------------------------------------------- kernels
// coverage role of a multiplicity value: 0 = deficient (cnt < mu),
// 1 = critical (1 <= cnt <= mu), anything else = literal cnt == target.
__device__ __forceinline__ int role(int32_t c, int target, int mu) {
  if (target == 0) return c < mu;
  if (target == 1) return (c >= 1) && (c <= mu);
  return c == target;
}

__device__ __forceinline__ int warp_sum(int s) {
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  return s;
}

// one warp per candidate row: coalesced row read, one cnt gather per
// element, warp reduction.
__global__ void k_field(const int32_t *__restrict__ inc,
                        const int32_t *__restrict__ cnt, int32_t *map,
                        long long ncand, int d, int target, int mu) {
  long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  long long nwarps = ((long long)gridDim.x * blockDim.x) >> 5;
  int lane = threadIdx.x & 31;
  for (long long r = warp; r < ncand; r += nwarps) {
    const int32_t *row = inc + r * d;
    int s = 0;
    for (int j = lane; j < d; j += 32) s += role(cnt[row[j]], target, mu);
    s = warp_sum(s);
    if (lane == 0) map[r] = s;
  }
}

// one warp per listed candidate
__global__ void k_gather(const int32_t *__restrict__ inc,
                         const int32_t *__restrict__ cnt,
                         const long long *__restrict__ words, long long m,
                         int d, int target, int mu, int32_t *out) {
  long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  long long nwarps = ((long long)gridDim.x * blockDim.x) >> 5;
  int lane = threadIdx.x & 31;
  for (long long i = warp; i < m; i += nwarps) {
    const int32_t *row = inc + words[i] * d;
    int s = 0;
    for (int j = lane; j < d; j += 32) s += role(cnt[row[j]], target, mu);
    s = warp_sum(s);
    if (lane == 0) out[i] = s;
  }
}

// one thread per (candidate, element) pair
__global__ void k_update(const int32_t *__restrict__ inc, int32_t *cnt,
                         const long long *__restrict__ words, long long m,
                         int d, int delta) {
  long long total = m * d;
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < total; t += gs) {
    long long i = t / d;
    int j = (int)(t - i * d);
    atomicAdd(&cnt[inc[words[i] * d + j]], delta);
  }
}

// ---- weighted CSR mode: exact weighted marginals (see incidence.py)
//   target 0: uw * min(w, max(0, mu - cnt))                  (gain)
//   target 1: uw * (max(0, mu - cnt + w) - max(0, mu - cnt)) for cnt >= w
//             (loss; the gate reproduces the critical role at w = 1)
//   other:    cnt == target                                  (literal)
__device__ __forceinline__ long long wval(int32_t c, int32_t w, int32_t u,
                                          int target, int mu) {
  if (target == 0) { int d = mu - c; if (d < 0) d = 0; return (long long)u * (w < d ? w : d); }
  if (target == 1) {
    if (c < w) return 0;              // never covered by this candidate
    int a = mu - c + w; if (a < 0) a = 0;
    int b = mu - c;     if (b < 0) b = 0;
    return (long long)u * (a - b);
  }
  return c == target;
}

__global__ void k_field_w(const int64_t *__restrict__ indptr,
                          const int32_t *__restrict__ idx,
                          const int32_t *__restrict__ w,
                          const int32_t *__restrict__ uw,
                          const int32_t *__restrict__ cnt, int32_t *map,
                          long long ncand, int target, int mu) {
  long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  long long nwarps = ((long long)gridDim.x * blockDim.x) >> 5;
  int lane = threadIdx.x & 31;
  for (long long r = warp; r < ncand; r += nwarps) {
    long long s0 = indptr[r], s1 = indptr[r + 1];
    long long acc = 0;
    for (long long j = s0 + lane; j < s1; j += 32) {
      int32_t u = idx[j];
      acc += wval(cnt[u], w[j], uw[u], target, mu);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) map[r] = (int32_t)acc;
  }
}

__global__ void k_gather_w(const int64_t *__restrict__ indptr,
                           const int32_t *__restrict__ idx,
                           const int32_t *__restrict__ w,
                           const int32_t *__restrict__ uw,
                           const int32_t *__restrict__ cnt,
                           const long long *__restrict__ words, long long m,
                           int target, int mu, int32_t *out) {
  long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  long long nwarps = ((long long)gridDim.x * blockDim.x) >> 5;
  int lane = threadIdx.x & 31;
  for (long long i = warp; i < m; i += nwarps) {
    long long r = words[i];
    long long s0 = indptr[r], s1 = indptr[r + 1];
    long long acc = 0;
    for (long long j = s0 + lane; j < s1; j += 32) {
      int32_t u = idx[j];
      acc += wval(cnt[u], w[j], uw[u], target, mu);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) out[i] = (int32_t)acc;
  }
}

__global__ void k_update_w(const int64_t *__restrict__ indptr,
                           const int32_t *__restrict__ idx,
                           const int32_t *__restrict__ w, int32_t *cnt,
                           const long long *__restrict__ words, long long m,
                           int delta) {
  long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  long long nwarps = ((long long)gridDim.x * blockDim.x) >> 5;
  int lane = threadIdx.x & 31;
  for (long long i = warp; i < m; i += nwarps) {
    long long r = words[i];
    long long s0 = indptr[r], s1 = indptr[r + 1];
    for (long long j = s0 + lane; j < s1; j += 32)
      atomicAdd(&cnt[idx[j]], delta * w[j]);
  }
}

__global__ void k_count_deficit_w(const int32_t *cnt, const int32_t *uw,
                                  int mu, long long N, long long *out) {
  __shared__ long long red[THREADS];
  long long gs = (long long)gridDim.x * blockDim.x;
  long long acc = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs) {
    int32_t c = cnt[i];
    if (c < mu) acc += (long long)uw[i] * (mu - c);
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd((unsigned long long *)out, (unsigned long long)red[0]);
}

__global__ void k_count_eq(const int32_t *cnt, int target, long long N,
                           long long *out) {
  __shared__ long long red[THREADS];
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
  if (threadIdx.x == 0)
    atomicAdd((unsigned long long *)out, (unsigned long long)red[0]);
}

__global__ void k_count_deficit(const int32_t *cnt, int mu, long long N,
                                long long *out) {
  __shared__ long long red[THREADS];
  long long gs = (long long)gridDim.x * blockDim.x;
  long long acc = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs) {
    int32_t c = cnt[i];
    if (c < mu) acc += mu - c;
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd((unsigned long long *)out, (unsigned long long)red[0]);
}

__global__ void k_max_i32(const int32_t *A, long long N, long long *out) {
  __shared__ int32_t red[THREADS];
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

// positive v binned at (v-1)*nbins/vmax, clipped to the last bin
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

// threshold compaction: atomic slot counter, unordered, the count keeps
// growing past cap while only cap entries are stored
__global__ void k_extract(const int32_t *A, long long N, int32_t thr,
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

__global__ void k_read_at(const int32_t *A, const long long *idx, long long k,
                          int32_t *out) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) out[i] = A[idx[i]];
}

int upload_words(const long long *idx, long long m) {
  if (m < 0) return -1;
  if (m == 0) return 0;
  int rc = ensure_words(m);
  if (rc) return rc;
  CK(cudaMemcpy(g_words, idx, (size_t)m * 8, cudaMemcpyHostToDevice));
  return 0;
}

int scalar_reduce(long long *r) {
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(r, g_scalar, 8, cudaMemcpyDeviceToHost));
  return 0;
}

}  // namespace

extern "C" {

int sparse_abi_version(void) { return 2; }

long long sparse_dev_bytes(void) { return g_bytes_dev; }
long long sparse_host_bytes(void) { return g_bytes_host; }

int sparse_free(void) {
  if (!g_live) return 0;
  free_inc();
  cudaFree(g_cnt); cudaFree(g_map); cudaFree(g_scalar); cudaFree(g_hist);
  cudaFree(g_xidx); cudaFree(g_xval);
  if (g_words) { cudaFree(g_words); cudaFree(g_out); }
  g_cnt = g_map = nullptr; g_scalar = nullptr; g_hist = nullptr;
  g_xidx = nullptr; g_xval = nullptr; g_words = nullptr; g_out = nullptr;
  g_wcap = 0; g_xcap = 0; g_mu = 1;
  g_bytes_dev = g_bytes_host = 0;
  g_live = false;
  return 0;
}

// inc: (n_cand x d) row-major int32, entries in [0, n_univ).
// inc_mode: 0 = HBM, 1 = pinned CPU memory (ATS; managed off GH200).
// bytes_dev / bytes_host: allocation totals, for the caller's accounting.
int sparse_init(long long n_cand, int d, long long n_univ, const int32_t *inc,
                int inc_mode, int extract_cap, long long *bytes_dev,
                long long *bytes_host) {
  if (g_live) return -1;
  if (n_cand < 1 || d < 1 || n_univ < 1 || n_univ > 0x7fffffffLL) return -2;
  if (inc_mode < 0 || inc_mode > 1 || extract_cap < 1) return -2;
  g_ncand = n_cand; g_d = d; g_nuniv = n_univ; g_mu = 1;
  g_bytes_dev = g_bytes_host = 0;
  CK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0));
  if (g_sms < 1) g_sms = 1;
  g_live = true;                       // so sparse_free releases on failure
  size_t ib = (size_t)n_cand * (size_t)d * 4;
  int rc = alloc_inc(ib, inc_mode, inc);
  if (rc) { sparse_free(); return rc; }
  CK(cudaMalloc((void **)&g_cnt, (size_t)n_univ * 4));
  CK(cudaMemset(g_cnt, 0, (size_t)n_univ * 4));
  CK(cudaMalloc((void **)&g_map, (size_t)n_cand * 4));
  CK(cudaMemset(g_map, 0, (size_t)n_cand * 4));
  CK(cudaMalloc((void **)&g_scalar, 8));
  CK(cudaMalloc((void **)&g_hist, (size_t)HIST_MAX * 4));
  g_xcap = extract_cap;
  CK(cudaMalloc((void **)&g_xidx, (size_t)g_xcap * 8));
  CK(cudaMalloc((void **)&g_xval, (size_t)g_xcap * 4));
  g_bytes_dev += (n_univ + n_cand) * 4 + 8 + HIST_MAX * 4 + (long long)g_xcap * 12;
  if (bytes_dev) *bytes_dev = g_bytes_dev;
  if (bytes_host) *bytes_host = g_bytes_host;
  return 0;
}

// Weighted CSR cell: row c covers indices[indptr[c]..indptr[c+1]) with
// weights w (>= 1); universe weights uw (>= 1).  indptr/indices/w are
// placed per inc_mode, uw and the fields in HBM.
int sparse_init_csr(long long n_cand, long long n_univ, long long nnz,
                    const int64_t *indptr, const int32_t *indices,
                    const int32_t *w, const int32_t *uw, int inc_mode,
                    int extract_cap, long long *bytes_dev,
                    long long *bytes_host) {
  if (g_live) return -1;
  if (n_cand < 1 || n_univ < 1 || n_univ > 0x7fffffffLL || nnz < 0) return -2;
  if (inc_mode < 0 || inc_mode > 1 || extract_cap < 1) return -2;
  g_ncand = n_cand; g_d = 0; g_nuniv = n_univ; g_nnz = nnz; g_mu = 1;
  g_bytes_dev = g_bytes_host = 0;
  CK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0));
  if (g_sms < 1) g_sms = 1;
  g_live = true;
  g_csr = true;
  int rc = alloc_arr((void **)&g_indptr, (size_t)(n_cand + 1) * 8, inc_mode,
                     indptr);
  if (rc) { sparse_free(); return rc; }
  size_t nb = (size_t)(nnz > 0 ? nnz : 1) * 4;
  rc = alloc_arr((void **)&g_inc, nb, inc_mode, indices);
  if (rc) { sparse_free(); return rc; }
  rc = alloc_arr((void **)&g_w, nb, inc_mode, w);
  if (rc) { sparse_free(); return rc; }
  CK(cudaMalloc((void **)&g_uw, (size_t)n_univ * 4));
  CK(cudaMemcpy(g_uw, uw, (size_t)n_univ * 4, cudaMemcpyHostToDevice));
  g_bytes_dev += n_univ * 4;
  CK(cudaMalloc((void **)&g_cnt, (size_t)n_univ * 4));
  CK(cudaMemset(g_cnt, 0, (size_t)n_univ * 4));
  CK(cudaMalloc((void **)&g_map, (size_t)n_cand * 4));
  CK(cudaMemset(g_map, 0, (size_t)n_cand * 4));
  CK(cudaMalloc((void **)&g_scalar, 8));
  CK(cudaMalloc((void **)&g_hist, (size_t)HIST_MAX * 4));
  g_xcap = extract_cap;
  CK(cudaMalloc((void **)&g_xidx, (size_t)g_xcap * 8));
  CK(cudaMalloc((void **)&g_xval, (size_t)g_xcap * 4));
  g_bytes_dev += (n_univ + n_cand) * 4 + 8 + HIST_MAX * 4 + (long long)g_xcap * 12;
  if (bytes_dev) *bytes_dev = g_bytes_dev;
  if (bytes_host) *bytes_host = g_bytes_host;
  return 0;
}

int sparse_inc_mode(void) { return g_live ? g_inc_mode : -1; }
int sparse_is_csr(void) { return g_live && g_csr; }

int sparse_set_mu(int mu) {
  if (mu < 1) return -1;
  g_mu = mu;
  return 0;
}

int sparse_get_mu(void) { return g_mu; }

// cnt[u] += delta for every element of every listed candidate's row
int sparse_ball_update(const long long *idx, long long m, int delta) {
  if (!g_live) return -1;
  int rc = upload_words(idx, m);
  if (rc) return rc;
  if (m == 0) return 0;
  if (g_csr)
    k_update_w<<<grid_for(m * 32), THREADS>>>(g_indptr, g_inc, g_w, g_cnt,
                                              g_words, m, delta);
  else
    k_update<<<grid_for(m * g_d), THREADS>>>(g_inc, g_cnt, g_words, m, g_d,
                                             delta);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  return 0;
}

// cnt = 0, then the solution's rows scattered in
int sparse_recount(const long long *idx, long long m) {
  if (!g_live) return -1;
  CK(cudaMemset(g_cnt, 0, (size_t)g_nuniv * 4));
  return sparse_ball_update(idx, m, 1);
}

// map[c] = #{u in row c : role(cnt[u], target)} for every candidate;
// target 0 = the gain field, 1 = the loss field.
int sparse_field(int target) {
  if (!g_live) return -1;
  if (g_csr)
    k_field_w<<<grid_for(g_ncand * 32), THREADS>>>(g_indptr, g_inc, g_w, g_uw,
                                                   g_cnt, g_map, g_ncand,
                                                   target, g_mu);
  else
    k_field<<<grid_for(g_ncand * 32), THREADS>>>(g_inc, g_cnt, g_map, g_ncand,
                                                 g_d, target, g_mu);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  return 0;
}

int sparse_ball_gather(const long long *idx, long long m, int target,
                       int32_t *out) {
  if (!g_live) return -1;
  int rc = upload_words(idx, m);
  if (rc) return rc;
  if (m == 0) return 0;
  if (g_csr)
    k_gather_w<<<grid_for(m * 32), THREADS>>>(g_indptr, g_inc, g_w, g_uw,
                                              g_cnt, g_words, m, target, g_mu,
                                              g_out);
  else
    k_gather<<<grid_for(m * 32), THREADS>>>(g_inc, g_cnt, g_words, m, g_d,
                                            target, g_mu, g_out);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out, g_out, (size_t)m * 4, cudaMemcpyDeviceToHost));
  return 0;
}

long long sparse_count_eq(int target) {
  if (!g_live) return -1;
  CK(cudaMemset(g_scalar, 0, 8));
  k_count_eq<<<grid_for(g_nuniv), THREADS>>>(g_cnt, target, g_nuniv, g_scalar);
  CK(cudaGetLastError());
  long long r;
  int rc = scalar_reduce(&r);
  return rc ? rc : r;
}

// sum_u max(0, mu - cnt[u]); 0 iff feasible
long long sparse_count_deficit(void) {
  if (!g_live) return -1;
  CK(cudaMemset(g_scalar, 0, 8));
  if (g_csr)
    k_count_deficit_w<<<grid_for(g_nuniv), THREADS>>>(g_cnt, g_uw, g_mu,
                                                      g_nuniv, g_scalar);
  else
    k_count_deficit<<<grid_for(g_nuniv), THREADS>>>(g_cnt, g_mu, g_nuniv,
                                                    g_scalar);
  CK(cudaGetLastError());
  long long r;
  int rc = scalar_reduce(&r);
  return rc ? rc : r;
}

int sparse_map_max(int32_t *out) {
  if (!g_live) return -1;
  CK(cudaMemset(g_scalar, 0, 8));
  k_max_i32<<<grid_for(g_ncand), THREADS>>>(g_map, g_ncand, g_scalar);
  CK(cudaGetLastError());
  long long r;
  int rc = scalar_reduce(&r);
  if (rc) return rc;
  *out = (int32_t)r;
  return 0;
}

int sparse_map_hist(int nbins, int32_t vmax, int32_t *out) {
  if (!g_live) return -1;
  if (nbins < 1 || nbins > HIST_MAX || vmax < 1) return -2;
  CK(cudaMemset(g_hist, 0, (size_t)nbins * 4));
  k_hist<<<grid_for(g_ncand), THREADS, nbins * 4>>>(g_map, g_ncand, nbins,
                                                    vmax, g_hist);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out, g_hist, (size_t)nbins * 4, cudaMemcpyDeviceToHost));
  return 0;
}

// candidates with map >= thr; returns the number found (may exceed cap;
// only min(found, cap) entries are stored, in no particular order)
long long sparse_map_extract(int32_t thr, int64_t *idx, int32_t *val,
                             int cap) {
  if (!g_live) return -1;
  if (cap > g_xcap) cap = g_xcap;
  if (cap < 0) return -2;
  CK(cudaMemset(g_scalar, 0, 8));
  k_extract<<<grid_for(g_ncand), THREADS>>>(g_map, g_ncand, thr, g_xidx,
                                            g_xval, cap, g_scalar);
  CK(cudaGetLastError());
  long long found;
  int rc = scalar_reduce(&found);
  if (rc) return rc;
  long long m = found < cap ? found : cap;
  if (m > 0) {
    CK(cudaMemcpy(idx, g_xidx, (size_t)m * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(val, g_xval, (size_t)m * 4, cudaMemcpyDeviceToHost));
  }
  return found;
}

int sparse_map_read_at(const long long *idx, long long k, int32_t *out) {
  if (!g_live) return -1;
  int rc = upload_words(idx, k);
  if (rc) return rc;
  if (k == 0) return 0;
  k_read_at<<<(int)((k + THREADS - 1) / THREADS), THREADS>>>(g_map, g_words,
                                                             k, g_out);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out, g_out, (size_t)k * 4, cudaMemcpyDeviceToHost));
  return 0;
}

// a range of the multiplicity field (over the UNIVERSE), as int32
int sparse_read_cnt(long long off, long long len, int32_t *out) {
  if (!g_live) return -1;
  if (off < 0 || len < 0 || off + len > g_nuniv) return -2;
  if (len == 0) return 0;
  CK(cudaMemcpy(out, g_cnt + off, (size_t)len * 4, cudaMemcpyDeviceToHost));
  return 0;
}

}  // extern "C"

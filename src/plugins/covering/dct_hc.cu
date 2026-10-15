// dct_hc.cu — C ABI of the dctcov core (working name), the host-coherent,
// problem-agnostic evolution of dct.cu.
//
// Layering (see PLUGIN.md for the contract):
//   dctcore_core.cuh     — problem-blind state core: field allocation with
//                          HBM/LPDDR placement (GH200 ATS / managed), fills,
//                          support-set walks (incremental exact update /
//                          gather over packed shift-pattern tables), the
//                          owner-trick losspass, reductions, extraction.
//   dctcore_hamming.cuh  — plugin 1: q-ary / mixed-radix covering codes
//                          (exact Hamming distance-count layer transform).
//   dctcore_torus.cuh    — plugin 2: dominating sets on torus grids under
//                          Chebyshev balls (circular window-sum transform).
//   dctcore_lee.cuh      — plugin 3: covering codes under the Lee metric
//                          (grade-truncated circulant-kernel transform).
//   dctcore_mu.cuh       — mu-fold multiple covering (K^mu): predicate twins
//                          of the three cnt-reading core kernels.
//   dctcore_grid.cuh     — plugin 4: dominating sets of the OPEN grid under
//                          L-infinity balls (clamped window sum + clamped
//                          support walk; not shift invariant).
//   dctcore_fast.cuh     — v0.3: set-side gather, reporting updates, swap
//                          evaluation, vectorised reductions.
//
// A problem plugin supplies ONE thing to this TU: an axis-pass driver that
// turns A0 = indicator(S) into the coverage field.  Everything else —
// including the search layer above (exact lazy greedy + LNS, see
// coldcore/search.py) — is shared and problem-blind.
//
// ABI: dct_init/dct_init2 keep the historical signatures (Hamming, all-HBM /
// mode-selected).  dct_init3 is the generic entry (problem id + axis sizes).
// No search claim rests on these counters: covers are re-verified from the
// written file by an independent host-side checker.
//
// Build: nvcc -O3 -arch=sm_90 --shared -Xcompiler -fPIC -o libdct_hc.so dct_hc.cu

#include <unordered_map>
#include <climits>
#include "dctcore_core.cuh"
#include "dctcore_hamming.cuh"
#include "dctcore_torus.cuh"
#include "dctcore_lee.cuh"
#include "dctcore_mu.cuh"
#include "dctcore_grid.cuh"
#include "dctcore_fast.cuh"

// A global occurrence limit guarantees that packed uint16 counters cannot
// overflow. Track the represented multiset to reject underflow before launch.
static constexpr int MAX_CODE = 65535;
static std::unordered_map<long long, int> uploaded, active;
static int active_size = 0;
static bool valid_words(const long long *words, int n) {
  if (n < 0 || n > MAX_CODE || (n && !words)) return false;
  for (int i = 0; i < n; ++i)
    if (words[i] < 0 || words[i] >= g_space) return false;
  return true;
}

// ---- host-call overhead (v0.3) ---------------------------------------------
// Every entry point runs on the legacy default stream (stream 0), so device
// work is ordered exactly as the calls were made.  Inputs and outputs go
// through PINNED host staging buffers with cudaMemcpyAsync and a single
// cudaStreamSynchronize per call that returns data; pageable cudaMemcpy
// (an internal bounce copy plus an implicit sync) is gone from the hot calls.
// ball_update of small batches does not synchronise at all: the words travel
// as a kernel PARAMETER (k_set_words), so no host buffer outlives the call,
// and the next call that reads anything waits on the stream.
#define NV_WORDS 64              // words per by-value staging launch
#define ASYNC_UPDATE_MAX 1024    // larger update batches stage + sync
struct WordsArg { long long w[NV_WORDS]; };
static long long *g_h_words;     // pinned staging (g_codecap words)
static int32_t *g_h_out;         // pinned staging (g_codecap int32)
static long long *g_h_scalar;    // pinned scalar

__global__ void k_set_words(WordsArg a, int n, long long *dst) {
  int i = threadIdx.x;
  if (i < n) dst[i] = a.w[i];
}

// copy nb words to g_d_words without any host-side wait (kernel parameters)
static int stage_words_param(const long long *words, int nb) {
  WordsArg a;
  for (int off = 0; off < nb; off += NV_WORDS) {
    int k = nb - off < NV_WORDS ? nb - off : NV_WORDS;
    memcpy(a.w, words + off, (size_t)k * 8);
    k_set_words<<<1, NV_WORDS>>>(a, k, g_d_words + off);
  }
  CK(cudaGetLastError());
  return 0;
}

// copy nb words to g_d_words through the pinned buffer; the caller MUST
// synchronise the stream before returning (the copy reads g_h_words late)
static int stage_words_pinned(const long long *words, int nb) {
  memcpy(g_h_words, words, (size_t)nb * 8);
  CK(cudaMemcpyAsync(g_d_words, g_h_words, (size_t)nb * 8,
                     cudaMemcpyHostToDevice, 0));
  return 0;
}

// grid size for a grid-stride pass over N items: enough blocks for ~8 items
// per thread, capped at g_blocks.  Results are integer sums, max or atomic
// slot claims, so they do not depend on the grid size.
static int pass_blocks(long long N) {
  long long b = (N + 256LL * 8 - 1) / (256LL * 8);
  if (b < 1) b = 1;
  if (b > g_blocks) b = g_blocks;
  return (int)b;
}

// 16-byte aligned field (all three allocators give page/256-B alignment;
// checked anyway, the scalar kernels are the fallback)
static bool vec_ok(const void *p) { return ((uintptr_t)p & 15) == 0; }

// read one device scalar written by the preceding launches
static int fetch_scalar(long long *r) {
  CK(cudaMemcpyAsync(g_h_scalar, g_scalar, 8, cudaMemcpyDeviceToHost, 0));
  CK(cudaStreamSynchronize(0));
  *r = g_h_scalar[0];
  return 0;
}

// ---- set-side gather state (dctcore_fast.cuh) ------------------------------
static bool g_set_ok;               // support table proven == metric ball
static uint64_t *g_zbuf;            // packed digits of the predicate set Z
static long long g_zcap;
static unsigned long long *g_zcount;
static long long g_cnt_ver;         // bumped on every cnt mutation
static long long g_plus_words;      // cumulative words placed (+1 updates)
static int g_set_policy = 1;
static int g_legacy_tf = 0;
static long long g_zcap_max;        // growth limit of g_zbuf (entries)
static long long g_lcap;            // swap_eval list capacity (entries)
static long long g_set_ratio = 2;   // DCT_SET_RATIO: set side while |Z| <= ratio*|B|         // DCT_LEGACY_TRANSFORM=1: v0.2 hamming path        // DCT_SET_GATHER: 0 off, 1 auto, 2 force
static long long g_set_min;         // auto: min nb*|B| to try the set side
// cached list: valid while (key, version) match
static int z_mode = -1, z_target, z_mu;
static long long z_ver = -1, z_n;
// last overflow of a shrink-only predicate: F words at g_plus_words = z_ovf_plus
static int o_mode = -1, o_target, o_mu;
static long long o_F, o_plus;

static long long g_stat[8];
static uint64_t *g_lbuf;            // swap_eval: words at cnt == mu per drop
static int32_t *g_ldrop;
static long long g_dd_last;         // last value of the running transition
                                    // counter (g_scalar[2], never reset)
static void *g_d_swap, *g_h_swap;   // swap_eval packed inputs
static void *g_d_sres, *g_h_sres;   // swap_eval packed results        // dct_stats counters (see there)
static void cnt_changed(void) { g_cnt_ver++; }
static void cnt_rebuilt(void) { g_cnt_ver++; o_mode = -1; }

// Is the v1 table exactly the metric ball of the problem?  Every entry must
// change distinct in-range positions by admissible deltas within radius R, and
// the entry count must equal the ball's cardinality (per-axis weight
// polynomial, truncated at R).  With distinct entries (v1 precondition) the
// table then IS the ball, and dist(x, c) <= R is the same relation the walk
// enumerates.
static bool table_is_ball(const uint64_t *pat, long long len) {
  int R = g_R, n = g_n;
  auto wt = [&](int i, int d) -> int {   // -1 = inadmissible delta
    int a = g_ax[i];
    if (d < 1 || d >= a) return -1;
    int c = d < a - d ? d : a - d;       // circular distance
    switch (g_prob) {
      case PROB_HAMMING: return 1;
      case PROB_LEE: return c <= R ? c : -1;
      default: return c <= R ? 0 : -1;   // torus / grid: L-inf, no coupling
    }
  };
  for (long long t = 0; t < len; t++) {
    uint64_t p = pat[t];
    unsigned seen = 0;
    int w = 0;
    for (int k = 0; k < MAXPAT; k++) {
      unsigned b = (unsigned)(p >> (8 * k)) & 255u;
      if (!b) { if (p >> (8 * k)) return false; break; }
      int pos = (int)(b >> 4), d = (int)(b & 15u);
      if (pos >= n || (seen >> pos) & 1u) return false;
      seen |= 1u << pos;
      int e = wt(pos, d);
      if (e < 0) return false;
      w += e;
    }
    if (w > R) return false;
  }
  // cardinality: coefficients 0..R of prod_i (1 + sum_d t^wt(i,d))
  long double poly[MAXR + 2] = {1};
  int W = (g_prob == PROB_HAMMING || g_prob == PROB_LEE) ? R : 0;
  for (int i = 0; i < n; i++) {
    long double nx[MAXR + 2] = {0};
    for (int w = 0; w <= W; w++) {
      if (poly[w] == 0) continue;
      nx[w] += poly[w];
      for (int d = 1; d < g_ax[i]; d++) {
        int e = wt(i, d);
        if (e >= 0 && w + e <= W) nx[w + e] += poly[w];
      }
    }
    for (int w = 0; w <= W; w++) poly[w] = nx[w];
  }
  long double tot = 0;
  for (int w = 0; w <= W; w++) tot += poly[w];
  return tot == (long double)len;
}

extern "C" {

long long dct_host_bytes(void) { return g_bytes_host; }

// ---- mu-fold multiple covering (optional v1 extension, feature-tested) ----
// Sets the covering multiplicity for every subsequent transform / gather /
// deficit query; mu = 1 (the default, and the state after dct_free) is the
// ordinary covering problem and takes the legacy code paths verbatim.
int dct_set_mu(int mu) {
  if (mu < 1 || mu > 65535) return -1;
  g_mu = mu;
  CK(cudaMemcpyToSymbol(c_mu, &g_mu, 4));
  cnt_rebuilt();                  // predicates changed meaning
  return 0;
}

int dct_get_mu(void) { return g_mu; }

// Generic init.  problem: PROB_HAMMING | PROB_TORUS_LINF | PROB_LEE.
// ax[n]: axis sizes.
// ballpat/balllen: packed support-set table (byte k = pos<<4 | delta, delta
// in 1..ax_pos-1, zero byte terminates; entry 0 = empty pattern = center).
// layers_mode/cnt_mode: MEM_DEV | MEM_ATS | MEM_MANAGED per array group.
// use_owner: allocate owner[] (4 B/word, same mode as cnt).
int dct_init3(int problem, int n, const int *ax, int R,
              const uint64_t *ballpat, long long balllen, int extract_cap,
              long long *bytes, int layers_mode, int cnt_mode, int use_owner) {
  if (n < 2 || n > MAXN || R < 1 || !ax || !ballpat ||
      balllen < 1 || balllen > INT_MAX || extract_cap < 1) return -1;
  if (problem == PROB_HAMMING) {
    if (R > MAXR) return -1;
    for (int i = 0; i < n; i++)
      if (ax[i] < 2 || ax[i] > MAXQH) return -1;
    g_nlayers = R + 1;
  } else if (problem == PROB_TORUS_LINF) {
    for (int i = 0; i < n; i++)
      if (ax[i] < 3 || ax[i] > MAXAX || R > (ax[i] - 1) / 2) return -1;
    if (n > MAXPAT) return -1;      // support patterns cap at 8 positions
    g_nlayers = 1;
  } else if (problem == PROB_LEE) {
    if (R > MAXR) return -1;
    for (int i = 0; i < n; i++)
      if (ax[i] < 2 || ax[i] > MAXAX) return -1;
    g_nlayers = R + 1;
  } else if (problem == PROB_GRID_LINF) {
    // open board: 2R+1 <= ax_j is NOT required (the window clamps)
    for (int i = 0; i < n; i++)
      if (ax[i] < 2 || ax[i] > MAXAX) return -1;
    if (n > MAXPAT) return -1;      // support patterns cap at 8 positions
    g_nlayers = 1;
  } else {
    return -1;
  }
  g_prob = problem; g_n = n; g_R = R;
  g_homog = 1;
  for (int i = 0; i < n; i++) {
    g_ax[i] = ax[i];
    if (ax[i] != ax[0]) g_homog = 0;
  }
  g_space = 1;
  for (int i = 0; i < n; i++) {
    if (g_space > (1LL << 40) / ax[i]) return -2;
    g_space *= ax[i];
  }
  if (g_space > (1LL << 40)) return -2;
  g_layers_mode = layers_mode; g_cnt_mode = cnt_mode; g_use_owner = use_owner;
  g_fmt = 0;
  g_bytes_dev = g_bytes_host = 0;
  const char *bl = getenv("DCT_BLOCKS");
  g_blocks = bl ? atoi(bl) : 4096;
  if (g_blocks < 256) g_blocks = 256;
  g_powq[0] = 1;
  for (int i = 1; i <= n; i++) g_powq[i] = g_powq[i - 1] * ax[i - 1];
  CK(cudaMemcpyToSymbol(c_powq, g_powq, sizeof(g_powq)));
  CK(cudaMemcpyToSymbol(c_ax, g_ax, sizeof(g_ax)));
  CK(cudaMemcpyToSymbol(c_n, &n, 4));
  CK(cudaMemcpyToSymbol(c_R, &R, 4));
  CK(cudaMemcpyToSymbol(c_space, &g_space, 8));
  for (int d = 0; d < g_nlayers; d++)
    if (alloc_buf((void **)&g_A[d], (size_t)g_space * 4, layers_mode))
      return -3;
  for (int d = g_nlayers; d <= MAXR; d++) g_A[d] = nullptr;
  if (alloc_buf((void **)&g_cnt, (size_t)((g_space + 1) & ~1LL) * 2, cnt_mode)) return -4;
  // zero the padding counter too: the packed 32-bit atomics read it
  if (cnt_mode == MEM_DEV)
    CK(cudaMemset(g_cnt, 0, (size_t)((g_space + 1) & ~1LL) * 2));
  g_owner = nullptr;
  if (use_owner) {
    if (alloc_buf((void **)&g_owner, (size_t)g_space * 4, cnt_mode))
      return -5;
    // defined contents (atomicExch reads the old tag; host modes are
    // already zeroed by alloc_buf)
    if (cnt_mode == MEM_DEV) CK(cudaMemset(g_owner, 0, (size_t)g_space * 4));
  }
  g_balllen = balllen;
  CK(cudaMalloc(&g_ballpat, (size_t)balllen * 8));
  g_bytes_dev += balllen * 8;
  CK(cudaMemcpy(g_ballpat, ballpat, (size_t)balllen * 8,
                cudaMemcpyHostToDevice));
  g_tbl2_mask = nullptr; g_tbl2_offs = nullptr;
  {
    const char *sp = getenv("DCT_SET_GATHER");
    g_set_policy = sp ? atoi(sp) : 1;
    g_timing = -1;                  // re-read DCT_TIMING / DCT_PAIR_PASSES
    const char *lt = getenv("DCT_LEGACY_TRANSFORM");
    g_legacy_tf = lt ? atoi(lt) : 0;
    const char *sm = getenv("DCT_SET_MIN");
    g_set_min = sm ? atoll(sm) : -1;
    g_set_ok = g_set_policy != 0 && n <= 16 && table_is_ball(ballpat, balllen);
    g_zcap = balllen / 4;
    if (g_zcap < 4096) g_zcap = 4096;
    if (g_zcap > (1LL << 22)) g_zcap = 1LL << 22;
    g_lcap = g_zcap;                  // swap_eval list: fixed size
    g_zcap_max = g_space < (1LL << 27) ? g_space : (1LL << 27);   // <= 1 GiB
    if (g_zcap_max < g_zcap) g_zcap_max = g_zcap;
    const char *sr = getenv("DCT_SET_RATIO");
    g_set_ratio = sr ? atoll(sr) : 2;
    g_zbuf = nullptr; g_lbuf = nullptr; g_ldrop = nullptr;
    if (g_set_ok) {
      CK(cudaMalloc(&g_zbuf, (size_t)g_zcap * 8));
      CK(cudaMalloc(&g_lbuf, (size_t)g_lcap * 8));
      CK(cudaMalloc(&g_ldrop, (size_t)g_lcap * 4));
      g_bytes_dev += g_zcap * 20;
    }
    z_mode = o_mode = -1;
    g_cnt_ver = 0; g_plus_words = 0;
    memset(g_stat, 0, sizeof(g_stat));
  }
  g_codecap = 1 << 16;
  CK(cudaMalloc(&g_codeidx, (size_t)g_codecap * 8));
  CK(cudaMalloc(&g_d_words, (size_t)g_codecap * 8));
  CK(cudaMalloc(&g_d_out, (size_t)g_codecap * 4));
  CK(cudaMalloc(&g_d_loss, (size_t)g_codecap * 4));
  g_bytes_dev += (size_t)g_codecap * 24;
  if (g_h_words) { cudaFreeHost(g_h_words); g_h_words = 0; }
  if (g_h_out) { cudaFreeHost(g_h_out); g_h_out = 0; }
  if (g_h_scalar) { cudaFreeHost(g_h_scalar); g_h_scalar = 0; }
  CK(cudaHostAlloc((void **)&g_h_words, (size_t)g_codecap * 8,
                   cudaHostAllocDefault));
  CK(cudaHostAlloc((void **)&g_h_out, (size_t)g_codecap * 4,
                   cudaHostAllocDefault));
  CK(cudaHostAlloc((void **)&g_h_scalar, 64, cudaHostAllocDefault));
  g_codeM = 0;
  uploaded.clear(); active.clear(); active_size = 0;
  CK(cudaMalloc(&g_scalar, 64 * 8));
  g_zcount = (unsigned long long *)(g_scalar + 8);
  CK(cudaMemset(g_scalar, 0, 64 * 8));
  g_dd_last = 0;
  {
    // swap_eval staging: packed inputs (cand, drops, cand->drop) and the
    // packed result region (gains, losses, list count)
    size_t in_b = (size_t)g_codecap * 20, r_b = (size_t)g_codecap * 8 + 16;
    CK(cudaMalloc(&g_d_swap, in_b));
    CK(cudaMalloc(&g_d_sres, r_b));
    if (g_h_swap) { cudaFreeHost(g_h_swap); g_h_swap = 0; }
    if (g_h_sres) { cudaFreeHost(g_h_sres); g_h_sres = 0; }
    CK(cudaHostAlloc(&g_h_swap, in_b, cudaHostAllocDefault));
    CK(cudaHostAlloc(&g_h_sres, r_b, cudaHostAllocDefault));
    g_bytes_dev += in_b + r_b;
  }
  CK(cudaMalloc(&g_hist, 4096 * 4));
  g_extract_cap = extract_cap;
  CK(cudaMalloc(&g_extract_idx, (size_t)extract_cap * 4));
  CK(cudaMalloc(&g_extract_idx64, (size_t)extract_cap * 8));
  CK(cudaMalloc(&g_extract_val, (size_t)extract_cap * 4));
  g_bytes_dev += (size_t)extract_cap * 16 + 64 * 8 + 4096 * 4;
  if (bytes) *bytes = g_bytes_dev + g_bytes_host;
  return 0;
}

// historical ABIs (Hamming, homogeneous q)
int dct_init2(int q, int n, int R, const uint64_t *ballpat, long long balllen,
              int extract_cap, long long *bytes, int layers_mode, int cnt_mode,
              int use_owner) {
  int ax[MAXN];
  if (n < 2 || n > MAXN) return -1;
  for (int i = 0; i < n; i++) ax[i] = q;
  return dct_init3(PROB_HAMMING, n, ax, R, ballpat, balllen, extract_cap,
                   bytes, layers_mode, cnt_mode, use_owner);
}

int dct_init(int q, int n, int R, const uint64_t *ballpat, long long balllen,
             int extract_cap, long long *bytes) {
  return dct_init2(q, n, R, ballpat, balllen, extract_cap, bytes,
                   MEM_DEV, MEM_DEV, 0);
}

int dct_free(void) {
  cudaDeviceSynchronize();        // async updates may still be in flight
  if (g_h_words) { cudaFreeHost(g_h_words); g_h_words = 0; }
  if (g_h_out) { cudaFreeHost(g_h_out); g_h_out = 0; }
  if (g_h_scalar) { cudaFreeHost(g_h_scalar); g_h_scalar = 0; }
  for (int d = 0; d <= MAXR; d++)
    if (g_A[d]) { free_buf(g_A[d], g_layers_mode); g_A[d] = 0; }
  if (g_cnt) { free_buf(g_cnt, g_cnt_mode); g_cnt = 0; }
  if (g_owner) { free_buf(g_owner, g_cnt_mode); g_owner = 0; }
  if (g_ballpat) { cudaFree(g_ballpat); g_ballpat = 0; }
  if (g_tbl2_mask) { cudaFree(g_tbl2_mask); g_tbl2_mask = 0; }
  if (g_tbl2_offs) { cudaFree(g_tbl2_offs); g_tbl2_offs = 0; }
  if (g_codeidx) { cudaFree(g_codeidx); g_codeidx = 0; }
  if (g_d_words) { cudaFree(g_d_words); g_d_words = 0; }
  if (g_d_out) { cudaFree(g_d_out); g_d_out = 0; }
  if (g_d_loss) { cudaFree(g_d_loss); g_d_loss = 0; }
  if (g_scalar) { cudaFree(g_scalar); g_scalar = 0; }
  if (g_zbuf) { cudaFree(g_zbuf); g_zbuf = 0; }
  if (g_lbuf) { cudaFree(g_lbuf); g_lbuf = 0; }
  if (g_ldrop) { cudaFree(g_ldrop); g_ldrop = 0; }
  if (g_d_swap) { cudaFree(g_d_swap); g_d_swap = 0; }
  if (g_d_sres) { cudaFree(g_d_sres); g_d_sres = 0; }
  if (g_h_swap) { cudaFreeHost(g_h_swap); g_h_swap = 0; }
  if (g_h_sres) { cudaFreeHost(g_h_sres); g_h_sres = 0; }
  g_set_ok = false; z_mode = o_mode = -1;
  if (g_hist) { cudaFree(g_hist); g_hist = 0; }
  if (g_extract_idx) { cudaFree(g_extract_idx); g_extract_idx = 0; }
  if (g_extract_idx64) { cudaFree(g_extract_idx64); g_extract_idx64 = 0; }
  if (g_extract_val) { cudaFree(g_extract_val); g_extract_val = 0; }
  uploaded.clear(); active.clear(); active_size = 0;
  g_codeM = 0;
  g_mu = 1;                       // singleton: never leak mu into the next cell
  cudaMemcpyToSymbol(c_mu, &g_mu, 4);
  cudaDeviceSynchronize();
  return 0;
}

// upload the ox-format table (must enumerate the same support set, any order)
int dct_set_tbl2(const uint16_t *mask, const uint64_t *offs, long long len) {
  if (len != g_balllen) return -1;
  if (!g_tbl2_mask) {
    CK(cudaMalloc(&g_tbl2_mask, (size_t)len * 2));
    CK(cudaMalloc(&g_tbl2_offs, (size_t)len * 8));
    g_bytes_dev += len * 10;
  }
  CK(cudaMemcpy(g_tbl2_mask, mask, (size_t)len * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(g_tbl2_offs, offs, (size_t)len * 8, cudaMemcpyHostToDevice));
  return 0;
}

int dct_use_fmt(int fmt) {
  if (fmt == 1 && !g_tbl2_mask) return -1;
  if (fmt < 0 || fmt > 1) return -1;
  // the ox walk is circulant; the open grid needs the clamped walk
  if (fmt == 1 && g_prob == PROB_GRID_LINF) return -1;
  g_fmt = fmt;
  return 0;
}

int dct_set_code(const long long *idx, int M) {
  if (!valid_words(idx, M)) return -1;
  CK(cudaMemcpy(g_codeidx, idx, (size_t)M * 8, cudaMemcpyHostToDevice));
  g_codeM = M;
  uploaded.clear();
  for (int i = 0; i < M; ++i) ++uploaded[idx[i]];
  return 0;
}

// init_mode: 0 = code multiset, 1 = [cnt==0], 2 = [cnt==1]
//            (mu > 1: 1 = [cnt<mu] deficient, 2 = [1<=cnt<=mu] critical —
//             the same two indicators, generalized; see dctcore_mu.cuh)
// out_mode:  1 = sum -> A0 (gain/loss map), 2 = exact sum -> cnt (code multiset input only)
int dct_transform(int init_mode, int out_mode) {
  if (init_mode < 0 || init_mode > 2 || out_mode < 1 || out_mode > 2 ||
      (out_mode == 2 && init_mode != 0)) return -1;
  long long N = g_space;
  int blocks = g_blocks, threads = 256;
  if (g_prob == PROB_HAMMING && !g_legacy_tf) {
    // v0.3 path: no layer fills, indicator init fused into pass 0
    int fuse = init_mode != 0, pm = MUP_EQ, pt = 0;
    if (init_mode == 0) {
      k_fill32<<<pass_blocks(N), threads>>>(g_A[0], 0, N);
      if (g_codeM > 0)
        k_scatter_code<<<(g_codeM + 255) / 256, 256>>>(g_A[0], g_codeidx,
                                                       g_codeM);
    } else if (g_mu == 1) {
      pt = init_mode == 1 ? 0 : 1;
    } else {
      pm = init_mode == 1 ? MUP_DEF : MUP_CRIT;
    }
    int rc = hamming_axes2(out_mode, fuse, pm, pt);
    if (rc) return rc;
    CK(cudaStreamSynchronize(0));
    if (out_mode == 2) {
      active = uploaded; active_size = g_codeM;
      cnt_rebuilt();
    }
    return 0;
  }
  for (int d = 1; d < g_nlayers; d++)
    k_fill32<<<blocks, threads>>>(g_A[d], 0, N);
  if (init_mode == 0) {
    k_fill32<<<blocks, threads>>>(g_A[0], 0, N);
    if (g_codeM > 0)
      k_scatter_code<<<(g_codeM + 255) / 256, 256>>>(g_A[0], g_codeidx,
                                                     g_codeM);
  } else if (g_mu == 1) {
    k_init_from_cnt<<<blocks, threads>>>(g_A[0], g_cnt,
                                         init_mode == 1 ? 0 : 1, N);
  } else if (init_mode == 1) {
    k_init_from_cnt_mu<MUP_DEF><<<blocks, threads>>>(g_A[0], g_cnt, N);
  } else {
    k_init_from_cnt_mu<MUP_CRIT><<<blocks, threads>>>(g_A[0], g_cnt, N);
  }
  int rc = (g_prob == PROB_HAMMING)
               ? hamming_axes(out_mode, blocks, threads)
               : (g_prob == PROB_LEE)
                     ? lee_axes(out_mode, blocks, threads)
                     : (g_prob == PROB_GRID_LINF)
                           ? grid_axes(out_mode, blocks, threads)
                           : torus_axes(out_mode, blocks, threads);
  if (rc) return rc;
  CK(cudaDeviceSynchronize());
  if (out_mode == 2) {
    active = uploaded; active_size = g_codeM;
    cnt_rebuilt();
  }
  return 0;
}

// split blocks per word so single-word updates still fill the GPU
static int pick_split(int nwords) {
  long long cover = (g_balllen + 255) / 256;   // blocks for 1 iter/thread
  long long split = 8192 / (nwords > 0 ? nwords : 1);
  if (split < 1) split = 1;
  if (split > cover) split = cover;
  if (split > 65535) split = 65535;
  return (int)split;
}

static int ball_update_inner(const long long *words, int nwords, int delta,
                             bool tag_owner) {
  if (nwords <= 0) return 0;
  cnt_changed();
  if (delta > 0) g_plus_words += nwords;
  for (int off = 0; off < nwords; off += g_codecap) {
    int nb = nwords - off > g_codecap ? g_codecap : nwords - off;
    // small batches: words as kernel parameters, no host wait at all
    bool pinned = nb > ASYNC_UPDATE_MAX;
    int src = pinned ? stage_words_pinned(words + off, nb)
                     : stage_words_param(words + off, nb);
    if (src) return src;
    int split = pick_split(nb);
    dim3 grid((unsigned)nb * split);
#define LAUNCH(FMT, OWN)                                                     \
    k_ball_update<FMT, OWN><<<grid, 256>>>(g_d_words, split, delta, g_cnt,   \
                                           g_owner, g_ballpat, g_tbl2_mask,  \
                                           g_tbl2_offs, g_balllen)
#define LAUNCH_GRID(OWN)                                                     \
    k_ball_update_grid<OWN><<<grid, 256>>>(g_d_words, split, delta, g_cnt,   \
                                           g_owner, g_ballpat, g_balllen)
    if (g_prob == PROB_GRID_LINF) {     // clamped (boundary-aware) walk
      if (tag_owner) LAUNCH_GRID(true); else LAUNCH_GRID(false);
    }
    else if (g_fmt == 0) { if (tag_owner) LAUNCH(0, true); else LAUNCH(0, false); }
    else                 { if (tag_owner) LAUNCH(1, true); else LAUNCH(1, false); }
#undef LAUNCH_GRID
#undef LAUNCH
    CK(cudaGetLastError());
    if (pinned) CK(cudaStreamSynchronize(0));   // g_h_words reused next
  }
  return 0;
}

int dct_ball_update(const long long *words, int nwords, int delta) {
  if (!valid_words(words, nwords) || (delta != 1 && delta != -1)) return -1;
  if (delta == 1 && nwords > MAX_CODE - active_size) return -1;
  std::unordered_map<long long, int> batch;
  for (int i = 0; i < nwords; ++i) ++batch[words[i]];
  if (delta == -1) {
    for (const auto &entry : batch) {
      auto it = active.find(entry.first);
      if (it == active.end() || it->second < entry.second) return -1;
    }
  }
  int rc = ball_update_inner(words, nwords, delta, false);
  if (rc) return rc;
  for (const auto &entry : batch) {
    int &count = active[entry.first];
    count += delta * entry.second;
    if (count == 0) active.erase(entry.first);
  }
  active_size += delta * nwords;
  return 0;
}

// Grow the predicate-set buffer to hold at least n entries (<= g_zcap_max).
static bool grow_zbuf(long long n) {
  long long cap = n + n / 4;
  if (cap < 2 * g_zcap) cap = 2 * g_zcap;
  if (cap > g_zcap_max) cap = g_zcap_max;
  if (cap < n) return false;
  uint64_t *nb = nullptr;
  if (cudaMalloc(&nb, (size_t)cap * 8) != cudaSuccess) {
    cudaGetLastError();              // clear; keep the current buffer
    return false;
  }
  cudaFree(g_zbuf);
  g_bytes_dev += (cap - g_zcap) * 8;
  g_zbuf = nb;
  g_zcap = cap;
  return true;
}

// Make sure g_zbuf lists Z = {x : pred(cnt[x])} for the current cnt.
// Returns 1 when the list is valid, 0 when Z has more than lim entries (the
// caller walks instead).  The buffer grows up to lim (<= g_zcap_max).
static int ensure_zlist(int mode, int target, long long lim) {
  if (z_mode == mode && z_target == target && z_mu == g_mu &&
      z_ver == g_cnt_ver)
    return 1;
  // shrink-only predicates: a +1 update removes at most |B| words from Z,
  // so a previous overflow of F words still exceeds lim while
  // F - placed_since * |B| > lim — skip the scan that would show it.
  if (o_mode == mode && o_target == target && o_mu == g_mu &&
      o_F - (g_plus_words - o_plus) * g_balllen > lim)
    return 0;
  for (;;) {
    g_stat[2]++;
    CK(cudaMemsetAsync(g_zcount, 0, 8, 0));
    int nbk = pass_blocks(g_space / 8 + 1);
    if (mode == MUP_DEF)
      k_compact_pred<MUP_DEF><<<nbk, 256>>>(g_cnt, g_space, target, g_zbuf,
                                           g_zcap, g_zcount);
    else
      k_compact_pred<MUP_EQ><<<nbk, 256>>>(g_cnt, g_space, target, g_zbuf,
                                          g_zcap, g_zcount);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(g_h_scalar + 1, g_zcount, 8, cudaMemcpyDeviceToHost,
                       0));
    CK(cudaStreamSynchronize(0));
    long long found = g_h_scalar[1];
    if (found <= g_zcap) {
      z_mode = mode; z_target = target; z_mu = g_mu;
      z_ver = g_cnt_ver; z_n = found;
      return 1;
    }
    if (found <= lim && grow_zbuf(found)) continue;   // rescan into it
    g_stat[3]++;
    o_mode = mode; o_target = target; o_mu = g_mu;
    o_F = found; o_plus = g_plus_words;
    z_mode = -1;
    return 0;
  }
}

// Use the set side for this chunk?  Only for predicates whose set shrinks
// under placements (deficient / literal cnt==0): those are the small ones
// near a cover.  Auto policy: when the walk would read at least
// max(|X|/8, 2^21) cnt entries (DCT_SET_MIN overrides the bound).
static bool want_set(int mode, int target, int nb) {
  if (!g_set_ok || !g_zbuf || ((uintptr_t)g_cnt & 15)) return false;
  if (!(mode == MUP_DEF || (mode == MUP_EQ && target == 0))) return false;
  if (g_set_policy == 2) return true;
  long long walk = (long long)nb * g_balllen;
  long long lim = g_set_min >= 0 ? g_set_min
                  : (g_space / 8 > (1LL << 21) ? g_space / 8 : (1LL << 21));
  return walk >= lim;
}

// One gather implementation for both entry points.
// mode: MUP_EQ (literal cnt == target) | MUP_DEF | MUP_CRIT.
// Launch the gather of the nb words already in g_d_words into g_d_out
// (set side or support walk); no final synchronisation.
static int gather_launch(int mode, int target, int nb, const long long *dw,
                         int32_t *dout, bool zeroed) {
  {
    // the set side pays one listing pass (atomics and writes grow with
    // |Z|) and nb * |Z| digit compares; the walk nb * |B| random reads.
    // Measured on the K8(9,4) greedy build (mean gather, ratio 1/4/8/32):
    // 212/226/256/411 us against 236 us walk-only, so |Z| <= 2|B|.
    long long lim = g_balllen * g_set_ratio;
    if (lim > g_zcap_max) lim = g_zcap_max;
    int zs = want_set(mode, target, nb) ? ensure_zlist(mode, target, lim)
                                         : 0;
    if (zs < 0) return zs;
    g_stat[zs == 1 ? 0 : 1]++;
    if (zs == 1 && z_n > 32768) {
      g_stat[7]++;
      if (!zeroed) CK(cudaMemsetAsync(dout, 0, (size_t)nb * 4, 0));
      int ny = (nb + 31) / 32;
      long long gx = (z_n + 1023) / 1024;
      long long gxm = 4096 / ny > 1 ? 4096 / ny : 1;
      if (gx > gxm) gx = gxm;
      dim3 grid((unsigned)gx, (unsigned)ny);
#define LST(P) k_gather_set_tiled<P, 32><<<grid, 256>>>(dw, nb, g_zbuf, \
                                                      z_n, dout)
      switch (g_prob) {
        case PROB_HAMMING: LST(PROB_HAMMING); break;
        case PROB_LEE: LST(PROB_LEE); break;
        case PROB_TORUS_LINF: LST(PROB_TORUS_LINF); break;
        default: LST(PROB_GRID_LINF); break;
      }
#undef LST
    } else if (zs == 1) {
      long long split = (z_n + 2047) / 2048;
      long long smax = 8192 / nb;
      if (split > smax) split = smax;
      if (split < 1) split = 1;
      if (split > 1) if (!zeroed) CK(cudaMemsetAsync(dout, 0, (size_t)nb * 4, 0));
      dim3 grid((unsigned)(nb * split));
#define LSET(P) k_gather_set<P><<<grid, 256>>>(dw, (int)split, g_zbuf, \
                                               z_n, dout)
      switch (g_prob) {
        case PROB_HAMMING: LSET(PROB_HAMMING); break;
        case PROB_LEE: LSET(PROB_LEE); break;
        case PROB_TORUS_LINF: LSET(PROB_TORUS_LINF); break;
        default: LSET(PROB_GRID_LINF); break;
      }
#undef LSET
    } else {
      int split = pick_split(nb);
      if (split > 1) if (!zeroed) CK(cudaMemsetAsync(dout, 0, (size_t)nb * 4, 0));
      dim3 grid((unsigned)nb * split);
#define LAUNCH(FMT, MODE)                                                     \
      k_ball_gather_mu<FMT, MODE><<<grid, 256>>>(dw, split, g_cnt,     \
                                                 g_ballpat, g_tbl2_mask,      \
                                                 g_tbl2_offs, g_balllen,      \
                                                 dout)
      if (g_prob == PROB_GRID_LINF) {
        if (mode == MUP_EQ)
          k_ball_gather_grid<MUP_EQ><<<grid, 256>>>(dw, split, target,
                                                    g_cnt, g_ballpat,
                                                    g_balllen, dout);
        else if (mode == MUP_DEF)
          k_ball_gather_grid<MUP_DEF><<<grid, 256>>>(dw, split, 0,
                                                     g_cnt, g_ballpat,
                                                     g_balllen, dout);
        else
          k_ball_gather_grid<MUP_CRIT><<<grid, 256>>>(dw, split, 0,
                                                      g_cnt, g_ballpat,
                                                      g_balllen, dout);
      } else if (mode == MUP_EQ) {
        if (g_fmt == 0)
          k_ball_gather<0><<<grid, 256>>>(dw, split, target, g_cnt,
                                          g_ballpat, g_tbl2_mask, g_tbl2_offs,
                                          g_balllen, dout);
        else
          k_ball_gather<1><<<grid, 256>>>(dw, split, target, g_cnt,
                                          g_ballpat, g_tbl2_mask, g_tbl2_offs,
                                          g_balllen, dout);
      } else if (g_fmt == 0) {
        if (mode == MUP_DEF) LAUNCH(0, MUP_DEF); else LAUNCH(0, MUP_CRIT);
      } else {
        if (mode == MUP_DEF) LAUNCH(1, MUP_DEF); else LAUNCH(1, MUP_CRIT);
      }
#undef LAUNCH
    }
    CK(cudaGetLastError());
  }
  return 0;
}

static int gather_impl(const long long *words, int nwords, int mode,
                       int target, int32_t *out) {
  if (nwords <= 0) return 0;
  for (int off = 0; off < nwords; off += g_codecap) {
    int nb = nwords - off > g_codecap ? g_codecap : nwords - off;
    int rc = stage_words_pinned(words + off, nb);
    if (rc) return rc;
    rc = gather_launch(mode, target, nb, g_d_words, g_d_out, false);
    if (rc) return rc;
    CK(cudaMemcpyAsync(g_h_out, g_d_out, (size_t)nb * 4,
                       cudaMemcpyDeviceToHost, 0));
    CK(cudaStreamSynchronize(0));
    memcpy(out + off, g_h_out, (size_t)nb * 4);
  }
  return 0;
}

// Optional diagnostic counters since init: [0] gather chunks served by the
// set side, [1] by the support walk, [2] predicate-set scans, [3] scans that
// overflowed the set buffer, [4] swap_eval calls on the fast path, [5]
// swap_eval calls that used the mutate-gather-undo fallback, [6]
// count_eq(0) calls answered by the zero-word listing pass, [7] set-side
// chunks that used the tiled (large-Z) kernel.  Writes min(n, 8) values; returns how many.
int dct_stats(long long *out, int n) {
  if (n > 8) n = 8;
  for (int i = 0; i < n; i++) out[i] = g_stat[i];
  return n;
}

int dct_ball_gather(const long long *words, int nwords, int target,
                    int32_t *out) {
  return gather_impl(words, nwords, MUP_EQ, target, out);
}

// mu-fold twin of dct_ball_gather (optional v1 extension): role 0 counts
// DEFICIENT words (cnt < mu) in each word's support set — the exact placement
// gain; role 1 counts CRITICAL words (1 <= cnt <= mu) — the exact removal
// loss.  At mu = 1 these are the cnt==0 / cnt==1 gathers exactly.
int dct_ball_gather_mu(const long long *words, int nwords, int role,
                       int32_t *out) {
  if (nwords <= 0) return 0;
  if (role < 0 || role > 1) return -1;
  return gather_impl(words, nwords, role == 0 ? MUP_DEF : MUP_CRIT, 0, out);
}

// ---- v0.3 optional: updates that report, batched swap evaluation ---------
static int walk_kind(void) {
  return g_prob == PROB_GRID_LINF ? 2 : (g_fmt == 1 ? 1 : 0);
}

// launch the transition-counting update for nb staged words (g_d_words)
static int update_dd_launch(int nb, int delta, long long *dtrans) {
  int split = pick_split(nb);
  dim3 grid((unsigned)nb * split);
#define UDD(W) k_ball_update_dd<W><<<grid, 256>>>(g_d_words, split, delta,   \
      g_cnt, g_ballpat, g_tbl2_mask, g_tbl2_offs, g_balllen, dtrans)
  switch (walk_kind()) {
    case 0: UDD(0); break;
    case 1: UDD(1); break;
    default: UDD(2); break;
  }
#undef UDD
  CK(cudaGetLastError());
  return 0;
}

// Same validation and effect as dct_ball_update; additionally *dd receives
// the EXACT change of the total deficit sum_x max(0, mu - cnt(x)) caused by
// the update (mu = 1: the change of the uncovered count), derived from the
// atomics' old values — no whole-space scan.  Synchronous (one sync).
int dct_ball_update_delta(const long long *words, int nwords, int delta,
                          long long *dd) {
  if (!dd) return -1;
  *dd = 0;
  if (!valid_words(words, nwords) || (delta != 1 && delta != -1)) return -1;
  if (delta == 1 && nwords > MAX_CODE - active_size) return -1;
  std::unordered_map<long long, int> batch;
  for (int i = 0; i < nwords; ++i) ++batch[words[i]];
  if (delta == -1)
    for (const auto &entry : batch) {
      auto it = active.find(entry.first);
      if (it == active.end() || it->second < entry.second) return -1;
    }
  if (nwords == 0) return 0;
  cnt_changed();
  if (delta > 0) g_plus_words += nwords;
  // running counter: no reset per call, the difference is this update's
  long long *dtrans = g_scalar + 2;
  for (int off = 0; off < nwords; off += g_codecap) {
    int nb = nwords - off > g_codecap ? g_codecap : nwords - off;
    bool pinned = nb > ASYNC_UPDATE_MAX;
    int rc = pinned ? stage_words_pinned(words + off, nb)
                    : stage_words_param(words + off, nb);
    if (rc) return rc;
    rc = update_dd_launch(nb, delta, dtrans);
    if (rc) return rc;
    if (pinned && off + nb < nwords) CK(cudaStreamSynchronize(0));
  }
  CK(cudaMemcpyAsync(g_h_scalar + 2, dtrans, 8, cudaMemcpyDeviceToHost, 0));
  CK(cudaStreamSynchronize(0));
  long long tr = g_h_scalar[2] - g_dd_last;
  g_dd_last = g_h_scalar[2];
  *dd = delta > 0 ? -tr : tr;
  for (const auto &entry : batch) {
    int &count = active[entry.first];
    count += delta * entry.second;
    if (count == 0) active.erase(entry.first);
  }
  active_size += delta * nwords;
  return 0;
}

// Batched swap evaluation, read-only.  For k drop words (each must be an
// active codeword) and, per drop i, candidates cand[off[i] .. off[i+1]):
//   loss[i]  = exact deficit increase of removing one copy of drops[i]
//              (= ball_gather role 1 of drops[i]),
//   gain[j]  = exact deficit decrease of then adding cand[j]
//              (= ball_gather role 0 of cand[j] evaluated on cnt with one
//              copy of its drop removed).
// So a swap (drops[i] -> cand[j]) changes the deficit by loss[i] - gain[j],
// exactly as the drop/update/gather/undo sequence would, without touching
// cnt.  Fast path: G = gather role 0 on the current cnt (set side or walk)
// plus the words of each drop's ball that sit exactly at mu (one walk per
// drop), matched by metric; fallback (table not provably the metric ball,
// or that list overflowing): the mutate-gather-undo sequence.  One sync on
// the fast path.  k <= 65535 and off[k] <= 65535.
int dct_swap_eval(const long long *drops, int k, const long long *cand,
                  const long long *off, int32_t *gain, int32_t *loss) {
  if (k <= 0) return 0;
  if (k > g_codecap || !off || off[0] != 0 || !drops || !loss) return -1;
  for (int i = 0; i < k; i++)
    if (off[i + 1] < off[i]) return -1;
  long long nc = off[k];
  if (nc > g_codecap || (nc && (!cand || !gain))) return -1;
  if (!valid_words(drops, k) || !valid_words(cand, (int)nc)) return -1;
  for (int i = 0; i < k; i++)
    if (active.find(drops[i]) == active.end()) return -1;
  int gmode = g_mu == 1 ? MUP_EQ : MUP_DEF;
  bool fast = g_set_ok && g_lbuf && vec_ok(g_cnt);
  if (fast) {
    // one packed upload  [cand nc x 8][drops k x 8][cand->drop nc x 4]
    // one zeroed result  [gain nc x 4][loss k x 4][pad][list count 8]
    // one packed download of the result region: 1 + 1 + 1 transfers
    size_t in_cand = 0, in_drop = (size_t)nc * 8,
           in_cdrop = (size_t)(nc + k) * 8,
           in_bytes = in_cdrop + (size_t)nc * 4;
    size_t r_loss = (size_t)nc * 4, r_cnt = ((size_t)(nc + k) * 4 + 7) & ~7,
           r_bytes = r_cnt + 8;
    char *h = (char *)g_h_swap;
    if (nc) memcpy(h + in_cand, cand, (size_t)nc * 8);
    memcpy(h + in_drop, drops, (size_t)k * 8);
    int32_t *hc = (int32_t *)(h + in_cdrop);
    for (int i = 0; i < k; i++)
      for (long long j = off[i]; j < off[i + 1]; j++) hc[j] = i;
    char *d = (char *)g_d_swap, *dr = (char *)g_d_sres;
    const long long *d_cand = (const long long *)(d + in_cand);
    const long long *d_drop = (const long long *)(d + in_drop);
    const int32_t *d_cdrop = (const int32_t *)(d + in_cdrop);
    int32_t *d_gain = (int32_t *)dr;
    uint32_t *d_loss = (uint32_t *)(dr + r_loss);
    unsigned long long *d_lcnt = (unsigned long long *)(dr + r_cnt);
    CK(cudaMemcpyAsync(d, h, in_bytes, cudaMemcpyHostToDevice, 0));
    CK(cudaMemsetAsync(dr, 0, r_bytes, 0));
    int split = pick_split(k);
    dim3 grid((unsigned)k * split);
#define DW(W) k_drop_walk<W><<<grid, 256>>>(d_drop, split, g_cnt, g_ballpat,  \
      g_tbl2_mask, g_tbl2_offs, g_balllen, d_loss, g_lbuf, g_ldrop, g_lcap,  \
      d_lcnt)
    switch (walk_kind()) {
      case 0: DW(0); break;
      case 1: DW(1); break;
      default: DW(2); break;
    }
#undef DW
    CK(cudaGetLastError());
    if (nc) {
      int rc = gather_launch(gmode, 0, (int)nc, d_cand, d_gain, true);
      if (rc) return rc;
      long long cs = (long long)(8192 / nc);
      if (cs < 1) cs = 1;
      if (cs > 64) cs = 64;
      dim3 g2((unsigned)(nc * cs));
#define SC(P) k_swap_corr<P><<<g2, 256>>>(d_cand, d_cdrop, (int)cs, g_lbuf,  \
      g_ldrop, g_lcap, d_lcnt, d_gain)
      switch (g_prob) {
        case PROB_HAMMING: SC(PROB_HAMMING); break;
        case PROB_LEE: SC(PROB_LEE); break;
        case PROB_TORUS_LINF: SC(PROB_TORUS_LINF); break;
        default: SC(PROB_GRID_LINF); break;
      }
#undef SC
      CK(cudaGetLastError());
    }
    char *hr = (char *)g_h_sres;
    CK(cudaMemcpyAsync(hr, dr, r_bytes, cudaMemcpyDeviceToHost, 0));
    CK(cudaStreamSynchronize(0));
    long long lfound = *(long long *)(hr + r_cnt);
    if (lfound <= g_lcap) {
      memcpy(loss, hr + r_loss, (size_t)k * 4);
      if (nc) memcpy(gain, hr, (size_t)nc * 4);
      g_stat[4]++;
      return 0;
    }
  }
  // fallback: mutate, gather, undo — per drop (exact; cnt restored)
  g_stat[5]++;
  int lmode = g_mu == 1 ? MUP_EQ : MUP_CRIT, lt = g_mu == 1 ? 1 : 0;
  for (int i = 0; i < k; i++) {
    int rc = gather_impl(drops + i, 1, lmode, lt, loss + i);
    if (rc) return rc;
    long long n_i = off[i + 1] - off[i];
    if (!n_i) continue;
    rc = ball_update_inner(drops + i, 1, -1, false);
    if (rc) return rc;
    rc = gather_impl(cand + off[i], (int)n_i, gmode, 0, gain + off[i]);
    int rc2 = ball_update_inner(drops + i, 1, +1, false);
    g_plus_words -= 1;              // an undo is not a placement
    if (rc) return rc;
    if (rc2) return rc2;
  }
  return 0;
}

// total deficit sum_x max(0, mu - cnt(x)) — 0 iff the solution is a mu-fold
// cover.  Optional v1 extension; equals dct_count_eq(0) at mu = 1.
long long dct_count_deficit(void) {
  CK(cudaMemsetAsync(g_scalar, 0, 8, 0));
  if (vec_ok(g_cnt))
    k_cnt_reduce<MUP_DEF><<<pass_blocks(g_space / 8 + 1), 256>>>(
        g_cnt, g_space, 0, g_scalar);
  else
    k_count_deficit<<<pass_blocks(g_space), 256>>>(g_cnt, g_space, g_scalar);
  CK(cudaGetLastError());
  long long r;
  int rc = fetch_scalar(&r);
  return rc ? rc : r;
}

// owner-trick losspass (requires use_owner=1): rebuild cnt+owner from the
// full solution by support marking, then one scan gives loss per element.
// words/M must be the CURRENT full solution; cnt is identical afterwards.
int dct_loss_owner(const long long *words, int M, int32_t *out) {
  if (!g_owner) return -1;
  if (!valid_words(words, M) || (M && !out)) return -2;
  int threads = 256;
  k_fill16<<<g_blocks, threads>>>(g_cnt, 0, (g_space + 1) & ~1LL);
  cnt_rebuilt();
  // owner reset not needed: every covered word gets re-tagged by the mark
  // pass below, and only cnt==1 words are read.
  CK(cudaDeviceSynchronize());
  int rc = ball_update_inner(words, M, +1, true);
  if (rc) return rc;
  active.clear();
  for (int i = 0; i < M; ++i) ++active[words[i]];
  active_size = M;
  // A critical point can contribute to several owners when mu > 1.
  if (g_mu > 1) return dct_ball_gather_mu(words, M, 1, out);
  if (M == 0) return 0;
  CK(cudaMemset(g_d_loss, 0, (size_t)M * 4));
  if (g_mu == 1)
    k_loss_scan<<<g_blocks, threads>>>(g_cnt, g_owner, g_d_loss, g_space);
  else
    k_loss_scan_mu<<<g_blocks, threads>>>(g_cnt, g_owner, g_d_loss, g_space);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out, g_d_loss, (size_t)M * 4, cudaMemcpyDeviceToHost));
  return 0;
}

long long dct_count_eq(int target) {
  // count_eq(0) (the uncovered count at mu = 1) by the same pass that lists
  // the zero words: the count is exact either way (the compaction counts
  // every hit, stored or not), and when the list fits, the next deficient
  // gather or swap_eval finds it cached instead of scanning cnt again.
  if (target == 0 && g_set_ok && g_zbuf && vec_ok(g_cnt)) {
    if (z_mode == MUP_EQ && z_target == 0 && z_mu == g_mu &&
        z_ver == g_cnt_ver)
      return z_n;
    bool ovf = o_mode == MUP_EQ && o_target == 0 && o_mu == g_mu &&
               o_F - (g_plus_words - o_plus) * g_balllen > g_zcap;
    // (a larger list is only built for gathers, which pay for it)
    if (!ovf) {
      g_stat[6]++;
      int r = ensure_zlist(MUP_EQ, 0, g_zcap);
      if (r < 0) return r;
      return r == 1 ? z_n : g_h_scalar[1];
    }
  }
  CK(cudaMemsetAsync(g_scalar, 0, 8, 0));
  if (vec_ok(g_cnt))
    k_cnt_reduce<MUP_EQ><<<pass_blocks(g_space / 8 + 1), 256>>>(
        g_cnt, g_space, target, g_scalar);
  else
    k_count_eq<<<pass_blocks(g_space), 256>>>(g_cnt, target, g_space,
                                              g_scalar);
  CK(cudaGetLastError());
  long long r;
  int rc = fetch_scalar(&r);
  return rc ? rc : r;
}

int dct_map_max(int32_t *out_max) {
  CK(cudaMemsetAsync(g_scalar, 0, 8, 0));
  if (vec_ok(g_A[0]))
    k_max_i32v<<<pass_blocks(g_space / 4 + 1), 256>>>(g_A[0], g_space,
                                                      g_scalar);
  else
    k_max_i32<<<pass_blocks(g_space), 256>>>(g_A[0], g_space, g_scalar);
  CK(cudaGetLastError());
  long long r;
  int rc = fetch_scalar(&r);
  if (rc) return rc;
  *out_max = (int32_t)r;
  return 0;
}

int dct_map_hist(int nbins, int32_t vmax, int32_t *out_hist) {
  if (nbins > 4096) return -1;
  CK(cudaMemsetAsync(g_hist, 0, (size_t)nbins * 4, 0));
  if (vec_ok(g_A[0])) {
    int hb = pass_blocks(g_space / 4 + 1);
    k_hist_v<<<hb < 1024 ? hb : 1024, 256, nbins * 4>>>(g_A[0], g_space,
                                                        nbins, vmax, g_hist);
  } else {
    int hb = pass_blocks(g_space);
    k_hist<<<hb < 1024 ? hb : 1024, 256, nbins * 4>>>(g_A[0], g_space, nbins,
                                                      vmax, g_hist);
  }
  CK(cudaGetLastError());
  CK(cudaMemcpyAsync(g_h_out, g_hist, (size_t)nbins * 4,
                     cudaMemcpyDeviceToHost, 0));
  CK(cudaStreamSynchronize(0));
  memcpy(out_hist, g_h_out, (size_t)nbins * 4);
  return 0;
}

// collect positions with map value >= thr; returns number found (may exceed
// cap; only cap entries are stored).  32-bit legacy (spaces < 2^32).
long long dct_map_extract(int32_t thr, uint32_t *out_idx, int32_t *out_val,
                          int cap) {
  if (cap > g_extract_cap) cap = g_extract_cap;
  CK(cudaMemsetAsync(g_scalar, 0, 8, 0));
  if (vec_ok(g_A[0]))
    k_extract_v<uint32_t><<<pass_blocks(g_space / 4 + 1), 256>>>(
        g_A[0], g_space, thr, g_extract_idx, g_extract_val, cap, g_scalar);
  else
    k_extract<<<pass_blocks(g_space), 256>>>(g_A[0], g_space, thr,
                                             g_extract_idx, g_extract_val,
                                             cap, g_scalar);
  CK(cudaGetLastError());
  long long found;
  int rc = fetch_scalar(&found);
  if (rc) return rc;
  long long m = found < cap ? found : cap;
  CK(cudaMemcpy(out_idx, g_extract_idx, (size_t)m * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(out_val, g_extract_val, (size_t)m * 4, cudaMemcpyDeviceToHost));
  return found;
}

long long dct_map_extract64(int32_t thr, int64_t *out_idx, int32_t *out_val,
                            int cap) {
  if (cap > g_extract_cap) cap = g_extract_cap;
  CK(cudaMemsetAsync(g_scalar, 0, 8, 0));
  if (vec_ok(g_A[0]))
    k_extract_v<int64_t><<<pass_blocks(g_space / 4 + 1), 256>>>(
        g_A[0], g_space, thr, g_extract_idx64, g_extract_val, cap, g_scalar);
  else
    k_extract64<<<pass_blocks(g_space), 256>>>(g_A[0], g_space, thr,
                                               g_extract_idx64, g_extract_val,
                                               cap, g_scalar);
  CK(cudaGetLastError());
  long long found;
  int rc = fetch_scalar(&found);
  if (rc) return rc;
  long long m = found < cap ? found : cap;
  CK(cudaMemcpy(out_idx, g_extract_idx64, (size_t)m * 8,
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(out_val, g_extract_val, (size_t)m * 4, cudaMemcpyDeviceToHost));
  return found;
}

int dct_map_read_at(const long long *idx, int k, int32_t *out) {
  if (k <= 0) return 0;
  for (int off = 0; off < k; off += g_codecap) {
    int nb = k - off > g_codecap ? g_codecap : k - off;
    int rc = stage_words_pinned(idx + off, nb);
    if (rc) return rc;
    k_read_at<<<(nb + 255) / 256, 256>>>(g_A[0], g_d_words, nb, g_d_out);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(g_h_out, g_d_out, (size_t)nb * 4,
                       cudaMemcpyDeviceToHost, 0));
    CK(cudaStreamSynchronize(0));
    memcpy(out + off, g_h_out, (size_t)nb * 4);
  }
  return 0;
}

// download a range of cnt (for CPU spot checks)
int dct_read_cnt(long long off, long long len, uint16_t *out) {
  CK(cudaStreamSynchronize(0));   // an async ball_update may be in flight
  CK(cudaMemcpy(out, g_cnt + off, (size_t)len * 2, cudaMemcpyDefault));
  return 0;
}

}  // extern "C"

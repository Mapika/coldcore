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

#include "dctcore_core.cuh"
#include "dctcore_hamming.cuh"
#include "dctcore_torus.cuh"
#include "dctcore_lee.cuh"
#include "dctcore_mu.cuh"
#include "dctcore_grid.cuh"

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
  if (n < 2 || n > MAXN || R < 1) return -1;
  if (problem == PROB_HAMMING) {
    if (R > MAXR) return -1;
    for (int i = 0; i < n; i++)
      if (ax[i] < 2 || ax[i] > MAXQH) return -1;
    g_nlayers = R + 1;
  } else if (problem == PROB_TORUS_LINF) {
    for (int i = 0; i < n; i++)
      if (ax[i] < 3 || ax[i] > MAXAX || 2 * R + 1 > ax[i]) return -1;
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
  for (int i = 0; i < n; i++) g_space *= ax[i];
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
  if (alloc_buf((void **)&g_cnt, (size_t)g_space * 2, cnt_mode)) return -4;
  if (cnt_mode == MEM_DEV) CK(cudaMemset(g_cnt, 0, (size_t)g_space * 2));
  g_owner = nullptr;
  if (use_owner) {
    if (alloc_buf((void **)&g_owner, (size_t)g_space * 4, cnt_mode))
      return -5;
  }
  g_balllen = balllen;
  CK(cudaMalloc(&g_ballpat, (size_t)balllen * 8));
  g_bytes_dev += balllen * 8;
  CK(cudaMemcpy(g_ballpat, ballpat, (size_t)balllen * 8,
                cudaMemcpyHostToDevice));
  g_tbl2_mask = nullptr; g_tbl2_offs = nullptr;
  g_codecap = 1 << 16;
  CK(cudaMalloc(&g_codeidx, (size_t)g_codecap * 8));
  CK(cudaMalloc(&g_d_words, (size_t)g_codecap * 8));
  CK(cudaMalloc(&g_d_out, (size_t)g_codecap * 4));
  CK(cudaMalloc(&g_d_loss, (size_t)g_codecap * 4));
  g_bytes_dev += (size_t)g_codecap * 24;
  g_codeM = 0;
  CK(cudaMalloc(&g_scalar, 64 * 8));
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
  if (g_hist) { cudaFree(g_hist); g_hist = 0; }
  if (g_extract_idx) { cudaFree(g_extract_idx); g_extract_idx = 0; }
  if (g_extract_idx64) { cudaFree(g_extract_idx64); g_extract_idx64 = 0; }
  if (g_extract_val) { cudaFree(g_extract_val); g_extract_val = 0; }
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
  if (M > g_codecap) return -1;
  CK(cudaMemcpy(g_codeidx, idx, (size_t)M * 8, cudaMemcpyHostToDevice));
  g_codeM = M;
  return 0;
}

// init_mode: 0 = code multiset, 1 = [cnt==0], 2 = [cnt==1]
//            (mu > 1: 1 = [cnt<mu] deficient, 2 = [1<=cnt<=mu] critical —
//             the same two indicators, generalized; see dctcore_mu.cuh)
// out_mode:  1 = sum -> A0 (gain/loss map), 2 = clamped sum -> cnt
int dct_transform(int init_mode, int out_mode) {
  long long N = g_space;
  int blocks = g_blocks, threads = 256;
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
  for (int off = 0; off < nwords; off += g_codecap) {
    int nb = nwords - off > g_codecap ? g_codecap : nwords - off;
    CK(cudaMemcpy(g_d_words, words + off, (size_t)nb * 8,
                  cudaMemcpyHostToDevice));
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
    CK(cudaDeviceSynchronize());
  }
  return 0;
}

int dct_ball_update(const long long *words, int nwords, int delta) {
  return ball_update_inner(words, nwords, delta, false);
}

int dct_ball_gather(const long long *words, int nwords, int target,
                    int32_t *out) {
  if (nwords <= 0) return 0;
  for (int off = 0; off < nwords; off += g_codecap) {
    int nb = nwords - off > g_codecap ? g_codecap : nwords - off;
    CK(cudaMemcpy(g_d_words, words + off, (size_t)nb * 8,
                  cudaMemcpyHostToDevice));
    int split = pick_split(nb);
    if (split > 1) CK(cudaMemset(g_d_out, 0, (size_t)nb * 4));
    dim3 grid((unsigned)nb * split);
    if (g_prob == PROB_GRID_LINF)
      k_ball_gather_grid<MUP_EQ><<<grid, 256>>>(g_d_words, split, target,
                                                g_cnt, g_ballpat, g_balllen,
                                                g_d_out);
    else if (g_fmt == 0)
      k_ball_gather<0><<<grid, 256>>>(g_d_words, split, target, g_cnt,
                                      g_ballpat, g_tbl2_mask, g_tbl2_offs,
                                      g_balllen, g_d_out);
    else
      k_ball_gather<1><<<grid, 256>>>(g_d_words, split, target, g_cnt,
                                      g_ballpat, g_tbl2_mask, g_tbl2_offs,
                                      g_balllen, g_d_out);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out + off, g_d_out, (size_t)nb * 4, cudaMemcpyDeviceToHost));
  }
  return 0;
}

// mu-fold twin of dct_ball_gather (optional v1 extension): role 0 counts
// DEFICIENT words (cnt < mu) in each word's support set — the exact placement
// gain; role 1 counts CRITICAL words (1 <= cnt <= mu) — the exact removal
// loss.  At mu = 1 these are the cnt==0 / cnt==1 gathers exactly.
int dct_ball_gather_mu(const long long *words, int nwords, int role,
                       int32_t *out) {
  if (nwords <= 0) return 0;
  if (role < 0 || role > 1) return -1;
  for (int off = 0; off < nwords; off += g_codecap) {
    int nb = nwords - off > g_codecap ? g_codecap : nwords - off;
    CK(cudaMemcpy(g_d_words, words + off, (size_t)nb * 8,
                  cudaMemcpyHostToDevice));
    int split = pick_split(nb);
    if (split > 1) CK(cudaMemset(g_d_out, 0, (size_t)nb * 4));
    dim3 grid((unsigned)nb * split);
#define LAUNCH(FMT, MODE)                                                     \
    k_ball_gather_mu<FMT, MODE><<<grid, 256>>>(g_d_words, split, g_cnt,       \
                                               g_ballpat, g_tbl2_mask,        \
                                               g_tbl2_offs, g_balllen,        \
                                               g_d_out)
#define LAUNCH_GRID(MODE)                                                     \
    k_ball_gather_grid<MODE><<<grid, 256>>>(g_d_words, split, 0, g_cnt,       \
                                            g_ballpat, g_balllen, g_d_out)
    if (g_prob == PROB_GRID_LINF) {
      if (role == 0) LAUNCH_GRID(MUP_DEF); else LAUNCH_GRID(MUP_CRIT);
    }
    else if (g_fmt == 0) { if (role == 0) LAUNCH(0, MUP_DEF); else LAUNCH(0, MUP_CRIT); }
    else                 { if (role == 0) LAUNCH(1, MUP_DEF); else LAUNCH(1, MUP_CRIT); }
#undef LAUNCH_GRID
#undef LAUNCH
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out + off, g_d_out, (size_t)nb * 4, cudaMemcpyDeviceToHost));
  }
  return 0;
}

// total deficit sum_x max(0, mu - cnt(x)) — 0 iff the solution is a mu-fold
// cover.  Optional v1 extension; equals dct_count_eq(0) at mu = 1.
long long dct_count_deficit(void) {
  CK(cudaMemset(g_scalar, 0, 8));
  k_count_deficit<<<g_blocks, 256>>>(g_cnt, g_space, g_scalar);
  CK(cudaDeviceSynchronize());
  long long r;
  CK(cudaMemcpy(&r, g_scalar, 8, cudaMemcpyDeviceToHost));
  return r;
}

// owner-trick losspass (requires use_owner=1): rebuild cnt+owner from the
// full solution by support marking, then one scan gives loss per element.
// words/M must be the CURRENT full solution; cnt is identical afterwards.
int dct_loss_owner(const long long *words, int M, int32_t *out) {
  if (!g_owner) return -1;
  if (M > g_codecap) return -2;
  int threads = 256;
  k_fill16<<<g_blocks, threads>>>(g_cnt, 0, g_space);
  // owner reset not needed: every covered word gets re-tagged by the mark
  // pass below, and only cnt==1 words are read.
  CK(cudaDeviceSynchronize());
  int rc = ball_update_inner(words, M, +1, true);
  if (rc) return rc;
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
  CK(cudaMemset(g_scalar, 0, 8));
  k_count_eq<<<g_blocks, 256>>>(g_cnt, target, g_space, g_scalar);
  CK(cudaDeviceSynchronize());
  long long r;
  CK(cudaMemcpy(&r, g_scalar, 8, cudaMemcpyDeviceToHost));
  return r;
}

int dct_map_max(int32_t *out_max) {
  CK(cudaMemset(g_scalar, 0, 8));
  k_max_i32<<<g_blocks, 256>>>(g_A[0], g_space, g_scalar);
  CK(cudaDeviceSynchronize());
  long long r;
  CK(cudaMemcpy(&r, g_scalar, 8, cudaMemcpyDeviceToHost));
  *out_max = (int32_t)r;
  return 0;
}

int dct_map_hist(int nbins, int32_t vmax, int32_t *out_hist) {
  if (nbins > 4096) return -1;
  CK(cudaMemset(g_hist, 0, (size_t)nbins * 4));
  k_hist<<<1024, 256, nbins * 4>>>(g_A[0], g_space, nbins, vmax, g_hist);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out_hist, g_hist, (size_t)nbins * 4, cudaMemcpyDeviceToHost));
  return 0;
}

// collect positions with map value >= thr; returns number found (may exceed
// cap; only cap entries are stored).  32-bit legacy (spaces < 2^32).
long long dct_map_extract(int32_t thr, uint32_t *out_idx, int32_t *out_val,
                          int cap) {
  if (cap > g_extract_cap) cap = g_extract_cap;
  CK(cudaMemset(g_scalar, 0, 8));
  k_extract<<<g_blocks, 256>>>(g_A[0], g_space, thr, g_extract_idx,
                               g_extract_val, cap, g_scalar);
  CK(cudaDeviceSynchronize());
  long long found;
  CK(cudaMemcpy(&found, g_scalar, 8, cudaMemcpyDeviceToHost));
  long long m = found < cap ? found : cap;
  CK(cudaMemcpy(out_idx, g_extract_idx, (size_t)m * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(out_val, g_extract_val, (size_t)m * 4, cudaMemcpyDeviceToHost));
  return found;
}

long long dct_map_extract64(int32_t thr, int64_t *out_idx, int32_t *out_val,
                            int cap) {
  if (cap > g_extract_cap) cap = g_extract_cap;
  CK(cudaMemset(g_scalar, 0, 8));
  k_extract64<<<g_blocks, 256>>>(g_A[0], g_space, thr, g_extract_idx64,
                                 g_extract_val, cap, g_scalar);
  CK(cudaDeviceSynchronize());
  long long found;
  CK(cudaMemcpy(&found, g_scalar, 8, cudaMemcpyDeviceToHost));
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
    CK(cudaMemcpy(g_d_words, idx + off, (size_t)nb * 8,
                  cudaMemcpyHostToDevice));
    k_read_at<<<(nb + 255) / 256, 256>>>(g_A[0], g_d_words, nb, g_d_out);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out + off, g_d_out, (size_t)nb * 4, cudaMemcpyDeviceToHost));
  }
  return 0;
}

// download a range of cnt (for CPU spot checks)
int dct_read_cnt(long long off, long long len, uint16_t *out) {
  CK(cudaMemcpy(out, g_cnt + off, (size_t)len * 2, cudaMemcpyDefault));
  return 0;
}

}  // extern "C"

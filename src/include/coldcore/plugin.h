/* coldcore/plugin.h — plugin C ABI, v1 (AGREED).
 *
 * Contract authority: docs/PLUGIN.md (v1, frozen 2026-08-22). This
 * header restates it for plugin authors; on any divergence PLUGIN.md
 * wins.
 *
 * v1 decisions:
 *  - SINGLETON context: one cell per process (planes are 10s-100s of
 *    GB). A `void *ctx` first argument is reserved for v2 — do not
 *    emulate handles.
 *  - FLAT `dct_*` SYMBOLS are the stable surface. Hosts may wrap them
 *    in their own vtable (coldcore does, in Python).
 *  - RETURN CODES: 0 = ok, negative = engine error, -100-e = CUDA
 *    error e (message on stderr). No last-error string in v1.
 *  - CALLER-ALLOCATED output buffers everywhere.
 *  - TYPED INIT ARGS, no string blobs; the per-plugin host-side Python
 *    builds the support table (see coldcore/patterns.py).
 *
 * Model: product space X = prod_j Z_{ax_j}, n <= 16 axes, |X| <= 2^40.
 * Dense int32 layer planes + uint16 multiplicity plane `cnt`, each
 * placeable in HBM (mode 0), LPDDR via ATS (1), or managed (2).
 *
 * REQUIRED symbols: dct_init3, dct_free, dct_set_code, dct_transform,
 *   dct_ball_update, dct_ball_gather, dct_count_eq, dct_map_max,
 *   dct_map_extract, dct_map_extract64, dct_map_read_at, dct_read_cnt.
 * OPTIONAL (feature-test via symbol presence / rc): dct_loss_owner,
 *   dct_set_tbl2, dct_use_fmt, dct_map_hist, dct_host_bytes;
 *   mu-fold multiple covering dct_set_mu / dct_get_mu /
 *   dct_ball_gather_mu / dct_count_deficit;
 *   legacy inits dct_init2 / dct_init (homogeneous Hamming).
 */
#ifndef COLDCORE_PLUGIN_H
#define COLDCORE_PLUGIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLDCORE_PLUGIN_ABI_VERSION 1

/* problem ids (registered in the engine's dct_init3/dct_transform) */
#define CCP_PROB_HAMMING    0   /* covering codes, mixed radix ok      */
#define CCP_PROB_TORUS_LINF 1   /* torus domination, Chebyshev balls   */
#define CCP_PROB_LEE        2   /* Lee-metric covering codes (sum of
                                   circular per-axis distances)        */
#define CCP_PROB_GRID_LINF  3   /* OPEN-boundary grid domination under
                                   Chebyshev balls (no wraparound: the
                                   per-axis window clamps at the edge —
                                   the coverage relation is not shift
                                   invariant)                          */

/* ---- lifecycle ----
 * ballpat/balllen: packed support table, v1 format — one uint64 per
 * pattern, byte k = (pos<<4 | delta), delta a circular shift in
 * 1..ax_pos-1, zero byte terminates, <= 8 bytes/pattern; entry 0 = the
 * empty pattern (center). Requires ax_j <= 16, pattern weight <= 8.
 * layers_mode/cnt_mode: 0 HBM | 1 ATS-LPDDR | 2 managed, per array
 * group. use_owner: allocate the owner plane (dct_loss_owner). */
int  dct_init3(int problem, int n, const int *ax, int R,
               const uint64_t *ballpat, long long balllen, int extract_cap,
               long long *bytes, int layers_mode, int cnt_mode,
               int use_owner);
int  dct_free(void);
long long dct_host_bytes(void);              /* optional: LPDDR bytes  */

/* legacy inits (homogeneous Hamming) */
int  dct_init2(int q, int n, int R, const uint64_t *ballpat,
               long long balllen, int extract_cap, long long *bytes,
               int layers_mode, int cnt_mode, int use_owner);
int  dct_init (int q, int n, int R, const uint64_t *ballpat,
               long long balllen, int extract_cap, long long *bytes);

/* ---- solution / transforms ----
 * init_mode: 0 = solution multiset (set_code), 1 = [cnt==0], 2 = [cnt==1]
 * out_mode:  1 = coverage sum -> map (A0), 2 = clamped sum -> cnt      */
int  dct_set_code(const long long *idx, int M);
int  dct_transform(int init_mode, int out_mode);

/* ---- mu-fold multiple covering (optional; K^mu_q(n,R)) ----
 * dct_set_mu(mu) makes every word need cnt >= mu.  It generalizes the
 * two transform indicators (init_mode 1 -> [cnt<mu] "deficient",
 * init_mode 2 -> [1<=cnt<=mu] "critical") and the owner losspass; at
 * mu = 1 (default, and the state after dct_free) all of them are the
 * legacy [cnt==0] / [cnt==1] fields verbatim.  Absence of these symbols
 * means the plugin supports mu = 1 only.                              */
int  dct_set_mu(int mu);                                         /* opt */
int  dct_get_mu(void);                                           /* opt */

/* ---- incremental / exact per-candidate ---- */
int  dct_ball_update(const long long *words, int nwords, int delta);
int  dct_ball_gather(const long long *words, int nwords, int target,
                     int32_t *out);
/* role 0 = # deficient (cnt<mu) in the support set = exact place gain;
 * role 1 = # critical (1<=cnt<=mu) = exact removal loss.            opt */
int  dct_ball_gather_mu(const long long *words, int nwords, int role,
                        int32_t *out);
int  dct_loss_owner(const long long *words, int M, int32_t *out); /* opt */

/* ---- reductions / map queries ----
 * map_extract: values >= thr, unordered; returns total found (may
 * exceed cap; only cap entries stored).
 * map_hist (optional): positive v binned at (v-1)*nbins/vmax, clipped. */
long long dct_count_eq(int target);
/* total deficit sum_x max(0, mu - cnt(x)); 0 iff the solution is a
 * mu-fold cover (= dct_count_eq(0) at mu = 1).                     opt */
long long dct_count_deficit(void);
int  dct_map_max(int32_t *out);
int  dct_map_hist(int nbins, int32_t vmax, int32_t *out);        /* opt */
long long dct_map_extract  (int32_t thr, uint32_t *idx, int32_t *val,
                            int cap);
long long dct_map_extract64(int32_t thr, int64_t *idx, int32_t *val,
                            int cap);
int  dct_map_read_at(const long long *idx, int k, int32_t *out);
int  dct_read_cnt(long long off, long long len, uint16_t *out);

/* ---- optional alternative support-table format (ox) ---- */
int  dct_set_tbl2(const uint16_t *mask, const uint64_t *offs,
                  long long len);
int  dct_use_fmt(int fmt);                   /* 0 = v1 bytes, 1 = ox    */

#ifdef __cplusplus
}
#endif
#endif /* COLDCORE_PLUGIN_H */

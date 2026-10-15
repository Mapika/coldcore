# coldcore plugin ABI — v1 specification

Status: **COMMITTED** (2026-08-22) for this engine generation. This is
the authoritative contract; `src/include/coldcore/plugin.h` restates it for
plugin authors, and on any divergence this document wins. Hosts may wrap
the flat symbols in their own vtable (coldcore's Python binding does);
the symbols below are the stable surface.

## What the engine is

A GPU engine for combinatorial covering-type problems whose move gains
and losses can be computed for ALL candidate positions at once by a
**separable transform over a finite product space**
`X = prod_j Z_{ax_j}` (heterogeneous axis sizes allowed, n <= 16 axes,
|X| <= 2^40).

Problems in scope: covering codes (q-ary and mixed-radix Hamming),
dominating sets on torus grids / Hamming graphs, packing and multi-cover
variants — any problem where "u covers x" is a shift-invariant product
predicate per axis.

Three layers:

1. **STATE CORE** (problem-blind)
   - dense int32 layer planes `A_0..A_L` and a uint16 multiplicity plane
     `cnt` over X, each placeable in HBM (`cudaMalloc`), LPDDR via ATS
     (pinned `malloc` — GH200 cache-coherent C2C), or managed memory
     with preferred-location CPU: memory modes 0/1/2 per array group;
   - support-set walks: exact incremental `cnt` +/- updates and
     `cnt==target` gathers over a move's coverage set, given as a packed
     shift-pattern table (formats below), with split-block launch
     geometry;
   - owner plane + owner-trick loss pass (mark tags owner, loss = one
     scan);
   - reductions (count_eq, max, histogram) and threshold extraction with
     32- and 64-bit indices.

2. **PROBLEM PLUGIN** (one `.cuh` + a host-side table builder)
   Supplies exactly two things:
   a) an **axis-pass driver** `int <name>_axes(int out_mode, int blocks,
      int threads)` that turns `A_0 = indicator(S)` into the coverage
      field via n separable per-axis passes (out_mode semantics below);
   b) a **support-table builder** (host side, Python is fine): the list
      of coordinate-shift patterns enumerating one move's coverage set.
   Registration: a `PROB_*` id, an arm in `dct_init3` validation
   (axis-size/R bounds, number of layer planes), an arm in
   `dct_transform`.

3. **SEARCH CORE** (exact lazy greedy, LNS ruin-and-recreate, peel,
   notch descent, verification gate). Problem-blind: talks to the engine
   only through the oracle interface below.

## The oracle interface (what search sees)

- `recount(S)`         -> `cnt(x)` = # elements of S covering x, for all x
- `gain_map()`         -> `A0(x)` = # currently-uncovered words a move at
                          x would cover (from `S = [cnt==0]`)
- `loss_map()`         -> `A0(x)` = private coverage of a move at x
                          (from `S = [cnt==1]`)
- `ball_update(w,±1)`  -> exact incremental `cnt` update for a placed or
                          removed word w
- `ball_gather(w,t)`   -> exact # of words with `cnt==t` in w's coverage set
- `count_eq(t)`, `map_max()`, `map_hist()`, `map_extract(thr)`,
  `map_read_at(idx)`, `read_cnt()`
- optional: `loss_owner(S)` (owner-trick), `enable_tbl2()/use_fmt(f)`

Submodularity of coverage gain is what makes the lazy greedy exact; any
plugin whose coverage predicate is a fixed-shape neighborhood inherits
it.

## C ABI (flat symbols, singleton context)

Decisions (2026-08-22):
- **Singleton, not handles.** One cell per process is the operating
  model (planes are tens to hundreds of GB). A `void *ctx` first
  argument is reserved for a future v2; do not emulate it.
- **Flat `dct_*` symbols**, int return codes (0 = ok, negative = engine
  error; `-100-e` = CUDA error e, message on stderr). No last-error
  string in v1.
- **Caller-allocated outputs** everywhere.
- **Typed init args**, no string blobs: the host-side per-plugin Python
  builds the support table and passes it in.

```
int  dct_init3(int problem, int n, const int *ax, int R,
               const uint64_t *ballpat, long long balllen, int extract_cap,
               long long *bytes, int layers_mode, int cnt_mode,
               int use_owner);
int  dct_init2(q, n, R, ...)                  // Hamming, homogeneous (legacy)
int  dct_init (q, n, R, ...)                  // + all-HBM (legacy ABI)
int  dct_free (void);
long long dct_host_bytes(void);               // LPDDR bytes of last init
int  dct_set_code(const long long *idx, int M);        // solution multiset
int  dct_transform(int init_mode, int out_mode);
     // init_mode: 0 = solution multiset (set_code), 1 = [cnt==0], 2 = [cnt==1]
     // out_mode:  1 = coverage sum -> A0,  2 = exact sum -> cnt (init_mode 0 only)
int  dct_ball_update(const long long *words, int nwords, int delta);
int  dct_ball_gather(const long long *words, int nwords, int target,
                     int32_t *out);
int  dct_loss_owner(const long long *words, int M, int32_t *out);  // optional
long long dct_count_eq(int target);
int  dct_map_max(int32_t *out);
int  dct_map_hist(int nbins, int32_t vmax, int32_t *out);
long long dct_map_extract  (int32_t thr, uint32_t *idx, int32_t *val, int cap);
long long dct_map_extract64(int32_t thr, int64_t  *idx, int32_t *val, int cap);
int  dct_map_read_at(const long long *idx, int k, int32_t *out);
int  dct_read_cnt(long long off, long long len, uint16_t *out);
int  dct_set_tbl2(const uint16_t *mask, const uint64_t *offs, long long len);
int  dct_use_fmt(int fmt);                    // 0 = v1 bytes, 1 = ox format
```

REQUIRED for any plugin: `init3/free, set_code, transform` (all three
init_modes), `ball_update/gather, count_eq, map_max/extract(64)/read_at,
read_cnt`. OPTIONAL (feature-test via symbol presence / rc):
`loss_owner, set_tbl2/use_fmt, map_hist, host_bytes`, the v0.3
`ball_update_delta, swap_eval, stats` (section "v0.3 additions" below),
and the μ-fold
multiple-covering group `set_mu/get_mu/ball_gather_mu/count_deficit`
(see `docs/multi-coverage.md`; with `dct_set_mu(μ)` active, init_modes
1/2 become the deficiency predicates `[cnt<μ]`/`[1<=cnt<=μ]`, and
`dct_ball_gather` targets 0/1 act as roles "deficient"/"critical" —
both collapse to the legacy literals at μ=1; `dct_free` resets μ to 1).

## Exact counter limits (v0.2.2)

The product-space CUDA plugin supports at most **65,535 active codeword
occurrences**, counting duplicates. This conservative global limit keeps
all uint16 multiplicities exact, including during packed atomic updates.
Recount, owner-loss reconstruction, and incremental insertions reject
larger multisets. Invalid indices, update signs other than ±1, and
removals exceeding the active multiplicity return an error before any
counter update. A rejected input leaves the current count field intact.
`set_code` uploads a pending multiset; only `transform(0,2)` replaces the
represented solution. Supported transform pairs are `(0,1)`, `(0,2)`,
`(1,1)`, and `(2,1)`. Indicator inputs cannot write the count field.
Support tables must enumerate each covered point once and have at most
INT32_MAX entries. The owner-loss helper uses per-codeword critical-point
gathers for mu > 1, since one owner tag cannot represent every loss.
Rebuild the covering plugin when upgrading from an earlier release.
These limits do not describe ABI 2, which has separate int32 counters.

## Support-table formats (state core walks both)

- v1 (`ballpat`): one uint64 per pattern; byte k = `pos<<4 | delta`,
  delta in `1..ax_pos-1` (circular shift), zero byte terminates; <= 8
  bytes per pattern; entry 0 = empty pattern (the center itself).
- ox (`mask`,`offs`): uint16 position mask + uint64 with delta nibble at
  `4*pos`. Same enumeration, any order.

Both formats therefore require `ax_j <= 16`, and the *pattern weight*
(number of changed positions) <= 8. Hamming balls have weight <= R;
torus Chebyshev balls have weight <= n, hence n <= 8 for `torus_linf`.

## Existing plugins

| id | name | axes | layers | transform |
|---|---|---|---|---|
| 0 | `hamming` | 2..10, mixed ok | R+1 (R<=6) | distance-count DP: `newA_d[a] = A_d[a] + T_{d-1} - A_{d-1}[a]` per fiber |
| 1 | `torus_linf` | 3..16, `2R+1<=ax_j` | 1 | circular window sum: `newA[a] = sum_{|e|<=R} A[(a+e) mod ax_j]` |
| 2 | `lee` | 2..16, mixed ok | R+1 (R<=6) | circulant Lee-weight DP: `newA_d[a] = A_d[a] + sum_w (A_{d-w}[a-w] + A_{d-w}[a+w])`, antipode once for even axes |
| 3 | `grid_linf` | 3..16, mixed ok | 1 | clamped window sum over `[max(0,a-R), min(ax_j-1,a+R)]` (open boundaries; support-table deltas resolved against the centre digit at walk time; ox format rejected) |

All verified exhaustively against independent CPU brute force; each
family additionally re-derives a known optimum through the untouched
search core (9×9 king torus = 9, perfect Golomb–Welch Lee code in Z_5^2
= 5, γ(8×8 king graph) = 9).

## Memory-placement policy (host binding, automatic)

If all planes fit in the HBM budget (env `COLDCORE_HBM_BUDGET`, default
80 GB): everything HBM. Else: layer planes -> LPDDR (mode 1, pinned
malloc + `mbind(MPOL_BIND, node0)`; the transforms stream them
sequentially over C2C), `cnt` stays HBM while `2*|X| <= 40 GB` (the
random-access ball walks live there). Measured placement numbers:
[benchmarks.md](benchmarks.md) — including the pinning caveat that makes
LPDDR numbers reproducible.

## v0.3 additions to the `dct_*` ABI

Added 2026-10 for v0.3.0. Every change is **additive**: the ABI stays
version 1, no signature changed, and a host written against v0.2 runs
unchanged. New symbols are optional and feature-tested by presence, as
before.

### New optional symbols

```
int dct_ball_update_delta(const long long *words, int nwords, int delta,
                          long long *dd);
    // dct_ball_update + *dd = exact change of the total deficit
    // sum_x max(0, mu - cnt(x)) caused by the update (mu = 1: change of
    // the uncovered count). Read off the update's own atomics: an
    // increment from c < mu lowers the deficit by one, a decrement from
    // c <= mu raises it by one. Synchronous. Same validation as
    // dct_ball_update.
int dct_swap_eval(const long long *drops, int k, const long long *cand,
                  const long long *off, int32_t *gain, int32_t *loss);
    // read-only batched swap evaluation. drops[0..k) must be active
    // codewords; drop i owns candidates cand[off[i] .. off[i+1]),
    // off[0] = 0, k <= 65535, off[k] <= 65535.
    //   loss[i] = deficit increase of removing one copy of drops[i]
    //           (= ball_gather role 1 of drops[i])
    //   gain[j] = deficit decrease of then adding cand[j]
    //           (= ball_gather role 0 of cand[j] on cnt with that copy
    //              removed)
    // cnt is not modified. One device round trip on the fast path.
int dct_stats(long long *out, int n);
    // diagnostic counters since init: [0] gather chunks served by the
    // set side, [1] by the support walk, [2] predicate-set scans,
    // [3] scans whose set overflowed, [4] swap_eval fast-path calls,
    // [5] swap_eval fallback calls, [6] count_eq(0) calls answered by
    // the zero-word listing pass, [7] set-side chunks that used the
    // tiled large-set kernel. Returns min(n, 8).
```

Python: `GpuBackend.ball_update_delta(idx, delta) -> int`,
`GpuBackend.swap_eval(drops, cands) -> (loss, gains)`,
`GpuBackend.gpu_stats() -> dict`. On a plugin without the symbols the
first two attributes are `None` (feature-test with
`getattr(backend, "swap_eval", None)`); the CPU backends do not have
them.

### Behaviour of existing symbols (results unchanged)

* **Asynchronous small updates.** `dct_ball_update` validates on the
  host, launches, and returns without waiting for the device when the
  batch has at most 1,024 words (the words travel as kernel parameters,
  so no host buffer outlives the call). Every call that returns data
  waits on the same stream, so results are always current. A device
  fault in such an update is reported by the next synchronising call
  instead of by the update itself. Larger batches synchronise as before.
* **One synchronisation per call.** Inputs and outputs go through
  pinned host staging buffers with asynchronous copies; reductions,
  gathers, extraction and `dct_map_read_at` wait once.
* **Set-side gather.** For the deficient predicate (role 0, or the
  literal `cnt == 0`), `dct_ball_gather(_mu)` may list the words that
  satisfy the predicate (one sequential pass over `cnt`, cached until
  `cnt` changes) and count, per candidate, those within distance R
  (Hamming, Lee, torus or open-grid L-infinity) instead of walking the
  candidate's support table. The two are the same count when the table
  is the metric ball; the plugin proves that at init (every entry within
  radius R by admissible deltas, entry count equal to the ball's
  cardinality; entries distinct is the v1 precondition) and otherwise
  never uses the set side. `dct_count_eq(0)` runs the same listing
  pass (its count is exact whether or not the list fits), so the gather
  after an `uncovered()` check finds the list cached. It is chosen when
  the walk would read at
  least `max(|X|/8, 2^21)` counters and the set has at most
  `2 |B|` entries (`DCT_SET_RATIO`; the list buffer grows on demand up
  to `min(|X|, 2^27)` entries); otherwise the walk runs. Sets above
  32,768 entries are matched by a tiled kernel that streams the list
  once per 32 candidates. Env:
  `DCT_SET_GATHER=0` off, `=2` whenever the set fits (testing);
  `DCT_SET_MIN=<n>` overrides the walk-size threshold.
* **Hamming transform.** No layer fill passes (after axis pass j only
  layers 0..min(j, R) can be nonzero, so pass j reads only those), the
  gain/loss indicator is computed inside the first pass, exact-Q kernels
  for every Q in 2..10 and R in 1..6, and two adjacent axes of equal size
  are processed by one kernel through shared memory. Env:
  `DCT_PAIR_PASSES=0` one axis per kernel, `DCT_LEGACY_TRANSFORM=1` the
  v0.2 kernels, `DCT_TIMING=1` per-kernel times on stderr. Lee, torus
  and grid transforms are unchanged.
* **Reductions** (`count_eq`, `count_deficit`, `map_max`, `map_hist`,
  `map_extract(64)`) read 16 bytes per load; extraction claims output
  slots with one atomic per warp. Slot order was and remains
  unspecified.
* **Support-table order.** The Python binding sorts the v1 table by
  unwrapped offset so that neighbouring threads touch neighbouring
  words (`coldcore.gpu.locality_order`; `COLDCORE_BALL_ORDER=native`
  keeps the builder's order). The walks are order-free.

## ABI 2: explicit incidence

Added 2026-09-01. A second, independent C ABI for problems whose
candidates and covered points are *different* finite sets and whose
coverage relation is an explicit list rather than a shift-invariant
product predicate — covering designs C(v,k,t) are the first. It is not a
v2 of the `dct_*` contract above and shares nothing with it (no
symbols, no translation units, no globals); a process may hold one cell
of each.

Model: `n_cand` candidates, `n_univ` universe elements, an
`(n_cand x d)` row-major int32 matrix `inc` whose row `c` lists the `d`
universe elements candidate `c` covers (fixed `d`, distinct entries per
row). Fields: `cnt[u]` int32 over the universe, `map[c]` int32 over the
candidates. The oracle interface is the one search sees everywhere
else; each operation is a pass over incidence rows.

Conventions: singleton context, flat `sparse_*` symbols, int return
codes (0 ok, negative engine error, `-100-e` = CUDA error e, message on
stderr), caller-allocated outputs. Source:
`src/plugins/sparse/sparse.cu`; host binding: `coldcore.incidence_gpu`.

```
int  sparse_abi_version(void);                 // 2
int  sparse_init(long long n_cand, int d, long long n_univ,
                 const int32_t *inc, int inc_mode, int extract_cap,
                 long long *bytes_dev, long long *bytes_host);
     // inc_mode: 0 = HBM, 1 = CPU memory pinned to NUMA node 0
     //           (posix_memalign + mbind; managed with preferred
     //           location CPU on devices without pageable access)
int  sparse_init_csr(long long n_cand, long long n_univ, long long nnz,
                     const int64_t *indptr, const int32_t *indices,
                     const int32_t *w, const int32_t *uw, int inc_mode,
                     int extract_cap, long long *bytes_dev,
                     long long *bytes_host);
     // weighted, variable-length rows: row c covers indices[indptr[c]..
     // indptr[c+1]) with weights w >= 1; universe weights uw >= 1.
     // cnt += w per chosen row; deficiency = sum uw * max(0, mu - cnt);
     // fields are the exact weighted marginals (coldcore/incidence.py,
     // WeightedIncidenceBackend). indptr/indices/w follow inc_mode.
int  sparse_free(void);
int  sparse_inc_mode(void);                    // placement in effect
int  sparse_is_csr(void);                      // 1 in the weighted mode
int  sparse_set_mu(int mu);  int sparse_get_mu(void);
int  sparse_recount(const long long *idx, long long M);   // cnt from scratch
int  sparse_field(int role);   // map[c] = #{u in row c : role(cnt[u])}
     // role 0 = deficient (cnt < mu) -> the gain field
     //      1 = critical (1 <= cnt <= mu) -> the loss field
     //      other = literal cnt == role
int  sparse_ball_update(const long long *idx, long long m, int delta);
int  sparse_ball_gather(const long long *idx, long long m, int role,
                        int32_t *out);         // exact, never stale
long long sparse_count_eq(int target);         // literal cnt == target
long long sparse_count_deficit(void);          // sum max(0, mu - cnt)
int  sparse_map_max(int32_t *out);
int  sparse_map_hist(int nbins, int32_t vmax, int32_t *out);  // nbins <= 4096
long long sparse_map_extract(int32_t thr, int64_t *idx, int32_t *val,
                             int cap);          // found may exceed cap
int  sparse_map_read_at(const long long *idx, long long k, int32_t *out);
int  sparse_read_cnt(long long off, long long len, int32_t *out);  // universe
long long sparse_dev_bytes(void);  long long sparse_host_bytes(void);
```

Semantics match the `dct_*` queries where both exist: `map_hist` bins
positive `v` at `(v-1)*nbins/vmax` clipped to the last bin;
`map_extract` compacts `map >= thr` through an atomic slot counter,
unordered, storing at most `cap` entries and returning the true count;
`ball_gather` targets 0/1 are the mu-fold roles and collapse to the
literal counts at mu = 1. Indices are 64-bit throughout (`n_cand` may
exceed 2^32; the universe is int32).

Placement: `cnt`, `map`, the extraction buffers and the staging buffers
for candidate lists are always in HBM. The incidence matrix goes to HBM
when the whole cell fits the host binding's budget
(`COLDCORE_HBM_BUDGET`, default 80 GB), else to pinned CPU memory and
is streamed by the kernels over C2C/ATS. The CPU implementation of the
same operations is `coldcore.incidence.SparseIncidenceBackend`, which
is also the parity oracle (`tests/test_incidence.py`). Measurements:
[incidence.md](incidence.md).

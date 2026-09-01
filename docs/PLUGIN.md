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
     // out_mode:  1 = coverage sum -> A0,  2 = clamped sum -> cnt
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
`loss_owner, set_tbl2/use_fmt, map_hist, host_bytes`, and the μ-fold
multiple-covering group `set_mu/get_mu/ball_gather_mu/count_deficit`
(see `docs/multi-coverage.md`; with `dct_set_mu(μ)` active, init_modes
1/2 become the deficiency predicates `[cnt<μ]`/`[1<=cnt<=μ]`, and
`dct_ball_gather` targets 0/1 act as roles "deficient"/"critical" —
both collapse to the legacy literals at μ=1; `dct_free` resets μ to 1).

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

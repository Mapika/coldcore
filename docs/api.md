# Python API

A hand-written tour of the public host-side surface, written from the
code (there is no autodoc step). For the C ABI the CUDA plugins speak,
see [PLUGIN.md](PLUGIN.md); this page is everything above that seam.

Stability: `coldcore.protocol.Backend` and `coldcore.space.ProductSpace`
are the surface third-party problems bind to and are meant to stay put.
The `Searcher` pass list grows; its existing signatures keep their
defaults, so `descend()` with no keywords is the proven record driver
and nothing more.

```
src/coldcore/
  space.py        ProductSpace          index <-> digit conversions
  protocol.py     Backend               the seam the search core runs on
  reference/      Ref*Backend           pure-numpy brute-force backends
  gpu.py          GpuBackend            ctypes binding of a plugin .so
  patterns.py     support_patterns()    host-side ball/support tables
  search.py       Searcher              greedy, peel, LNS, anneal, descend
  io.py           read/write_solution   digit-row solution files
  incidence.py    SparseIncidenceBackend explicit-incidence family (host)
  incidence_gpu.py GpuIncidenceBackend  the same over plugin ABI 2
  problems/       CoveringDesignBackend covering designs C(v,k,t)
  cli.py          main()                the `coldcore` console script
```

---

## `coldcore.space.ProductSpace`

The finite product space `X = Z_{a_0} x … x Z_{a_{n-1}}`, little-endian:
`index = sum_j digit_j * prod_{i<j} a_i`.

```python
ProductSpace([9, 9])        # mixed-radix axes
ProductSpace(3, 4)          # homogeneous shorthand: 3^4
```

| member | meaning |
|---|---|
| `.axes`, `.n`, `.size` | radices, number of axes, `prod(axes)` |
| `.q` | the common radix, or `None` if the axes are mixed |
| `.word_index(words)` | `(k, n)` digit array → `int64` indices |
| `.index_word(idx)` | `int64` indices → `(k, n)` `uint8` digit array |
| `.all_words()` | the whole space as digit rows (tiny spaces only) |

---

## `coldcore.protocol.Backend`

One problem instance, resident somewhere (GPU plugin or numpy). It owns
two dense fields over the whole space:

- **`cnt`** — the exact multiplicity field: how many solution points
  cover each point;
- **a map** — scratch, written by `gain_map()`/`loss_map()` and read by
  the `map_*` queries.

### Attributes a backend must set

| attribute | meaning |
|---|---|
| `space` | a `ProductSpace` |
| `extract_cap` | largest number of entries one `map_extract` may return |
| `indep_dist` | removals farther apart than this (digit distance) cannot affect each other's loss — `2R` for metric-ball problems; `None` disables batched peeling |
| `supports_hist` | `False` if `map_hist` is unimplemented; the greedy falls back to threshold escalation |

### Methods a backend must implement

| method | contract |
|---|---|
| `recount(idx)` | rebuild `cnt` exactly from the solution point indices |
| `gain_map()` | write, for **every** point `w`, the exact marginal gain of adding `w` |
| `loss_map()` | write the exact marginal loss field; read at incumbents it is their private coverage |
| `ball_update(idx, delta)` | incremental `+1`/`-1` on `cnt`; does not touch the map |
| `ball_gather(idx, target)` | per candidate, the exact count of points in its neighborhood with `cnt == target` (`0` = fresh gain, `1` = private coverage) — always current, never stale |
| `count_eq(target)` | how many points have `cnt == target` over the whole space |
| `map_max()` | maximum map value |
| `map_hist(nbins, vmax)` | positive values `v` binned at `(v-1)*nbins//vmax`, clipped to the last bin |
| `map_extract(thr, cap=None)` | `(idx, val, found)` for map values `>= thr`, **unordered**; `found` may exceed `len(idx)` when the cap bites |
| `map_read_at(idx)` | map values at those indices |

Provided by the base class: `uncovered()` (= `count_eq(0)`),
`word_index`/`index_word` passthroughs, `close()`, and the context
manager protocol (`with backend: …`).

The split between `gain_map()` (whole space, one transform, possibly
stale by the time a candidate is popped) and `ball_gather()` (a handful
of candidates, exact, now) is what lets the greedy be lazy *and* exact.
A backend that gets that split wrong is the usual source of drift.

### Writing your own

`examples/03_custom_problem.py` is a complete, commented implementation
of every method above for a problem coldcore does not ship — including
the note on how to restrict which points may be chosen (mask the gain
map; candidates only enter the search through `map_extract`). Start
there rather than from an existing backend.

---

## `coldcore.reference` — brute-force backends

Pure numpy, `O(|X|^2)`, capped at `2^16` points. Ground truth for the
GPU parity tests and the way to exercise the search core without a
device.

| class | problem |
|---|---|
| `RefHammingBackend(axes, R, mu=1)` | covering codes, mixed-radix alphabets |
| `RefCoveringBackend(q, n, R, mu=1)` | homogeneous shorthand for `K_q(n,R)` |
| `RefTorusBackend(axes, R, mu=1)` | Chebyshev balls on a torus (king domination) |
| `RefGridBackend(axes, R, mu=1)` | Chebyshev balls on an open-boundary grid |
| `RefLeeBackend(axes, R, mu=1)` | Lee-metric covering codes |

All of them subclass `reference.base.BruteForceBackend`, which
implements the entire protocol generically and leaves exactly one thing
to the subclass: `_covers(a, b)`, the coverage predicate between
broadcastable digit arrays. `mu > 1` asks for µ-fold covering (every
point covered at least µ times).

Everything that only reads the dense `cnt` / `map` fields (the µ-fold
role predicates, `count_eq`, `uncovered`, `map_max`, `map_hist`,
`map_extract`, `map_read_at`, `read_cnt`) lives once in
`protocol.DenseFieldBackend`, shared with the CPU backend below.

---

## `coldcore.dp` — the fast CPU backend

| name | meaning |
|---|---|
| `NumpyDPBackend(axes, R, problem="hamming", extract_cap=1<<22, mu=1, seed_shuffle=None, dtype=np.int32)` | one cell in host memory, transformed by the plugin's axis DP in numpy |

The same grade-truncated axis DP the CUDA plugin runs (distance-count
layers for `hamming`/`lee`, windowed sums for `torus_linf`/`grid_linf`),
vectorised over fibers: `O(n |X| (R+1))` instead of the reference's
`O(|X|^2)`, so CPU campaigns reach `|X|` of order 10⁷ instead of 2¹⁶.
Fields, queries and even `map_extract` ordering are **bit-identical to
the reference**, so a `Searcher` run on `cpu` reproduces the `ref` run
move for move (`tests/test_dp_backend.py`). Unlike the plugin it is not
bound by the packed support table's `ax_j <= 16` / weight ≤ 8 caps.

---

## `coldcore.incidence` — the explicit-incidence family

For problems whose candidates and covered points are different finite
sets, with coverage given as an `(n_cand, d)` incidence matrix (row `c`
= the universe elements candidate `c` covers). Design and measurements:
[incidence.md](incidence.md).

| name | meaning |
|---|---|
| `IncidenceSpace(size)` | one-axis stand-in for `ProductSpace` (`n=1`, `axes=[size]`; `index_word` returns `(m, 1)` int64 rows) so the search core runs unchanged |
| `SparseIncidenceBackend(inc, n_univ, mu=1, extract_cap=1<<22, seed_shuffle=None)` | host backend; `cnt` over the universe, `map` over the candidates; `verify(idx)` / `coverage(idx)` recompute from the incidence list alone |
| `incidence_gpu.GpuIncidenceBackend(inc, n_univ, mu=1, extract_cap=1<<22, inc_mode=None, plugin_path=None, hbm_budget=None)` | the same over `libcoldcore_sparse.so`; `inc_mode` 0 = HBM, 1 = pinned CPU memory, `None` = by size against the budget |

`read_cnt` on these backends addresses the *universe*, not the
candidate set (`space.size` is the candidate count).

Optional locality hooks (`protocol.py`, feature-tested by the search
core, product-space code used when absent): `distance(a, b)`,
`neighbours(idx, radius=1, m=None, rng=None)`,
`deficient_universe(limit, rng=None)`, `universe_distance(idx, univ)`.
The incidence backends define `deficient_universe`; the covering-design
layer adds the block metric for the other three (plus
`candidates_covering(univ_idx)`) and sets `indep_dist = k - t`.

| `WeightedIncidenceBackend(indptr, indices, w, n_univ, uw=None, mu=1, ...)` | weighted, variable-length rows (CSR): `cnt += w` per chosen candidate, deficiency `sum uw * max(0, mu - cnt)`, exact weighted gain/loss marginals |
| `incidence_gpu.GpuWeightedIncidenceBackend(...)` | the same over `sparse_init_csr` |

## `coldcore.problems.perm_groups` / `design_orbits` — group-invariant designs

| name | meaning |
|---|---|
| `PermGroup(v, gens)`, `parse_design_group(v, spec)`, `builtin_specs(v)` | permutation groups by generators; `elements()`, `order()`, `subset_orbits(r)` (an `OrbitPartition`) |
| `DesignOrbits(v, k, t, group)` | block orbits, t-orbits, quotient weights; `quotient_csr()`, `expand(idx)`, `block_count(idx)` |
| `QuotientDesignBackend(v, k, t, group, mu=1, short_orbits=False)` | the quotient as a host backend (candidates = orbits) |
| `orbit_search(v, k, t, group, budget_s, device=...)` | greedy / peel / descent on the quotient (CPU or GPU); returns expanded blocks |
| `orbit_search_points(...)` | `SymmetricSearcher` over the block orbits of the full backend (counts blocks) |
| `release(v, k, t, blocks, budget_s, ...)` | the free descent seeded with a block set |

## `coldcore.problems.covering_design` — covering designs

| name | meaning |
|---|---|
| `CoveringDesignBackend(v, k, t, mu=1, device="cpu")` | C(v,k,t): candidates = k-subsets (colex rank), universe = t-subsets; returns the host class (`device` `"cpu"` / `"ref"`) or the GPU one (`"gpu"`) |
| `.blocks`, `.rank_block(idx)`, `.block_rank(blocks)` | the `(C(v,k), k)` block table and the rank conversions |
| `.read_cover(path)`, `.write_cover(path, idx)` | LJCR format: one block per line, 1-based, space separated |
| `.verify(idx)` | feasibility recomputed from the blocks, independent of the incidence matrix |
| `all_blocks`, `build_incidence`, `rank_subsets`, `unrank_subsets`, `binom_table`, `read_cover`, `write_cover` | the module-level pieces |

---

## `coldcore.backends` — the factory

| name | meaning |
|---|---|
| `make_backend(kind, axes, R, problem="hamming", mu=1, plugin_path=None, **kw)` | build one backend; `kind` in `("ref", "cpu", "gpu")` |
| `parse_cell(spec)` / `cell_label(...)` | cell spec strings: `K3(4,1)`, `lee:5x5:1`, `torus_linf:9x9:1`, `K2(6,1)mu3` |
| `backend_for_spec(kind, spec, ...)` | `make_backend` straight from a spec string |

---

## `coldcore.gpu` — the plugin binding

| name | meaning |
|---|---|
| `GpuBackend(axes, R, problem="hamming", plugin_path=None, extract_cap=1<<22, layers_mode=None, cnt_mode=None, use_owner=False, mu=1)` | one cell resident on the GPU / host-coherent memory |
| `GpuCoveringBackend(q, n, R, **kw)` | homogeneous `K_q(n,R)` convenience wrapper |
| `load_plugin(path)` | load and cache a plugin shared library |
| `default_plugin_path()` | `COLDCORE_PLUGIN_PATH`, else `<repo>/build/plugins/libcoldcore_covering.so` |

`problem` is one of `hamming`, `torus_linf`, `grid_linf`, `lee`.
`layers_mode` / `cnt_mode` of `None` means the automatic HBM-vs-LPDDR
placement policy; pass explicit modes to override it. The binding is
ctypes and the ABI is a singleton per loaded `.so`: **one live cell per
plugin per process.** Optional ABI capabilities are feature-tested by
symbol presence, so an older plugin still loads with fewer features.

---

## `coldcore.search.Searcher`

```python
Searcher(backend, seed=1, deadline=None,
         on_log=None, on_improve=None, on_solved=None, yield_check=None)
```

Callbacks, all optional:

| callback | fires |
|---|---|
| `on_log(str)` | progress lines |
| `on_improve(searcher)` | new best-by-uncovered snapshot |
| `on_solved(M, idx)` | a feasible solution of size `M` passed a full recount — **the supported way to read results out of a run**; hook record gates and checkpoint writers here |
| `yield_check()` | called periodically; put GPU cooperation (pause/yield) here |

State: `.code` (current solution as `int64` indices), `.uncov`,
`.best_code`, `.best_uncov`, `.stats` (a counter dict —
`transforms`, `placements`, `removals`, `rounds`, `recounts`, `moves`,
`accepts`, `restarts`).

`deadline` is an absolute `time.time()` value. Passes check it and stop
early; a deadline that is already close can leave `.code` mid-ruin,
which is the second reason to read results from `on_solved`.

### Entry points

| method | what it does |
|---|---|
| `load(code)` | adopt a solution (1-D indices or `(M, n)` digit rows), recount, snapshot |
| `verify_full()` | from-scratch recount; returns the exact uncovered count |
| `greedy_fill(target_m, batch=64, refresh_every=None)` | exact lazy greedy until covered / `target_m` / deadline (`refresh_every` defaults to the backend's `greedy_refresh_every`, else 512; a backend with `field_greedy = True` takes every placement from a fresh gain field). With `Searcher(..., repair="random")` every insertion is instead a uniformly random candidate covering one deficient word (ablation only) |
| `losses()` | one loss transform; returns the exact loss at each incumbent |
| `ruin(k, mode="low")` | remove `k` points; modes `low`, `random`, `cluster`, `hole`, and `deficit` (aims at the actual uncovered words; needs `read_cnt`, else falls back to `hole`) |
| `uncovered_words(limit=…)` | exact indices of currently deficient words via a chunked `read_cnt` scan, cached per distinct solution |
| `peel()` | remove zero-loss (fully redundant) points to a fixpoint; feasibility preserved |
| `lns(M, round_budget_s, kmin=8, kmax=None, log_every=20, select="fixed", deficit=False, seed=None, stall_s=None)` | ruin-and-recreate at fixed `M` until feasible, out of budget, or (with `stall_s`) that many seconds without a decrease of the best uncovered count; `select="adaptive"` re-weights ruin operators by realised reward; `deficit=True` adds the deficit operator to the draw |
| `anneal(M, budget_s, mode="lahc", neighbourhood="pool", pool_refresh=64, …)` | swap-neighbourhood polish at fixed `M` that accepts regressions. `neighbourhood="shell"` swaps only around the just-opened hole (drop cheapest-loss incumbents, add from the dropped word's distance-1/2 shell + top of the gain map) — the wall-regime mode; `"auto"` starts on `pool` and switches when the accept rate collapses. `pool_refresh="auto"` sets the pool cadence from measured transform cost |
| `two_swap(budget_s=…, depth=2)` | exact, budget-capped depth-2 (add, drop) endgame search for states where no single swap improves; deterministic under seed |
| `portfolio(M, budget_s, streams=4, …)` | time-sliced restart portfolio at fixed `M` around a shared incumbent |
| `break_wall(M_target, budget_s, …)` | the endgame driver: rotates focused-ruin LNS, shell polish, and two-swap bursts at one fixed `M` with restart-from-best; trajectory in `stats["wall"]`. Measured the strongest wall configuration (15/15 target hits vs 10/15 for default `descend` on the benchmark cells; see `docs/search-passes.md`) |
| `descend(floor=1, notch_budget_s=300.0, step0=4, …, select="fixed", polish=None, polish_share=0.5, portfolio=1, seed=None, salvage=False, carry=False, patience=False, persist=False, fail_limit=3, reserve_s=30.0, min_notch_s=20.0)` | the record driver: notch `M` down, repair each notch, halve the step on failure, double it on a fast success. The schedule keywords are described in `docs/search-passes.md` ("Descent schedule (v0.3)"); `coldcore solve` turns them on |

`descend()` requires a feasible starting solution (`uncov == 0`) and
raises otherwise. With no keyword arguments it is bit-for-bit the pass
that set the covering-code records; `select`, `polish` and `portfolio`
opt into the newer passes.

Every pass that takes `seed=` runs under a scoped RNG, so a single pass
can be reproduced without replaying the whole run.

---

## `coldcore.symmetry` — groups and orbit-restricted search

Full treatment (concepts, the win/loss measurements, the K_8(4,2)
analysis) in [symmetry.md](symmetry.md); the surface:

| object | meaning |
|---|---|
| `GroupElement(axes, perm, vperms)` | one automorphism: `(g.x)_j = vperms[j][x_{perm[j]}]`. `apply_words`, `apply_index(space, idx)`, `compose`, `inverse`, `permutation(space)` |
| `Group(space, gens)` | a group by generators. `orbits()` → `OrbitPartition`, `orbit_of(idx)`, `elements(cap)`, `order()`, `is_automorphism_of(backend)`, `join`/`+` |
| `OrbitPartition` | CSR partition: `labels`, `members`, `starts`, `sizes`, `reps`, `orbit(o)`, `segment_sum(field)` |
| `SymmetricSearcher(backend, group, …)` | the searcher whose atoms are orbits: `greedy_fill`, `peel`, `ruin`, `lns`, `descend`, `add_orbit`/`drop_orbit` (exact marginals), `gain_bounds`/`loss_bounds`, `free_searcher()`, `summary()` |
| `orbit_solve(backend, group, budget_s)` | greedy → peel → orbit descent in one call; returns `(code, searcher)` |
| `parse_group(space, spec)` | build a group from a spec string (`translate:1011+shift:2`, `diag`, `rot`, `sym`, `mono`, `iso:lee`, `trivial`) |
| constructors | `translation`, `translation_group` (= `linear_code_translations`), `diagonal_translations`, `value_shift`, `value_shifts`, `coordinate_permutation`, `coordinate_rotation`, `coordinate_symmetric`, `value_permutation`, `monomial_group`, `isometry_group`, `trivial_group` |

A solution produced by `SymmetricSearcher` is a plain index array: every
existing verifier, writer and `Searcher` accepts it unchanged.

---

## `coldcore.symmetry_exact` — exhaustive settlement

| object | meaning |
|---|---|
| `IsomorphFreeSearch(backend, group=None, iso_depth=4, verify=True, strong_bound=0, table=None)` | decision procedure for one tiny cell (needs a reference backend) |
| `binary_monomial_table(space)` | `(index-permutation table, root reps)` for the full monomial group of `Z_2^n`, built by broadcasting rather than by enumerating elements — pass it as `table=` |
| `.decide(M, node_limit=None, time_limit=None)` | `("feasible", witness, stats)` / `("infeasible", None, stats)` / `("unknown", None, stats)` |
| `.minimum(hi, lo=1, **kw)` | walk the ladder down from `hi`; returns `(M, witness, trace)` |
| `.estimate_nodes(M, samples, seed)` | Knuth random-dive estimate of the backtrack tree size |
| `settle(backend, hi, **kw)` | one-call `minimum` |

Sized for `|X| <= ~2^12` and groups of at most `MAX_GROUP = 2^17`
enumerated elements; `table=` lifts that to whatever fits in memory.
`"infeasible"` is a proof; `"unknown"` means a budget stopped the search
and nothing is claimed. `strong_bound` is opt-in and measured to be a
wash — see [symmetry.md](symmetry.md) §6.

---

## `coldcore.io`

| function | meaning |
|---|---|
| `write_solution_atomic(path, words)` | write `(M, n)` digit rows via tmp+rename (crash-safe) |
| `read_solution(path, q=None, n=None)` | read digit rows → `(M, n)` `uint8`, validating against `q`/`n` |

The format is one point per line, digits separated by spaces — the same
files the upstream record repo uses.

---

## `coldcore.patterns`

Host-side support-table builders shared by the GPU backends:
`support_patterns(axes, R, problem)` and `support_patterns_ox(...)`
enumerate the per-axis delta patterns a plugin's axis pass walks;
`ball_size(q, n, R)` and `lee_ball_size(axes, R)` are the closed forms.
`PROBLEM_IDS` maps problem names to the ABI's integer problem ids.
Plugin authors need this module; users of the search core do not.

---

## CLI

Installed as the `coldcore` console script, and equivalently
`python -m coldcore`.

```
coldcore solve --backend {ref,cpu,gpu} [--plugin PATH] [-p key=value ...]
               [--floor N] [--hours H] [--seed S] [--seed-file FILE]
               [--out FILE] [--notch-budget S] [--step0 N]
               [--ruin-select {fixed,adaptive}]
               [--polish {none,anneal,lahc}] [--polish-share F]
               [--portfolio N] [--wall M]
               [--start {auto,greedy,cosets}] [--coset-share F]
               [--coset-slice F] [--repair {exact,random}]
               [--legacy-descent]
               [--symmetry SPEC] [--symmetry-polish]
coldcore info [--plugin PATH]
```

`solve` = greedy build (or `--seed-file`) → peel → (on homogeneous
Hamming cells with `--hours`: coset start, [cosets.md](cosets.md)) →
notch descent, writing every improvement to `--out` atomically.
`--legacy-descent` restores the v0.2 descent schedule; `--repair random`
is the repair ablation of [search-passes.md](search-passes.md). `info` prints the
version and probes a plugin `.so` for its init entry points.

`--symmetry SPEC` replaces the word-level driver with the
orbit-restricted one (a solution is a union of orbits of the group
`SPEC` names); `--symmetry-polish` continues with the word-level descent
afterwards. See [symmetry.md](symmetry.md).

Problem parameters (`-p`, repeatable, comma-separated pairs allowed):
`problem=`, `q=`/`n=` or `axes=9x9`, `R=`, `mu=`, `group=` (the same
spec as `--symmetry`). See [getting-started.md](getting-started.md) for
worked invocations.

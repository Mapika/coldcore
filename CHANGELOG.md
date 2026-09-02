# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/); this project
follows semantic versioning.

## [0.2.1] — 2026-09-02

### Fixed
- Tests only. The locality-hook trajectory pins in `tests/test_incidence.py`
  end with a LAHC shell polish whose move sequence differs by an ULP
  between aarch64 and x86-64; they are now per-architecture pins, as the
  polish pins already were, with the x86-64 digests taken from CI. The
  `auto` neighbourhood test in `tests/test_search_wall.py` gets a 20 s
  budget so a loaded runner cannot exhaust it before the switch.

## [0.2.0] — 2026-09-02

### Added
- **Explicit-incidence backend family** (`src/coldcore/incidence.py`,
  `src/coldcore/incidence_gpu.py`): candidates and universe are
  different finite sets and coverage is an `(n_cand x d)` incidence
  matrix. `SparseIncidenceBackend` (numpy) and `GpuIncidenceBackend`
  (CUDA) implement the unchanged backend protocol, µ-fold roles
  included; `IncidenceSpace` stands in for `ProductSpace` so the search
  core runs as is. Design notes and measurements: `docs/incidence.md`.
- **Plugin ABI 2** (`src/plugins/sparse/sparse.cu`, flat `sparse_*`
  symbols, `build/plugins/libcoldcore_sparse.so`): the GPU side of that
  family — incidence matrix in HBM or in pinned CPU memory read over
  C2C/ATS, fields in HBM, the same query semantics as ABI 1
  (`docs/PLUGIN.md`, "ABI 2: explicit incidence"). Independent of the
  `dct_*` contract.
- **Covering designs C(v,k,t)** (`src/coldcore/problems/covering_design.py`,
  `CoveringDesignBackend(v, k, t, mu, device)`): blocks and incidence via
  the combinatorial number system, LJCR block-list files, an
  independent verifier. CLI: `coldcore solve --problem covering_design
  -p v=.. -p k=.. -p t=.. --backend ref|cpu|gpu`; `--seed-file` and
  `--out` use the block-list format for this problem through the new
  optional backend hooks `read_solution_file` / `write_solution_file`.
- `--problem NAME` on `coldcore solve` as an alias of `-p problem=NAME`;
  `coldcore info` also probes the sparse plugin.
- `tests/test_incidence.py`: oracle parity, GPU parity (both incidence
  placements), search to known optima, file round-trip.
- **Optional locality hooks** on the backend protocol (`distance`,
  `neighbours`, `deficient_universe`, `universe_distance`; documented in
  `protocol.py`). The cluster/hole/deficit ruins, the `shell` polish
  neighbourhood and the batched peel use them when a backend defines
  them and run the unchanged product-space code otherwise (trajectory
  pins in `tests/test_incidence.py` and `tests/test_search_determinism.py`
  are unmoved). Covering designs define them through the block metric
  (`k - |A n B|`, single-swap neighbours, exact deficient t-subsets,
  `indep_dist = k - t`). `scripts/bench_incidence_ops.py` compares the
  operators; table in `docs/incidence.md`.

- **Group-invariant covering designs** (`problems/perm_groups.py`,
  `problems/design_orbits.py`; `--symmetry SPEC` on `covering_design`,
  `--symmetry-polish`, `--orbit-budget`): permutation groups by
  generators with built-ins (cyclic, cyclic-1/-2, dihedral, prod:AxB,
  affine, mult, explicit cycles), orbits of k- and t-subsets, the
  quotient problem as a **weighted CSR incidence cell**
  (`WeightedIncidenceBackend`, `GpuWeightedIncidenceBackend`,
  `sparse_init_csr` in plugin ABI 2), orbit search on the quotient or
  — the product-space precedent — `SymmetricSearcher` over the block
  orbits of the full cell, and the release to the free search.
  Validation and a group sweep: `docs/incidence.md`.

### Fixed
- `SymmetricSearcher._lns` never ruined a feasible solution down to the
  target size, so `descend` reported every notch failed unless greedy
  plus peel already sat at the optimum. It now drops the cheapest orbits
  to the target before the round loop.

## [0.1.1] — 2026-09-01

### Fixed
- `docs/symmetry.md` referred to a script that is not part of the
  repository.
- `coldcore info` after a non-editable install (`pip install .`) reported
  only "plugin: not loadable"; it now says how to build the CUDA plugin
  or point `COLDCORE_PLUGIN_PATH` / `--plugin` at it.

### Changed
- `scripts/probe_k8_42.py`, a one-off probe tied to a private scratch
  directory, is no longer shipped.

## [0.1.0] — 2026-08-29

Public release snapshot (github.com/Mapika/coldcore, tag `v0.1.0`). This
release includes everything listed below, up to and including the
2026-08-24 block.

### Added
- **Fast CPU backend** (`src/coldcore/dp.py`, `NumpyDPBackend`) — the CUDA
  plugin's own axis DP, in numpy. Grade-truncated distance-count layers
  for `hamming`/`lee`, circular / boundary-clamped window sums for
  `torus_linf`/`grid_linf`, vectorised over fibers: `O(n |X| (R+1))`
  instead of the reference backends' `O(|X|²)`, with incremental
  `ball_update`/`ball_gather` over the plugin's support enumeration and
  µ-fold support throughout. Reachable as `--backend cpu`.
  - **Parity is the gate.** `tests/test_dp_backend.py` compares every
    field, every `map_*`/`count_*` query, every ball and every
    incremental operation against all four `Ref*Backend`s over
    randomised configurations with µ ∈ {1,2} — and then compares a whole
    ruin/recreate *move trace*, which agrees only because `map_extract`
    ordering agrees too. `cpu` is a drop-in for `ref`, not merely an
    equal-quality substitute.
  - **Measured** (`scripts/bench_backends.py`, raw records in
    `docs/bench_backends.json`, tables in `docs/benchmarks.md`): gain
    transform 400 ms → 0.48 ms at K_3(8,2) (831×), 238 → 0.59 ms at
    K_2(12,2) (404×), 2816 → 13.3 ms at K_2(16,3) (212×); the 20-74×
    floor is the cells small enough to sit in cache. End to end at
    K_2(16,3) the reference cannot finish the greedy build in 60 s while
    `cpu` completes the descent to M=256 in 3.1 s; K_3(12,3), K_5(8,3)
    and torus 64³ — impossible for the reference — descend in the same
    budget.
  - Not bound by the plugin ABI's `ax_j ≤ 16` / weight ≤ 8 support-table
    caps: the support set is enumerated into (position, delta) columns
    and asserted equal to the packed table wherever both apply.
- **Backend factory** (`src/coldcore/backends.py`) — `make_backend`,
  `parse_cell`, `cell_label`, `split_cells`. One home for `ref`/`cpu`/
  `gpu` and for cell spec strings (`K3(8,2)`, `lee:5x5:1`,
  `torus_linf:9x9:1`, `K2(6,1)mu3`).
- **Campaign runner** (`src/coldcore/campaign.py`, `coldcore campaign`) —
  the loop every campaign has hand-rolled: N cells, one worker process
  each, atomic solution checkpoints, per-cell logs and JSON records, a
  collected `campaign.json`, and **resume** from the checkpoint on a
  rerun. Deliberately thin (a `ProcessPoolExecutor`, no daemon); one
  process per cell is also required by the GPU plugin's singleton ABI.
  A crashed cell is a record with `error` set, never a dead campaign.
- **`docs/performance.md`** — every measurement on this page in one
  place, including the things that did *not* pay, plus the GPU-side
  roadmap that this pass deliberately did not touch.
- **Move-trace pins** (`tests/test_search_determinism.py`) — frozen
  SHA-256 digests of whole traces for greedy/peel/every ruin operator,
  the shell polish, the two-swap endgame and the orbit layer, taken from
  the pre-optimisation core. The pass sequences are clock-free by
  construction, so the digests record decisions rather than wall time.
- **`scripts/bench_backends.py`**, **`scripts/bench_orbit.py`**,
  **`scripts/profile_search.py`** — the harnesses behind the numbers
  above.

### Changed
- `ball_gather` on the CPU backend picks between building the
  whole-space coverage-role mask and gathering `cnt` values by measured
  cost rather than always doing the former; the answers are identical
  and a forced-path test proves it. Shell-polish throughput +5.5% to
  +14.3% across K_5(8,3), K_7(8,3), K_3(8,2), K_3(12,3).
- `Searcher.greedy_fill` builds its bound heap with one `.tolist()`
  instead of per-element numpy scalar indexing (the only search-core
  line in the profile's top ten), and `_shell1` uses per-axis
  displacement tables built once instead of rebuilt per move. Move
  traces unchanged — that is what the new pins are for.
- `SymmetricSearcher` materialises its point array on demand and undoes
  a greedy probe arithmetically instead of re-scanning the whole space
  for a deficit it already knows. Orbit-layer benchmark (16 cells,
  clock-free, `scripts/bench_orbit.py`): see `docs/performance.md`.
- The µ-fold coverage-role predicates and every dense `cnt`/`map` query
  moved from `reference.base.BruteForceBackend` into
  `protocol.DenseFieldBackend`, shared with the CPU backend.

### Added
- **Symmetry layer** (`src/coldcore/symmetry.py`, `src/coldcore/symmetry_exact.py`,
  `docs/symmetry.md`) — group-invariant construction, orbit-restricted
  search, and isomorph-free exhaustive settlement. Every space the
  framework searches has a large automorphism group; until now the search
  core ignored it.
  - **Groups as an IR concept**: an element is a *(coordinate
    permutation, per-coordinate value permutations)* pair — the general
    element of the wreath product `(∏_j S_{a_j}) ⋊ S_n` — with exact
    composition, inversion, a mixed-radix action on word *indices* that
    never materialises words, and a vectorised whole-space orbit
    partition (label propagation with pointer jumping) in CSR form.
    Library subgroups: translations by any vector set (over `Z_q`: the
    cosets of a linear code), diagonal translations, per-axis cyclic
    value shifts, coordinate rotations/`S_n`, the full monomial group,
    and per-family isometry groups. Spec grammar for CLI/plugin use:
    `translate:1011+shift:2`, `diag`, `rot`, `sym`, `mono`, `iso:lee`.
  - **`SymmetricSearcher`**: greedy / peel / notch descent over
    **orbits** instead of words — a solution is a union of chosen
    orbits, which is the 1990s group-invariant-code construction,
    mechanised. Exact throughout: an orbit's gain is measured by
    applying it and reading the exact deficit back, and the gain map's
    segment sum over an orbit is a genuine, monotonically decreasing
    upper bound on it, so the lazy-greedy bound heap is *exact* greedy
    with one whole-space transform amortised over the whole build. Peel
    uses the loss-map sum as an exact necessary filter and re-checks by
    dropping. Correctness never depends on the group being an
    automorphism group — a bad group only restricts the search to a bad
    subfamily. `free_searcher()` hands a group-invariant solution to the
    ordinary word-level searcher.
  - **Measured, both ways** (`scripts/bench_symmetry.py`, raw records in
    `docs/bench_symmetry.json`; time to a *verified* solution at the
    target size, 5 seeds, 30 s cap, mean over the seeds that arrived).
    Orbit search reaches structured optima 3–4 orders of magnitude
    faster and on every seed: K_2²(6,1)=20 (9438×, free 3/5 seeds),
    μ=2 Golomb–Welch Z_5²=10 (9613×), K_4²(4,1)=48 (2856×, free 1/5),
    K_2³(8,1)=96 (>1754×, free 0/5), K_2(6,1)=12 (1076×, free 1/5),
    K_2⁴(8,1)=128 (792×, free 4/5) — the μ-fold rows corroborating the
    campaign finding that construction beats free search there. Honest
    negatives in the same table: on K_4(4,1), Lee 7×7 and the 11×11 king
    torus the optimum is *not* a union of orbits and orbit search never
    reaches it (32 vs 24, 14 vs 12, 22 vs 16) while free search does —
    the group has to divide the answer; and on a |X|=8 cell orbit search
    is measurably *slower* (0.7×), the bookkeeping not being repaid.
    (Methodology note recorded in `docs/symmetry.md`: `Searcher.descend`
    reserves 30 s of its deadline before starting a notch, so a free run
    given only its working budget never descends at all. The first
    version of this table made that mistake and overstated the win by
    ~1000×; the reserve is now a named constant in both the benchmark and
    the tests.)
  - **`IsomorphFreeSearch`**: exhaustive decision procedure for tiny
    cells — coverage branching on the least-covered word (complete),
    root orbit-representative plus inner stabiliser-orbit reduction of
    the branch set (exact isomorph rejection; orderly generation and
    canonical augmentation are discussed and rejected in the module
    docstring), and counting bounds. Returns `feasible` + witness,
    `infeasible`, or `unknown` — never a guess. Settles, from scratch in
    the test suite, K_2(3,1)=2, K_2(4,1)=4, K_2(5,1)=7, K_3(3,1)=5,
    K_3(4,2)=3, K_4(3,1)=8, Lee Z_5²=5, king 6×6 torus=4, king 9×9
    torus=9, king 8×8 board=9, K_2²(3,1)=4, K_2³(3,1)=6, K_2²(4,1)=8,
    K²(Lee Z_5²)=10; and outside the suite K_2(6,1)=12 (1.2e6 nodes,
    7.3 s) and K_3(4,1)=9. A group that is not an automorphism group is
    refused rather than yielding an unsound proof.
  - **K_8(4,2): a measured "out of reach", not a guess.** At a measured
    1.15e6 nodes/s, Knuth random-dive estimation (20000 dives, calibrated
    against completed searches in-test to within a few percent) puts the
    M=22 tree at ~5.0e47 nodes ≈ 1.4e34 years; even a hypothetical
    perfect isomorph rejection under the full monomial group
    (8!⁴·4! = 6.3e19) leaves ~7.8e27 nodes. What would be needed
    (LP/surplus node bounds rather than more symmetry) is written down in
    `docs/symmetry.md`. The orbit-restricted upper-bound probe over every
    order-2 translation group of Z_8⁴ reaches M=32, against M=35 for a
    free greedy+peel — better, and nowhere near 22.
  - **Faster exhaustive settlement at μ > 1**: the state is now μ
    bitmasks (`lay[k]` = words covered ≥ k+1, updated by
    `lay'[k] = lay[k] | (lay[k-1] & B)`) instead of a whole-space numpy
    count array copied per node — 2.3× on K(8,2,2) M=14 for a
    bit-identical tree (7 802 251 nodes either way).
  - **`binary_monomial_table(space)`** plus an `IsomorphFreeSearch(...,
    table=)` escape hatch: the full monomial group of `Z_2^n`
    (`n!·2^n` — 10 321 920 at n=8) as an index-permutation table built by
    broadcasting, far past what element-by-element enumeration reaches.
    On K(8,2,2) M=14 it cuts the tree from 7.80e6 nodes (|G|=2048) to
    1.21e5 — **64×** — and the wall time from 12.1 s to 0.31 s. The root
    and depth-1 branch sets are cached, since both are invariant.
  - **`strong_bound`** (opt-in, default off): the exact greedy bound,
    replacing `|B|` in `r·|B| ≥ D` with the largest `|B ∩ short|` any
    ball still achieves. Correct, and measured to be a **wash** on both
    target cells — K(8,2,2) M=14: −14% nodes for +5% wall; K_8(4,2)
    M=22: ~1000× fewer nodes for ~820× lower throughput. Recorded as a
    negative result because it sharpens what a *useful* bound would have
    to look like (incremental along the branch, not a whole-space
    rescan).
  - **K(8,2,2) settlement attempt** (`docs/symmetry.md` §5). Proved
    `K(8,2,2) ≥ 15` exhaustively under the full monomial group; verified
    the existing 18-word cover independently. The cell is **not
    settled**: Knuth estimates (calibrated on this cell — M=14: 5.83e6
    predicted vs 6.70e6 actual) put M=17 at 3.2e17 nodes (~27 000
    core-years) and M=16 at 3.3e14 (~27 core-years), both far past any
    budget. M=15 (~5e10, heavy-tailed) is inside budget and was launched.
    Seeded LNS found no 17-word cover in 8 seeds — a flat wall at
    deficit 12, which is evidence, not proof.
  - **`monomial_table(space, per_axis=, coord=, max_bytes=)`** —
    `binary_monomial_table` generalized to any mixed-radix product
    space, so the broadcast construction now serves the q-ary table too
    (`binary_monomial_table` is a wrapper over it). Verified row-for-row
    against `Group.elements()` on Z_2³, Z_3², Z_4², Z_2⁴, Z_3³ and Z_5²;
    handles mixed radix (only equal-size axes may be permuted), which
    `monomial_group` cannot express at all. `affine_perms(q)` gives the
    `AGL(1,q)` per-axis fallback for when `(q!)^n · n!` will not fit —
    a genuine subgroup, so more nodes but never a different verdict —
    and `max_bytes` refuses to build rather than being OOM-killed.
  - **Four tight q-ary μ-fold cells estimated** (`docs/symmetry.md` §6).
    Only `K_4(3,1;4)` at M=26 is tractable (3.34e9 nodes, 0.14
    core-days) and is running; `K_4(3,1;4)` M=27 (7060 core-days),
    `K_5(3,1;4)` M=39 (1.1e5), `K_4(4,1;3)` M=61 (4.1e11) and
    `K_5(4,2;4)` M=23 (8.8e13) are all reported out of reach rather than
    attempted. Per-cell node rates: 2.8–3.4e5/s at |X| ≤ 256, 4.7e4/s at
    |X| = 625 (the state is |X|-bit masks). `K_5(4,2;4)` needs the
    `AGL(1,5)⁴ ⋊ S₄` fallback — its full monomial group is 4.98e9
    elements, a 12.4 TB table.
  - **CLI**: `--symmetry SPEC` (also `-p group=SPEC`) switches `solve` to
    the orbit-restricted driver and prints the orbit count and
    search-space reduction; `--symmetry-polish` continues with the
    word-level descent afterwards. Both opt-in; the default solve path is
    untouched.
- **μ-fold multiple covering** across every problem family (`-p mu=N`,
  `GpuBackend(..., mu=N)`, K_q^μ(n,R)): coverage requirement `cnt ≥ μ`
  implemented as a predicate change at the ABI seam (optional symbols
  `dct_set_mu`/`dct_get_mu`/`dct_ball_gather_mu`/`dct_count_deficit`,
  feature-detected; exact deficit objective and marginals). At μ=1
  everything collapses to the legacy kernels byte-for-byte. See
  `docs/multi-coverage.md`.
- **`grid_linf` plugin** (problem id 3): dominating sets of bounded
  (open-boundary) grids under L∞ balls — the first non-shift-invariant
  family; clamped axis windows, boundary-aware support-table walk.
  Known optima proven in-test (γ(8×8 king graph) = 9, γ(5×9) = 6). See
  `docs/plugin-grid.md`.
- **Wall-breaking passes** (GPU-measured on live record cells): `deficit`
  ruin (aims at the actual uncovered words), a local-shell polish
  neighbourhood (`anneal(neighbourhood="shell"|"auto")` — swaps around
  the just-opened hole; fixes the measured 0%-accept collapse of the
  pool neighbourhood at walls), transform-cost-adaptive polish cadence
  (`pool_refresh="auto"`), an exact budget-capped `two_swap` endgame,
  and the packaged `break_wall` driver (CLI `--wall M`) — 15/15 target
  hits on the benchmark wall cells vs 10/15 for default `descend`.
  Honest negatives documented in `docs/search-passes.md` (strict-nearest
  deficit ruin *loses* standalone; two-swap marginal at small scale).
- **Search passes**: adaptive (ALNS roulette) ruin-operator selection,
  a swap-neighbourhood polish pass at fixed M (LAHC default, Metropolis
  alternative) for escaping near-feasible walls, and a time-sliced
  portfolio driver with a shared incumbent. All opt-in; the
  record-producing defaults are byte-compatible and pinned by tests.
  CLI: `--ruin-select`, `--polish`, `--polish-share`, `--portfolio`.
  See `docs/search-passes.md`.
- **Installable package.** `pip install -e .` now works from a clean
  environment: modern `pyproject.toml` metadata (authors, MIT license,
  project URLs, keywords, trove classifiers, `requires-python >= 3.10`),
  a `coldcore = coldcore.cli:main` console script, and `[test]` /
  `[lint]` extras. The distribution is pure Python (`py3-none-any`), so
  it installs the same on x86-64 and aarch64 with no compiler; the CUDA
  plugins stay a cmake step.
- `examples/` — four runnable, heavily commented scripts that each
  assert their own expected result: `01_covering_code.py` (K_3(4,1) = 9
  and K_3(4,2) = 3, independently verified and printed),
  `02_domination.py` (9×9 king torus → 9 kings), `04_symmetry.py` (orbit
  search vs free search vs an exhaustive proof on K_2(6,1) = 12, the
  μ-fold construction, and the case where orbit search loses), and
  `03_custom_problem.py`, which implements *toroidal queen domination* —
  a problem whose coverage neighborhood is not a metric ball — directly
  against `coldcore.protocol.Backend` and solves it to proven optimality
  with an unchanged search core and no CUDA.
- `tests/test_examples.py` — smoke-runs every example as a subprocess on
  the reference backend, and fails if `examples/` grows a script the
  list does not know about.
- `docs/getting-started.md` (install → CLI → examples → Python API → GPU
  build → tests, including the environment variables) and `docs/api.md`
  (hand-written tour of `ProductSpace`, the `Backend` protocol and its
  semantics, the reference and GPU backends, every `Searcher` entry
  point, `io`, `patterns`, and the CLI grammar).
- README: `Install` and `Examples` sections.

### Fixed
- `Searcher._descend` could return with an infeasible working solution:
  the end-of-budget break fired *after* the notch ruin, leaving the
  ruined state in place. The budget check now precedes the ruin, so the
  solver always ends on the last verified solution. (`on_solved`
  consumers were never affected.)
- `tests/test_grid.py` helper: cast `np.uint8` digits to int before
  `max(0, a-R)` (underflow wrapped to 255 under NumPy 2 / NEP 50).
- `default_plugin_path()` now probes installed locations after the
  in-tree cmake output instead of hard-coding the checkout layout;
  `COLDCORE_PLUGIN_PATH` still wins.
- Package docstring no longer references the retired RENAME.md;
  `__version__` bumped to 0.1.0 (single-sourced into the wheel).

### Changed
- CI (`.github/workflows/ci.yml`): the CPU job now installs the package
  (`pip install -e ".[test]"`) and runs a matrix over Python 3.10, 3.11
  and 3.12, checks that the package imports and both CLI entry points
  work from *outside* the checkout, and runs the examples via the test
  suite. Added a pinned `ruff` lint job and a commented self-hosted-runner
  stub for the GPU parity suite. Concurrency group added so superseded
  runs are cancelled.
- Version is now single-sourced from `coldcore.__version__` via
  setuptools' dynamic metadata, instead of being duplicated in
  `pyproject.toml`.
- The declared numpy floor is `>= 1.21` (was `>= 1.24`); the suite is
  verified on 1.21.5 and on 2.2.

*2026-08-24 — initial content of the 0.1.0 cycle.*

### Added
- Three-layer architecture (problem plugin / state core / search core)
  with two narrow seams: the plugin C ABI (GPU side) and the Python
  backend protocol (host side). See `ARCHITECTURE.md`.
- **Plugin C ABI v1** (agreed 2026-08-22, spec in `docs/PLUGIN.md`,
  header `src/include/coldcore/plugin.h`): flat `dct_*` symbols, singleton
  context, typed `dct_init3(problem, axes[], R, …)`, return-code
  convention `-100-e` for CUDA errors, caller-allocated outputs,
  optional capabilities feature-tested by symbol presence.
- Two problem plugins speaking the ABI:
  - `hamming` (id 0): covering codes K_q(n,R), mixed-radix alphabets —
    the record-holding founding problem, vendored unmodified from the
    upstream record engine (`src/plugins/covering/VENDORED.md`);
  - `torus_linf` (id 1): dominating sets on torus grids under Chebyshev
    balls; re-derives the known 9-vertex optimum for the 9×9 king torus
    through the unchanged search core.
- Python package `coldcore`: ctypes plugin loading
  (`GpuBackend(axes, R, problem=…)` via `dct_init3` with legacy
  fallback), mixed-radix product spaces, support-table builders, backend
  protocol, and the search core (exact lazy greedy, ruin-and-recreate
  LNS, peel, notch descent) ported from the engine that produced the
  covering-code records (arXiv:2608.19872).
- Automatic HBM vs. pinned-LPDDR array placement for cells beyond device
  memory (GH200-class coherent memory; `COLDCORE_HBM_BUDGET` override);
  managed-memory degradation on non-coherent devices.
- Pure-numpy reference backends for both problems; CPU-only pytest suite
  plus GPU parity tests (skipped without a CUDA device).
- CLI: `python -m coldcore solve` (greedy build → peel → notch descent,
  atomic solution checkpoints) and `python -m coldcore info`.
- CMake build (CUDA, default arch sm_90, configurable) for `libcoldcore`
  and plugin shared libraries; GitHub Actions CI (compile-only CUDA
  build + CPU reference tests).
- Documentation: `docs/PLUGIN.md` (ABI spec), `docs/plugin-authoring.md`,
  `docs/benchmarks.md` (measured GH200 numbers, including the first
  exact whole-space transform over a 10¹⁰-word cell and the
  LPDDR-pinning reproduction caveat), `docs/reproducing-records.md`.

### Changed
- The project name **coldcore** was confirmed final on 2026-08-24
  (previously marked as a placeholder; the rename checklist file has
  been removed).
- `map_hist` demoted to an optional ABI capability; the greedy falls
  back to threshold escalation when a backend lacks it.

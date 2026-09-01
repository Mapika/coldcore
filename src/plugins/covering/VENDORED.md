# Provenance — transform engine (hamming + torus_linf + lee plugins)

**In-tree since:** 2026-08-24. coldcore grew out of a single-purpose
prototype record engine (the one behind the covering-code records in the
companion paper). During the transition, CMake compiled the prototype's
CUDA source unmodified; that arrangement ended when its stated condition
— plugin ABI v1 frozen (docs/PLUGIN.md), verification gates PASS — was
met and a third problem family needed wiring into the plugin TU's
`dct_init3` / `dct_transform` switches.

**Base sync:** prototype engine state of 2026-08-22 ~14:15 UTC
(post-refactor, post MEM_ATS mbind fix), copied verbatim:
- `dctcore_core.cuh`     — problem-blind state core (planes, walks,
  reductions, memory modes; MEM_ATS is mbind(MPOL_BIND, node0)-pinned)
- `dctcore_hamming.cuh`  — covering-code axis-pass driver (problem 0)
- `dctcore_torus.cuh`    — torus L-inf domination driver (problem 1)
- `dct_hc.cu`            — the v1 C ABI translation unit

**In-tree additions on top of that base:**
- `dctcore_lee.cuh`      — NEW: Lee-metric covering codes (problem 2,
  `PROB_LEE`), grade-truncated circulant axis DP
- `dctcore_mu.cuh`       — NEW: mu-fold multiple covering (`K^mu`), an
  optional ABI extension (`dct_set_mu` & co, docs/multi-coverage.md).
  Deliberately ADDITIVE: predicate twins of the three `cnt`-reading core
  kernels rather than edits to them, so `dctcore_core.cuh` stays
  byte-identical to the prototype base and `mu == 1` runs the legacy
  kernels.
- `dctcore_grid.cuh`     — NEW: open-boundary L-infinity grid domination
  (problem 3, `PROB_GRID_LINF`, docs/plugin-grid.md). Clamped window-sum
  axis pass plus its OWN support-walk kernels: the core's walk is
  circulant, and an open board needs boundary rejection, so the walk is
  duplicated here rather than made conditional in the core.
- `dct_hc.cu`            — the ONLY divergent copy: `PROB_LEE` arm in
  `dct_init3` validation, a `lee_axes` arm in `dct_transform`, the
  `PROB_GRID_LINF` arms in `dct_init3` / `dct_transform` /
  `ball_update_inner` / `dct_ball_gather(_mu)` / `dct_use_fmt`, the
  `dctcore_lee.cuh` / `dctcore_mu.cuh` / `dctcore_grid.cuh` includes,
  the `dct_set_mu` /
  `dct_get_mu` / `dct_ball_gather_mu` / `dct_count_deficit` entry points
  and the `g_mu == 1` fast-path branches in `dct_transform`,
  `dct_loss_owner` and `dct_free` (plus comment lines)

**History note:** the base files are kept as close to the prototype as
possible so fixes can be adopted verbatim in either direction; `git log
-p` on this directory shows the in-tree hunks on top of the base.

**Contract:** docs/PLUGIN.md (v1) is the authoritative kernel-side
contract; `include/coldcore/plugin.h` restates it (with `CCP_PROB_LEE`
for the in-tree family). Host-side bindings: `coldcore/gpu.py` +
`coldcore/patterns.py`.

**Benchmark caveat:** LPDDR/ATS numbers measured with a
pre-2026-08-22-14:15 build (no mbind pinning) are INVALID — GH200 HMM
silently migrates GPU-hot pageable pages into HBM.

**Build line equivalence:** the prototype builds with
`nvcc -O3 -arch=sm_90 --shared -Xcompiler -fPIC -o libdct_hc.so dct_hc.cu`;
the CMake target reproduces this (Release = `-O3`, `sm_90` default arch,
shared + PIC).

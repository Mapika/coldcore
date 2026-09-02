# Architecture

coldcore is organized like a compiler. A **problem plugin** (the
frontend) lowers a combinatorial problem — a mixed-radix product space
plus a coverage neighborhood — into a small typed C ABI (the IR). The
**GPU engine** (the backend) executes that IR: separable axis-DP
transforms over dense whole-space fields, with automatic HBM vs.
pinned-LPDDR placement. The **search core** (the passes) runs on top,
problem-agnostic, and works unchanged on any plugin.

| compiler | coldcore layer | lives in |
|---|---|---|
| frontend | problem plugin | `src/plugins/*` (CUDA `.so`) + `src/coldcore/patterns.py` |
| IR | plugin ABI v1 | `docs/PLUGIN.md`, `src/include/coldcore/plugin.h` |
| backend | state core / GPU engine | plugin `.so` internals + `src/coldcore/gpu.py` |
| passes | search core | `src/coldcore/search.py` |

The layers meet at two narrow interfaces: the **C ABI** between the
engine and problem plugins (GPU side), and a **Python protocol** between
the search core and backends (host side). Everything above the C ABI is
problem-agnostic.

```
┌─────────────────────────────────────────────────────────────┐
│ search core (Python)          src/coldcore/search.py        │
│   exact lazy greedy · ruin-and-recreate LNS · peel ·        │
│   notch ladder descent · restarts · record callbacks        │
├───────────────── Python backend protocol ───────────────────┤
│                 src/coldcore/protocol.py                    │
│  ┌──────────────────────────┐  ┌──────────────────────────┐ │
│  │ GPU backend (ctypes)     │  │ reference backend (numpy)│ │
│  │ src/coldcore/gpu.py      │  │ src/coldcore/reference/* │ │
│  ├──────── C ABI v1 ────────┤  │  same protocol, brute    │ │
│  │ state core: dense fields │  │  force, tiny instances,  │ │
│  │ HBM/LPDDR placement,     │  │  used for correctness    │ │
│  │ ball walks, reductions   │  │  tests without a GPU     │ │
│  ├──────── axis pass ───────┤  └──────────────────────────┘ │
│  │ plugin .so (CUDA)        │                               │
│  │ hamming: distance-DP     │                               │
│  │ torus_linf: window sum   │                               │
│  └──────────────────────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

## The computational model

A problem instance lives on a **product space** `X = Z_ax1 × … × Z_axn`
(heterogeneous axis sizes allowed, n ≤ 16, |X| ≤ 2^40; points are
integer-indexed by mixed-radix digits). The solution is a **set of
points** (codewords / centers / facilities). The engine maintains dense
per-point fields over the whole space:

- `cnt[w]` — exact multiplicity (how many solution points cover word
  `w`), rebuilt by a full transform from the solution set or updated
  incrementally by ball walks;
- `A0[w]` — a work field holding the most recent **gain map** (exact
  marginal gain of adding point `w`) or **loss map** (exact marginal
  loss of removing an incumbent).

The defining trick: gain/loss over the *entire* space is computed by a
separable transform — n axis sweeps of local DP, `O(n·q^n)` total —
instead of per-candidate ball enumeration. The search layer then only
ever needs cheap queries on those fields: max, histogram, threshold
extraction, point reads, and exact per-candidate re-checks
(`ball_gather`).

This model fits any problem where "u covers x" is a shift-invariant
per-axis product predicate: covering codes, dominating sets on Hamming
graphs and torus grids, quantizer/facility placement on grids, packing
and multi-cover variants.

## Frontend: the problem plugin

A plugin owns the mathematics — what the fields mean, how the transform
computes them, what "ball" means — and supplies exactly two things:

1. an **axis-pass driver** (CUDA, one `.cuh`): n separable per-axis
   passes turning `A_0 = indicator(S)` into the whole-space coverage
   field;
2. a **support-table builder** (host-side Python,
   `src/coldcore/patterns.py`): the packed list of coordinate-shift patterns
   enumerating one move's coverage set.

Everything else — memory placement, ball walks, owner-trick loss pass,
reductions, threshold extraction — is inherited from the problem-blind
state core. Authoring guide: `docs/plugin-authoring.md`. Existing
plugins: `hamming` (covering codes, mixed radix — the record-holding
founding problem) and `torus_linf` (torus domination, Chebyshev balls —
added upstream in a day). A Lee-metric plugin is in progress.

## IR: the plugin ABI v1

Agreed and committed 2026-08-22. Authoritative spec: `docs/PLUGIN.md`;
restated for plugin authors in `src/include/coldcore/plugin.h`. The
essentials:

- flat `dct_*` symbols, **singleton context** (one cell per process;
  `void *ctx` reserved for v2 — do not emulate handles);
- typed init: `dct_init3(problem, n, axes[], R, ballpat, …,
  layers_mode, cnt_mode, use_owner)` — no string blobs;
- int return codes: 0 ok, negative engine error, `-100-e` = CUDA error
  e; caller-allocated outputs everywhere.

Required operation families:

| family | operations | contract |
|---|---|---|
| lifecycle | `dct_init3` / `dct_free` | per-array-group memory-mode arguments |
| solution | `dct_set_code(idx[])`, `dct_transform(0,2)` | rebuild `cnt` exactly from the set |
| fields | `dct_transform(1,1)` gain, `dct_transform(2,1)` loss | write exact whole-space map into `A0` |
| incremental | `dct_ball_update(idx[], ±1)` | apply adds/removes to `cnt` |
| exact query | `dct_ball_gather(idx[], target)` | exact per-candidate count, never stale |
| map queries | `dct_map_max`, `dct_map_extract(64)`, `dct_map_read_at` | host-visible views of `A0` |
| scalars | `dct_count_eq(target)` | e.g. `count_eq(0)` = uncovered points |

Optional capabilities, feature-tested by symbol presence: `dct_map_hist`
(the greedy falls back to threshold escalation without it),
`dct_loss_owner` (owner-trick loss pass — measured slower in dense
regimes, see `docs/benchmarks.md`), `dct_set_tbl2`/`dct_use_fmt`
(alternative support-table format), `dct_host_bytes`, 64-bit extraction
for `|X| > 2^32`.

## Backend: the state core

Owns *where bytes live* and *how plugins are loaded*, so plugins stay
pure math:

- **Memory placement.** On GH200 (480 GB LPDDR + 96 GB HBM,
  ATS-coherent), dense fields larger than the HBM budget are placed in
  LPDDR — pinned with `mbind(MPOL_BIND, node0)` so GH200's HMM page
  migration cannot silently move hot pages into HBM — and streamed over
  C2C by the transforms, while the random-access `cnt` field stays in
  HBM. Placement is automatic from array sizes and an HBM budget (env
  `COLDCORE_HBM_BUDGET`), overridable per array group. On non-coherent
  devices the LPDDR modes degrade to managed memory. Measured placement
  numbers, including the first exact transform over a 10¹⁰-word cell:
  `docs/benchmarks.md`.
- **Plugin loading.** `coldcore.gpu.GpuBackend(axes, R, problem=…)`
  binds the ABI via ctypes (`dct_init3` with legacy fallback, rc
  decoding, optional-symbol feature tests) and wraps it in the Python
  backend protocol.
- **Checkpointing.** Solution snapshots are host-side numpy arrays;
  atomic-write helpers persist every improvement, so a kill at any
  moment loses nothing.

## Passes: the search core

Problem-agnostic port of the algorithms that set the covering-code
records (arXiv:2608.19872):

- **Exact lazy greedy** — extract a threshold slice of the gain map,
  keep a stale-bound max-heap, re-verify candidates exactly with
  `ball_gather`; a placement is accepted only when its exact gain beats
  the next stale bound. Refreshes the map when bounds decay. Exactness
  of the field makes this true greedy, not approximate greedy.
- **Ruin** — remove k incumbents by one of: exact-loss-biased ("low",
  exponential jitter), cluster (digit-distance ball around a random
  incumbent), hole (cluster around a high-gain point of the current gain
  map), random.
- **Recreate** — greedy fill back to the target size M.
- **LNS at fixed M** — ruin/recreate rounds with best-snapshot
  bookkeeping, ~10% random accept of regressions, periodic full-recount
  drift checks.
- **Peel** — remove zero-loss incumbents in pairwise-independent batches
  (separation > 2R so removals stay independent), to fixpoint.
- **Notch descent** — from a feasible solution at M, attempt M−step,
  halving the step on failure, doubling on fast success.

Callbacks: `on_log(str)`, `on_improve(state)`, `on_solved(M, idx)` — the
latter is where a record gate (independent verifier) hooks in. The
search core never declares records; the gate does.

## Passes over orbits: the symmetry layer

`src/coldcore/symmetry.py` adds a second set of atoms. A **group** is stored
by generators — *(coordinate permutation, per-coordinate value
permutations)* pairs, the general element of the wreath product that is
the automorphism group of a mixed-radix product space — and its whole-
space **orbit partition** is computed as a CSR index structure. The same
passes then run over orbits: a solution is a union of chosen orbits, and
add/drop is a whole orbit.

The seam is the same one everything else uses. An orbit's exact gain is
`ball_update(O, +1)` followed by a deficit readback; the per-orbit sums
of the existing gain/loss maps (one transform, one segment sum over the
CSR) supply the bounds. No new backend operation, no new CUDA — the
symmetry layer is a *client* of the protocol, so it works unchanged on
every plugin and on the reference backends.

`src/coldcore/symmetry_exact.py` sits beside it and answers the complementary
question — *is there nothing smaller?* — by isomorph-free exhaustive
generation on tiny cells. See `docs/symmetry.md`.

## Testing model

Every plugin ships a **pure-numpy reference backend** implementing the
same Python protocol by brute force on tiny instances. This yields:

1. CPU-only correctness tests of the search core (runs in CI, no GPU);
2. GPU parity tests: identical field values reference-vs-plugin on tiny
   instances (`pytest -m gpu`, skipped when no CUDA device).

## Vendoring policy (transition period)

The covering CUDA engine remains upstream in the record-hunting repo
while it is refactored along these seams. coldcore builds it
**unmodified** from the upstream source path; no fork-and-diverge.
Details and sync date: `src/plugins/covering/VENDORED.md`.

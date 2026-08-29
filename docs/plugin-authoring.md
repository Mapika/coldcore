# Writing a coldcore plugin

Contract: **v1, agreed 2026-08-22** — flat `dct_*` symbols, singleton
context. Authoritative source: [PLUGIN.md](PLUGIN.md); restated in
`src/include/coldcore/plugin.h`. On divergence, PLUGIN.md wins.

## What a plugin is

The engine is already generic over a finite product space
`X = prod_j Z_{ax_j}` (heterogeneous axis sizes, n <= 16, |X| <= 2^40).
A plugin adds one *coverage geometry* to it — any problem where
"u covers x" is a shift-invariant per-axis product predicate qualifies
(covering codes, torus domination, packing/multicover variants).

A plugin supplies exactly two things:

1. **An axis-pass driver** (CUDA, one `.cuh`):
   `int <name>_axes(int out_mode, int blocks, int threads)` — turns
   `A_0 = indicator(S)` into the whole-space coverage field via n
   separable per-axis passes.
   - `out_mode 1`: coverage sum -> the map plane `A0`
   - `out_mode 2`: clamped sum -> the multiplicity plane `cnt`
   Registration in the engine: a `PROB_*` id, a validation arm in
   `dct_init3` (axis-size/R bounds, number of layer planes), a dispatch
   arm in `dct_transform`.

2. **A support-table builder** (host-side Python — see
   `src/coldcore/patterns.py::support_patterns[_ox]`): the packed list of
   coordinate-shift patterns enumerating one move's coverage set.
   - v1 format: uint64 per pattern, byte k = `(pos<<4 | delta)`, delta
     a circular shift in `1..ax_pos-1`, zero byte terminates.
   - ox format (optional): uint16 position mask + uint64 delta nibbles.
   Both need `ax_j <= 16` and pattern weight (changed positions) <= 8:
   Hamming balls have weight <= R; torus Chebyshev balls weight <= n.

Everything else — memory placement (HBM / ATS-LPDDR / managed per array
group), incremental ball walks, owner-trick losspass, reductions,
threshold extraction — is the problem-blind state core
(`dctcore_core.cuh`).

## ABI rules (v1)

- Singleton: one live cell per process; `dct_free` before re-init.
  `void *ctx` first-arg is reserved for v2 — do not emulate handles.
- int return codes: 0 ok, negative engine error, `-100-e` = CUDA error
  e (message on stderr). No exceptions across the ABI.
- Caller-allocated outputs everywhere.
- Typed init args (`dct_init3`), no string blobs.
- REQUIRED symbols: `dct_init3/free, set_code, transform, ball_update,
  ball_gather, count_eq, map_max, map_extract(+64), map_read_at,
  read_cnt`. OPTIONAL, feature-tested by presence: `loss_owner,
  set_tbl2/use_fmt, map_hist, host_bytes`.

## Existing problems

| id | name | axes | layer planes | axis pass |
|---|---|---|---|---|
| 0 | `hamming` | 2..10, mixed ok | R+1 (R<=5) | distance-count DP per fiber |
| 1 | `torus_linf` | 3..16, 2R+1<=ax_j | 1 | circular window sum |

## The Python side (this repo)

1. Add the problem name/id to `src/coldcore/patterns.py` (`PROBLEM_IDS`,
   `_delta_sets`) if it introduces a new delta geometry.
2. `coldcore.gpu.GpuBackend(axes, R, problem=...)` already speaks the
   v1 ABI; extend it only if your plugin adds optional symbols.
3. **Ship a reference backend** (non-negotiable): subclass
   `src/coldcore/reference/base.py::BruteForceBackend`, implement
   `_covers(a, b)` — usually < 15 lines. See `reference/torus.py`.
4. **Tests**: field identities + search on the reference (CPU, runs in
   CI), and parity tests GPU-vs-reference on tiny instances marked
   `@pytest.mark.gpu` (see `tests/test_gpu_parity_torus.py`).
   Remember the singleton rule: one live GPU cell at a time.

## Search-core semantics you must honor

See `src/coldcore/protocol.py`. In particular: `ball_gather` is never
stale; after any sequence of `ball_update`s, a full `recount` must
reproduce `cnt` exactly (the search core checks for drift and will
call your bug out in its logs); `map_extract(thr)` returns values
`>= thr`, unordered, with the true total as `found`.

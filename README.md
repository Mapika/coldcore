# coldcore

[![ci](https://github.com/Mapika/coldcore/actions/workflows/ci.yml/badge.svg)](https://github.com/Mapika/coldcore/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2608.19872-b31b1b.svg)](https://arxiv.org/abs/2608.19872)

**coldcore** is a GPU framework for exhaustive-coverage combinatorial
optimization — covering codes, domination problems, and other
set-cover-like problems on finite product spaces — structured the way a
compiler is structured.

The founding observation: for many hard covering-type problems over a
product space `q^n`, the *exact* marginal gain of every candidate move
("how many uncovered points would this newly cover?") and the *exact*
marginal loss of every incumbent ("how many points does this privately
cover?") can be computed for **the whole space at once** by a separable
axis-by-axis dynamic-programming transform — a few dense sweeps costing
`O(n·q^n)`, no sampling, no surrogate. Exact whole-space gain/loss
fields turn simple search (lazy greedy plus ruin-and-recreate LNS) into
a record machine.

## An LLVM for exhaustive-coverage problems

coldcore separates the *problem*, the *machine*, and the *search* the
way LLVM separates frontends, targets, and passes:

| compiler | coldcore | concretely |
|---|---|---|
| frontend | **problem plugin** | defines a mixed-radix product space and a coverage neighborhood; one CUDA axis-pass driver + a host-side support-table builder |
| IR | **plugin ABI v1** | flat `dct_*` C symbols; `dct_init3(problem, axes[], R, …)`; gain/loss/count *full-space* transforms; incremental ball updates; exact per-candidate gathers |
| backend | **GPU engine** | separable axis-DP transforms; automatic HBM vs. pinned-LPDDR array placement (Grace-Hopper coherent memory for cells beyond HBM) |
| passes | **search core** | exact lazy greedy, ruin-and-recreate LNS (fixed or adaptive operator selection), swap-polish (LAHC/annealing), portfolio restarts, peel, notch descent — pure Python + numpy, problem-agnostic, runs unchanged on any plugin ([docs/search-passes.md](docs/search-passes.md)) |

The evidence that the seams are real: the same search core that produced
**26 new best-known upper bounds** on the covering-code tables K_q(n,R)
([arXiv:2608.19872](https://arxiv.org/abs/2608.19872),
[github.com/Mapika/coldcase](https://github.com/Mapika/coldcase)) also
finds the known optimum for 9×9 king-torus domination through the torus
plugin — with zero changes to the search core.

## Install

```bash
git clone https://github.com/Mapika/coldcore && cd coldcore
pip install -e .
```

Python 3.10+ and numpy, and that is the entire runtime dependency list
— the GPU bindings are ctypes, so there is no pybind11/nanobind/torch
in the tree. The distribution is pure Python (`py3-none-any`) and
installs identically on x86-64 and aarch64 (Grace/GH200) with no
compiler involved.

pip gives you the **search core, the CLI, and the numpy reference
backends** — enough to solve, verify, and prototype a new problem
end-to-end on a CPU. The **CUDA plugins are built by cmake, not by
pip** (`cmake -B build && cmake --build build -j`); see
[docs/getting-started.md](docs/getting-started.md#5-gpu-backends).

New here: [docs/getting-started.md](docs/getting-started.md) is the
twenty-minute walkthrough, [docs/api.md](docs/api.md) the reference for
the public Python surface.

## Quickstart

```bash
# build the core library + plugins (CUDA toolkit >= 12 required to build; GPU not required)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CPU-only sanity: search core against the numpy reference backend (no GPU needed)
python -m pytest tests/ -m "not gpu"

# smoke test: solve K_2(4,1) with the pure-numpy reference backend (optimum is 4)
python -m coldcore solve --backend ref -p q=2 -p n=4 -p R=1

# a cell the reference cannot touch, still with no GPU: the axis-DP
# CPU backend runs the plugin's own transforms in numpy
python -m coldcore solve --backend cpu -p q=5 -p n=8 -p R=3 --hours 0.1

# the same cell on the GPU plugin
python -m coldcore solve --backend gpu -p q=2 -p n=4 -p R=1

# a whole sweep: one worker process per cell, resumable, checkpointed
python -m coldcore campaign --backend cpu --workers 8 --hours 1 \
    --cells 'K3(8,2),K5(8,3),lee:9x9x9:2,K2(6,1)mu3' --outdir runs/sweep
```

`python -m coldcore solve` runs: exact greedy build → peel → notch
descent, writing every improvement through an atomic checkpoint callback
(`--out`). `campaign` is the same thing across a cell list, N processes
at a time, with per-cell logs and JSON records collected in `--outdir`.

Backends: `ref` (brute-force numpy oracle, `|X| ≤ 2^16`), `cpu`
(`coldcore.dp` — the plugin's axis DP in numpy, measured 20-831× the
reference and good to `|X| ~ 10^7`), `gpu` (the CUDA plugin). All three
are bit-identical on the fields the search reads; see
[docs/performance.md](docs/performance.md).

## Examples

Three self-contained scripts under [`examples/`](examples/), each one a
plain `.py` you run and read — no notebooks, no GPU, seconds each. Each
ends by asserting its own expected result against a known optimum, so
they are also tests (`tests/test_examples.py` runs them in CI).

```bash
python examples/01_covering_code.py    # K_3(4,1) = 9 and K_3(4,2) = 3, verified
python examples/02_domination.py       # 9x9 king torus: 9 kings, printed as a board
python examples/03_custom_problem.py   # a problem coldcore has never seen
```

The third is the one to read if you came here to extend coldcore. It
teaches the framework **toroidal queen domination** — a problem whose
coverage neighborhood is a union of four wrapped lines, not a metric
ball — as a single ~90-line class implementing
`coldcore.protocol.Backend`, and the stock search core solves it to
proven optimality (4 queens on a 7×7 torus, checked against exhaustive
enumeration) with no change anywhere else in the tree. That is the
plugin seam, demonstrated without writing a line of CUDA.

## Problem plugins

| id | plugin | problem |
|---|---|---|
| 0 | `hamming` | covering codes K_q(n,R), mixed-radix alphabets — the record-holding founding problem |
| 1 | `torus_linf` | dominating sets on torus grids under L∞ (Chebyshev) balls — king-graph domination and its n-D generalizations |
| 2 | `lee` | covering codes under the Lee metric (circular per-digit distance, summed), mixed-radix — includes the perfect Golomb–Welch codes as test optima |
| 3 | `grid_linf` | dominating sets of bounded (non-wrapping) grids under L∞ balls — king domination on a chessboard; the first non-shift-invariant family |

All families also support **μ-fold multiple covering** (`-p mu=N`: every
point must be covered at least μ times, the K_q^μ(n,R) tables) — μ is a
predicate change at the ABI seam, not a plugin, so it composes with
every problem; see [docs/multi-coverage.md](docs/multi-coverage.md).

Adding a problem means writing one CUDA axis-pass driver and a small
host-side table builder; the memory placement, ball walks, reductions,
and the entire search stack come for free. Each of `torus_linf`, `lee`,
and `grid_linf` was added in about a day — that is the point of the
seam.

```bash
# 9-vertex domination of the 9x9 king torus, pure numpy backend
python -m coldcore solve --backend ref -p problem=torus_linf -p axes=9x9 -p R=1
```

## The plugin ABI

Plugin C ABI **v1**: flat `dct_*` symbols, singleton context, typed init
arguments, int return codes (`-100-e` = CUDA error e), caller-allocated
outputs. The authoritative specification is
[docs/PLUGIN.md](docs/PLUGIN.md); the header is
[`src/include/coldcore/plugin.h`](src/include/coldcore/plugin.h); the authoring
walkthrough is [docs/plugin-authoring.md](docs/plugin-authoring.md).
The Python protocol (`src/coldcore/protocol.py`) is the stable host-side
surface.

## Symmetry

Every space here has a huge automorphism group — the wreath product
`(∏_j S_{a_j}) ⋊ S_n` for Hamming, per-axis rotations and reflections for
Lee and the torus. The symmetry layer makes that a first-class concept:

```python
from coldcore.symmetry import SymmetricSearcher, parse_group, orbit_solve

G = parse_group(b.space, "translate:1011")   # cosets of a linear code
code, s = orbit_solve(b, G)                  # solution = a union of G-orbits
```

`SymmetricSearcher` runs the same greedy / peel / notch-descent passes
over **orbits** instead of words — the 1990s group-invariant-code
construction, mechanised — and stays exact: an orbit's gain is measured
by applying it and reading the exact deficit back, and the gain map's
segment sum over an orbit is a monotone upper bound on it, so the
lazy-greedy heap is exact greedy with one whole-space transform amortised
over the whole build. Correctness never depends on the group; a bad group
only restricts the search to a bad subfamily.

Where the optimum is a union of cosets this is a different regime, not a
better constant. Time to a *verified* solution at the known optimum, mean
over the seeds that got there out of 5, 30 s cap
(`scripts/bench_symmetry.py`):

| cell | free search | orbit search | speedup |
|---|---|---|---|
| K_2²(6,1) = 20 | 14.2 s, 3/5 seeds | 1.5 ms, 5/5 | 9438x |
| K²(Lee Z_5²) = 10 | 3.8 s, 5/5 | 0.4 ms, 5/5 | 9613x |
| K_4²(4,1) = 48 | 18.6 s, 1/5 | 6.5 ms, 5/5 | 2856x |
| K_2³(8,1) = 96 | never, 0/5 | 17.1 ms, 5/5 | >1754x |
| K_2(6,1) = 12 | 3.8 s, 1/5 | 3.5 ms, 5/5 | 1076x |
| K_4(4,1) = 24 | 5.7 s, 5/5 | **never** | the optimum is not group-invariant |
| king 11x11 = 16 | 10 ms, 5/5 | **never** | 16 is not a multiple of 11 |

`src/coldcore/symmetry_exact.py` closes the other half for tiny cells:
isomorph-free exhaustive search that returns `feasible` + witness or an
`infeasible` **proof** (coverage branching, stabiliser-reduced branch
sets, counting bounds). It settles K_2(6,1) = 12 in 7 s — the same cell
orbit search finds the 12 for. It also says, with measured node rates and
a calibrated tree-size estimate, that K_8(4,2) is ~1e34 years out of
reach and what would have to change.

Concepts, API, the full win/loss table and the K_8(4,2) analysis:
[docs/symmetry.md](docs/symmetry.md). CLI: `--symmetry SPEC`.

## Benchmarks

Measured numbers — kernel throughput, memory-placement study including
the first exact transform over a 10¹⁰-word cell (260 GB of dense arrays,
73–110 s), the CPU backend against brute force, and honest negative
results — live in [docs/benchmarks.md](docs/benchmarks.md).
[docs/performance.md](docs/performance.md) collects the profiles, the
before/after tables, the ideas that did **not** pay, and the GPU-side
roadmap. Reproducing the covering-code records:
[docs/reproducing-records.md](docs/reproducing-records.md).

## Hardware requirements

Any CUDA GPU builds and runs the framework (default arch sm_90,
configurable via `COLDCORE_CUDA_ARCHS`). Cells whose dense fields fit in
device memory need nothing else. The big-memory path — cells larger than
HBM, up to 2^40 points — requires Grace-Hopper-class hardware with
cache-coherent CPU memory (tested on GH200 480GB); on non-coherent
devices the LPDDR placement modes degrade to managed memory.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for ground rules (exactness,
references, paired benchmarks), development setup, and the plugin
checklist. Bug reports and questions go through
[GitHub issues](https://github.com/Mapika/coldcore/issues).

## License and provenance

MIT — Copyright (c) 2026 Mark Marosi.

coldcore was developed in collaboration with Claude (Anthropic), an AI
system that wrote and benchmarked code autonomously on the author's
hardware under the author's direction; Mark Marosi is the author and
maintainer and reviewed the results. All record claims are checked by
verification tooling independent of the search engine.

## Citation

If you use coldcore in academic work, please cite it via
[CITATION.cff](CITATION.cff), which points at the records paper:

```bibtex
@misc{marosi2026covering,
  title         = {New upper and lower bounds on covering codes {$K_q(n,R)$}
                   for alphabets of size {$5 \le q \le 21$}},
  author        = {Marosi, Mark},
  year          = {2026},
  eprint        = {2608.19872},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IT}
}
```

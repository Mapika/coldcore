# coldcore

[![ci](https://github.com/Mapika/coldcore/actions/workflows/ci.yml/badge.svg)](https://github.com/Mapika/coldcore/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2608.19872-b31b1b.svg)](https://arxiv.org/abs/2608.19872)

coldcore is a GPU framework for covering-type combinatorial optimization
problems on finite product spaces, such as covering codes and dominating
sets.

For many of these problems on a product space of size `q^n`, two
quantities can be computed exactly for every point at once by a
separable dynamic-programming transform applied one axis at a time, at
cost `O(n·q^n)`:

- the gain of each candidate: the number of currently uncovered points
  it would cover;
- the loss of each chosen element: the number of points that only it
  covers.

The search algorithms (lazy greedy construction and large neighbourhood
search) read these exact values instead of sampled estimates.

## Structure

coldcore has four parts:

| part | contents |
|---|---|
| problem plugins | each defines a mixed-radix product space and a coverage neighbourhood; one CUDA driver for the per-axis pass and a host-side table builder |
| plugin ABI v1 | flat `dct_*` C symbols; `dct_init3(problem, axes[], R, …)`; full-space gain, loss and count transforms; incremental updates; per-candidate queries |
| GPU engine | the separable transforms; placement of arrays in GPU memory or in pinned CPU memory (coherent CPU memory on Grace-Hopper, for spaces larger than GPU memory) |
| search core | lazy greedy, ruin-and-recreate large neighbourhood search (fixed or adaptive operator selection), swap improvement (late-acceptance hill climbing or annealing), restarts, and descent passes; pure Python and numpy, independent of the problem ([docs/search-passes.md](docs/search-passes.md)) |
| coset start | for homogeneous Hamming cells: unions of cosets of random subgroups of the alphabet group, searched on the quotient with the same search core and FFT gain fields; the default first stage of `coldcore solve` on those cells ([docs/cosets.md](docs/cosets.md)) |

The search core does not depend on the problem. The same code is used
for covering codes and, through the torus plugin, finds the known
optimum of 9 for domination of the 9×9 king torus. Some of the improved
upper bounds on K_q(n,R) in
[arXiv:2608.19872](https://arxiv.org/abs/2608.19872) were found with
this software; the codes are in
[github.com/Mapika/coldcase](https://github.com/Mapika/coldcase).

## Install

```bash
git clone https://github.com/Mapika/coldcore && cd coldcore
pip install -e .
```

The runtime dependencies are Python 3.10 or later and numpy. The GPU
bindings use ctypes. The distribution is pure Python (`py3-none-any`)
and installs the same way on x86-64 and aarch64 without a compiler.

pip installs the search core, the command-line interface, and the
numpy reference backends. These are sufficient to solve, verify, and
prototype a new problem on a CPU. The CUDA plugins are built with cmake
(`cmake -B build && cmake --build build -j`); see
[docs/getting-started.md](docs/getting-started.md#5-gpu-backends).

[docs/getting-started.md](docs/getting-started.md) is a tutorial and
[docs/api.md](docs/api.md) is the reference for the Python API.

## Quickstart

```bash
# build the core library + plugins (CUDA toolkit >= 12 required to build; GPU not required)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CPU-only sanity: search core against the numpy reference backend (no GPU needed)
python -m pytest tests/ -m "not gpu"

# smoke test: solve K_2(4,1) with the pure-numpy reference backend (optimum is 4)
python -m coldcore solve --backend ref -p q=2 -p n=4 -p R=1

# a larger instance on the CPU backend, which runs the plugin's
# transforms in numpy (no GPU needed)
python -m coldcore solve --backend cpu -p q=5 -p n=8 -p R=3 --hours 0.1

# the K_2(4,1) instance on the GPU plugin
python -m coldcore solve --backend gpu -p q=2 -p n=4 -p R=1

# many instances: one worker process each, resumable, with checkpoints
python -m coldcore campaign --backend cpu --workers 8 --hours 1 \
    --cells 'K3(8,2),K5(8,3),lee:9x9x9:2,K2(6,1)mu3' --outdir runs/sweep
```

`python -m coldcore solve` builds a solution by greedy construction,
removes redundant elements, and then runs a descent that tries to reduce
the solution size by one at a time. Each improvement is written to the
file given by `--out`. `campaign` runs `solve` on a list of instances in
parallel processes and writes per-instance logs and JSON results to
`--outdir`.

There are three backends. `ref` is a brute-force numpy implementation
for spaces of at most 2^16 points. `cpu` (`coldcore.dp`) runs the
plugin's per-axis transforms in numpy; it measured 20 to 831 times
faster than `ref` and is practical up to about 10^7 points. `gpu` is the
CUDA plugin. All three return identical values for the quantities the
search uses; see
[docs/performance.md](docs/performance.md).

## Examples

[`examples/`](examples/) contains three scripts. Each runs in seconds
without a GPU and ends by checking its result against a known optimum.
CI runs them through `tests/test_examples.py`.

```bash
python examples/01_covering_code.py    # K_3(4,1) = 9 and K_3(4,2) = 3, verified
python examples/02_domination.py       # 9x9 king torus: 9 kings, printed as a board
python examples/03_custom_problem.py   # a new problem defined in the script
```

The third example shows how to add a problem. It defines toroidal
queen domination, where the coverage neighbourhood is a union of four
lines rather than a metric ball, as a class of about 90 lines
implementing `coldcore.protocol.Backend`. The unmodified search core
finds the optimum of 4 queens on the 7×7 torus, which the script checks
by exhaustive enumeration.

## Problem plugins

| id | plugin | problem |
|---|---|---|
| 0 | `hamming` | covering codes K_q(n,R), including mixed-radix alphabets |
| 1 | `torus_linf` | dominating sets on torus grids under L∞ (Chebyshev) balls, including king-graph domination and its n-dimensional generalizations |
| 2 | `lee` | covering codes under the Lee metric, including mixed-radix alphabets; the perfect Golomb–Welch codes are used as test cases |
| 3 | `grid_linf` | dominating sets of bounded (non-wrapping) grids under L∞ balls, such as king domination on a chessboard |

All plugins support μ-fold multiple covering (`-p mu=N`), in which
every point must be covered at least μ times (the K_q^μ(n,R) tables).
This is implemented in the ABI and works with every plugin; see [docs/multi-coverage.md](docs/multi-coverage.md).

A second backend family handles problems in which the candidates and
the points to be covered are different sets and coverage is given as an
explicit incidence list (`coldcore.incidence`; on the GPU, plugin ABI 2,
`libcoldcore_sparse.so`). The implemented problem is covering designs
C(v,k,t): every t-subset of a v-set must lie in at least one (or μ) of
the chosen k-subsets. Use `--problem covering_design -p v=.. -p k=.. -p
t=..`. Input and output use the one-block-per-line file format of the La
Jolla Covering Repository. The search core is the same. Design,
limitations and timings are in
[docs/incidence.md](docs/incidence.md).

Adding a problem requires one CUDA driver for the per-axis pass and a
host-side table builder. Memory placement, neighbourhood enumeration,
reductions and the search are shared.

```bash
# 9-vertex domination of the 9x9 king torus, pure numpy backend
python -m coldcore solve --backend ref -p problem=torus_linf -p axes=9x9 -p R=1
```

## The plugin ABI

The plugin C ABI v1 uses flat `dct_*` symbols, a single context, typed
initialization arguments, integer return codes (`-100-e` means CUDA
error e), and caller-allocated outputs. The specification is
[docs/PLUGIN.md](docs/PLUGIN.md), the header is
[`src/include/coldcore/plugin.h`](src/include/coldcore/plugin.h), and
[docs/plugin-authoring.md](docs/plugin-authoring.md) describes how to
write a plugin. On the Python side, `src/coldcore/protocol.py` defines
the stable interface.

## Symmetry

The spaces have large automorphism groups: the wreath product
`(∏_j S_{a_j}) ⋊ (∏_a S_{m_a})` for the Hamming space (where `m_a` is
the number of axes of alphabet size `a`), and per-axis rotations and
reflections for the Lee space and the torus. The symmetry module
restricts the search to solutions invariant under a chosen group:

```python
from coldcore.symmetry import SymmetricSearcher, parse_group, orbit_solve

G = parse_group(b.space, "translate:1011")   # cosets of a linear code
code, s = orbit_solve(b, G)                  # solution = a union of G-orbits
```

`SymmetricSearcher` runs the same greedy, removal and descent passes
on orbits instead of single points, as in the classical construction of
group-invariant codes. The gain of an orbit is computed exactly by
adding it and recounting the uncovered points. The sum of the gain map
over an orbit is an upper bound on that gain and does not increase as
the construction proceeds, so lazy greedy on orbits selects the same
orbits as exact greedy. Every returned solution is verified, whatever
group is chosen. A poorly chosen group only restricts the search to a
family that does not contain good solutions.

The table gives the time to a verified solution of the known optimal
size, averaged over the seeds (out of 5) that reached it, with a 30 s
limit (`scripts/bench_symmetry.py`). Orbit search is much faster when
the optimum is invariant under the group, and fails when it is not.

| instance | unrestricted search | orbit search | speedup |
|---|---|---|---|
| K_2²(6,1) = 20 | 14.2 s, 3/5 seeds | 1.5 ms, 5/5 | 9438x |
| K²(Lee Z_5²) = 10 | 3.8 s, 5/5 | 0.4 ms, 5/5 | 9613x |
| K_4²(4,1) = 48 | 18.6 s, 1/5 | 6.5 ms, 5/5 | 2856x |
| K_2³(8,1) = 96 | not reached, 0/5 | 17.1 ms, 5/5 | >1754x |
| K_2(6,1) = 12 | 3.8 s, 1/5 | 3.5 ms, 5/5 | 1076x |
| K_4(4,1) = 24 | 5.7 s, 5/5 | not reached | the optimum is not group-invariant |
| king 11x11 = 16 | 10 ms, 5/5 | not reached | 16 is not a multiple of 11 |

For small instances, `src/coldcore/symmetry_exact.py` performs an
exhaustive search up to isomorphism, using branching on uncovered
points, branch sets reduced by the stabilizer, and counting bounds. It
returns either a solution or a proof that none of the given size
exists. It shows K_2(6,1) = 12 in 7 s. Based on measured node rates and
an estimate of the tree size, it would need about 10^34 years for
K_8(4,2).

Details, the full benchmark table and the K_8(4,2) estimate are in
[docs/symmetry.md](docs/symmetry.md). The command-line option is
`--symmetry SPEC`.

## Benchmarks

[docs/benchmarks.md](docs/benchmarks.md) reports kernel throughput, a
memory-placement study that includes a transform over a space of 10^10
points (260 GB of arrays, 73 to 110 s), a comparison of the CPU backend
with brute force, and optimizations that did not improve performance.
[docs/performance.md](docs/performance.md) contains profiles,
before-and-after measurements, and planned GPU work.
[docs/reproducing-records.md](docs/reproducing-records.md) describes how
to reproduce the covering-code results.

## Hardware requirements

The framework builds and runs on any CUDA GPU. The default
architecture is sm_90 and can be changed with `COLDCORE_CUDA_ARCHS`.
Instances whose arrays fit in GPU memory have no further requirements.
Instances larger than GPU memory, up to 2^40 points, require
Grace-Hopper hardware with cache-coherent CPU memory; this was tested on
a GH200 with 480 GB. On GPUs without coherent CPU memory, the CPU-memory
placement modes fall back to CUDA managed memory.

The CUDA covering plugin supports at most 65,535 active codeword
occurrences (including duplicates), with exact uint16 counters. Since
version 0.2.2 it rejects oversized loads and invalid incremental updates
instead of allowing saturated or wrapped counts. Rebuild the CUDA plugin when
upgrading. See [the ABI contract](docs/PLUGIN.md#exact-counter-limits-v022).
The explicit-incidence backend uses separate int32 counters.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for requirements on
contributions, development setup, and the checklist for new plugins. Bug reports and questions go through
[GitHub issues](https://github.com/Mapika/coldcore/issues).

## License and provenance

MIT License. Copyright (c) 2026 Mark Marosi.

coldcore was developed in collaboration with Claude (Anthropic), an AI
system that wrote and benchmarked code autonomously on the author's
hardware under the author's direction; Mark Marosi is the author and
maintainer and reviewed the results. All reported bounds are checked by
verification tooling independent of the search engine.

## Citation

If you use coldcore in academic work, please cite it via
[CITATION.cff](CITATION.cff), which points to the covering-codes paper:

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

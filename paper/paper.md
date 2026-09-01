---
title: 'coldcore: a GPU framework for exhaustive-coverage combinatorial optimization'
tags:
  - CUDA
  - GPU computing
  - combinatorial optimization
  - covering codes
  - dominating sets
  - large-neighbourhood search
  - Grace Hopper
authors:
  - name: Mark Marosi
    affiliation: 1
affiliations:
  - name: Department of Artificial Intelligence and Systems Engineering, Budapest University of Technology and Economics (BME), Budapest, Hungary
    index: 1
date: 24 August 2026
bibliography: paper.bib
---

# Summary

`coldcore` is a GPU framework for a family of hard discrete optimization
problems — covering codes, dominating sets, and other set-cover-like
problems on product spaces — whose common trait is that the *exact* value
of every candidate move can be computed for the **entire** search space at
once. For a space $X = \mathbb{Z}_{q_1} \times \dots \times
\mathbb{Z}_{q_n}$ and a radius-like neighbourhood, a separable
axis-by-axis dynamic-programming transform computes, in $O(n\,|X|)$ work
and a few dense sweeps, the exact marginal gain of adding a point at
*every* $x \in X$ (and, symmetrically, the exact private coverage of every
incumbent). Exhaustive-coverage evaluation turns simple search — exact
lazy greedy construction plus ruin-and-recreate large-neighbourhood search
(LNS) — into a record machine: the engine generalized by `coldcore`
produced the first improvements to the classical covering-code tables
$K_q(n,R)$ for $q \ge 5$ since their 2011 revision [@Keri; @Marosi2026],
including bounds established on spaces of $10^{10}$ words.

The framework's organizing idea is borrowed from compiler construction
[@LattnerAdve2004]. Like LLVM, `coldcore` separates concerns around a
small typed intermediate representation: **problem plugins** (frontends)
lower a discrete space and neighbourhood structure to a narrow C ABI of
full-space *gain*, *loss*, and *count* transforms plus incremental ball
updates; a **GPU state core** (backend) implements memory placement,
kernels, and reductions once; and a problem-agnostic **search core**
(passes) runs unchanged over any plugin. Adding a new problem means
writing one axis-pass kernel and one host-side neighbourhood-table
builder — the search algorithms, memory management, checkpointing, and
verification hooks are inherited.

# Statement of need

The tables of upper bounds on $K_q(n,R)$ — the minimum size of a $q$-ary
code of length $n$ covering every word within Hamming distance $R$
[@CohenEtAl1997] — had been frozen for fifteen years: Kéri's reference
tables were last revised in November 2011 [@Keri; @KeriOstergard2005], and
no upper bound with $q \ge 5$ had been improved since. The obstruction is
computational. CPU local search evaluates one candidate move at a time by
enumerating its coverage ball; on cells with $10^8$–$10^{10}$ words the
search saturates long before it can distinguish good moves from mediocre
ones. Sampling and surrogate evaluation trade away exactly the signal —
small exact differences between near-tied moves — that decides whether a
record falls.

Full-space exact transforms change the economics: one $O(n\,|X|)$ sweep
prices *every* possible placement simultaneously and exactly, so lazy
greedy reproduces the exact argmax and LNS repair always sees the
globally best fill. What no existing tool provided is the infrastructure
to run this at scale: dense fields of tens to hundreds of gigabytes,
automatic placement between HBM and CPU-attached coherent memory,
incremental updates, and search drivers decoupled from the problem
definition. `coldcore` packages exactly that seam, with a pure-NumPy
[@Harris2020] reference backend so that correctness is testable without
a GPU.

# Software design

`coldcore` has three layers meeting at two narrow interfaces.

**Problem plugins (frontends).** A plugin owns the mathematics: what the
fields mean, how the separable transform computes them, and what "ball"
means. It compiles to a shared library exporting the v1 C ABI
(`src/include/coldcore/plugin.h`): typed init over mixed-radix axes,
full-space `transform` (three input modes producing count, gain, or loss
fields), incremental `ball_update`/`ball_gather`, and map queries. Four
plugins currently speak the ABI: `hamming` (covering codes, mixed radix
— the founding problem), `torus_linf` (dominating sets on torus grids
under Chebyshev balls), `lee` (covering codes under the Lee metric
[@GolombWelch1970]), and `grid_linf` (dominating sets of bounded grids
with open boundaries — the first non-shift-invariant family).
Orthogonally to the problem family, $\mu$-fold multiple covering (the
$K_q^\mu(n,R)$ tables) is a predicate change at the seam rather than a
new plugin, so it composes with every family and reduces exactly to the
classical case at $\mu = 1$.

**State core (backend).** Implemented once: dense int32 layer planes and
a uint16 multiplicity plane, each placeable per array in HBM, in
CPU-attached LPDDR accessed cache-coherently (on NVIDIA GH200, with
`mbind`-pinned allocations to defeat counter-driven page migration), or
in managed memory as the portable fallback. Placement is automatic from
array sizes against an HBM budget: streaming arrays (layers) tolerate
LPDDR, randomly-accessed arrays (`cnt`) stay in HBM. This is what lifts
the problem-size ceiling past device memory on a single node.

**Search core (passes).** Pure Python over a backend protocol, with the
GPU backend bound via ctypes and a brute-force NumPy reference backend
implementing the identical protocol for CPU-only tests and GPU parity
checks. Passes: exact lazy greedy, ruin-and-recreate under fixed or
adaptive operator selection, zero-loss peeling, swap-neighbourhood
polish, restart portfolios, and notch-ladder descent. The search core
never declares records; an `on_solved` callback hands every candidate
solution to an independent verification gate.

# Scale and performance

All numbers are paired A/B runs on the same cells and seeds on one GH200
(96 GB HBM + 480 GB LPDDR); details in `docs/benchmarks.md`. A full
coverage transform on $K_8(10,4)$ ($1.1\times10^9$ words, 24 GB) costs
0.143 s in HBM, 3.4 s with layers in pinned coherent LPDDR, and 8.7 s
managed — the coherent path is the committed beyond-HBM mode. At the
extreme, $K_{10}(10,5)$ ($10^{10}$ words, 260 GB of dense arrays)
sustains one *exact* full-space transform in 73–110 s — to our knowledge
the first exhaustive-evaluation transform at $10^{10}$ words on a single
node. The documentation also records the negatives: a loss pass that is
kept only for sparse regimes because it loses to the plain transform in
dense ones, the transform-dominated wall-clock at $10^{10}$ words (an
8-hour run completed 8 LNS rounds), and a measured failure mode —
word-level LNS stalls against *private-uniform* incumbents — by which
the framework delimits its own method [@Marosi2026].

# Research enabled

The methods packaged here produced the results of the companion paper
[@Marosi2026]: 26 new upper bounds on $K_q(n,R)$ for $5 \le q \le 15$ —
the first at $q \ge 5$ since 2011 — including $K_{10}(10,5) \le 5799$
(previously 7106) on a $10^{10}$-word space via the coherent-memory
path; the same paper adds 58 certified lower bounds from the
semidefinite hierarchy of @GijswijtPolak2025. Every code was verified by
tooling independent of the search engine. Generality has been exercised,
not just claimed: the `torus_linf` and `lee` plugins were each added in
a day without modifying the search core, and the identical passes
reproduce known optima through them — a 9-vertex dominating set of the
$9\times 9$ king torus, and the perfect Golomb--Welch Lee code in
$\mathbb{Z}_5^2$ [@GolombWelch1970] — verified against brute-force
reference backends.

# AI usage disclosure

In accordance with JOSS policy, we disclose that the software and this
manuscript were developed by the AI system Claude (Anthropic) — Claude
Code agents, Opus-class models, June–August 2026 — operating
autonomously on the author's hardware under the author's direction, with
the author reviewing and validating the outputs, the benchmark
methodology, and the claims made here. All record claims rest on
independent verification tooling rather than on the search engine's own
bookkeeping.

# Acknowledgements

The author thanks the maintainers of Kéri's covering-code tables, whose
careful curation defined the target, and acknowledges the NVIDIA GH200
Grace Hopper platform on which the coherent-memory results were obtained.

# References

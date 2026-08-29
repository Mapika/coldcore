# Symmetry

Every space coldcore searches carries a large automorphism group, and
until now the search core ignored it: it treated the `|X|` words as `|X|`
unrelated candidates. This page describes the symmetry layer —
`src/coldcore/symmetry.py` (group-invariant construction and orbit-restricted
search) and `src/coldcore/symmetry_exact.py` (isomorph-free exhaustive
settlement of tiny cells) — what each is for, and, with measurements,
when each one wins and when it does not.

The motivation is empirical. Three findings from the record campaigns
all point at the same missing concept:

* in the binary μ-fold table the 1990s **construction-based** bounds
  (group-invariant codes) beat our free search from n ≥ 14 — structure
  wins whole regimes;
* `K_8(4,2)` could not be settled because there was no isomorph-free
  exhaustive search;
* direct-sum (structured) incumbents are provably rigid against
  word-level LNS — the records paper documents it.

---

## 1. The group

A group element is a pair *(coordinate permutation, per-coordinate value
permutations)*: the general element of the wreath product

```
(S_{a_0} x S_{a_1} x ... x S_{a_{n-1}}) |x| S_n
```

which is the full automorphism group of the Hamming space and contains
the isometry groups of the Lee and torus-Chebyshev spaces as subgroups
(per-axis rotations and reflections, plus permutations of equal axes).
Acting on a word `x`:

```
(g.x)_j = v_j[ x_{p_j} ]
```

`p_j` names the **source** coordinate of output position `j` (so the
action is a numpy gather), and `v_j` permutes the alphabet of axis `j`.
Mixed radix is supported: `p` may only move coordinates of equal axis
size, which the constructor enforces.

```python
from coldcore.space import ProductSpace
from coldcore.symmetry import translation, coordinate_permutation

sp = ProductSpace(3, 4)                       # Z_3^4
g = translation(sp, [1, 0, 1, 1])             # x -> x + (1,0,1,1)
h = coordinate_permutation(sp, [1, 2, 3, 0])  # cyclic coordinate rotation

g.apply_words(words)          # (k, n) digits -> (k, n) digits
g.apply_index(sp, idx)        # int64 indices -> int64 indices, O(n) passes
g.compose(h)                  # h first, then g
g.inverse()
g.permutation(sp)             # the induced permutation of the whole index set
```

`apply_index` never materialises words: digit `s` of an index is
`(idx // pow_q[s]) % axes[s]`, and the image is the mixed-radix
recomposition of the permuted, value-mapped digits — `n` vectorised
passes over whatever array you hand it.

### Orbits

`Group.orbits()` returns an `OrbitPartition`: a CSR partition of the
whole space (`labels`, `members`, `starts`, `sizes`, `reps`), computed by
label propagation on the generator permutations (`lab = min(lab,
lab[sigma])` with pointer jumping, generators *and* inverses, so the
relation is symmetric; at the fixpoint `lab` is constant on each orbit
and equals its smallest word index). `Group.orbit_of(idx)` computes a
single orbit by closure and works at any space size.

`OrbitPartition.segment_sum(field)` turns any whole-space field (a gain
map, a loss map) into one number per orbit in a single call. That is the
bridge between the symmetry layer and the existing exact backends.

### Library subgroups

| function / spec term | group |
|---|---|
| `trivial_group` / `trivial` | one element: orbits are singletons, orbit search degenerates *exactly* to free search |
| `translation(sp, v)` | the single automorphism `x -> x + v` |
| `translation_group(sp, vecs)` / `translate:VEC` | translations by the given vectors — over `Z_q` the orbits are the **cosets of the linear code** they span. Aliased `linear_code_translations` |
| `diagonal_translations` / `diag[:S]` | `x -> x + c*(1,...,1)` |
| `value_shift(sp, j)` / `shift:J[:S]` | cyclic value shift on one axis (the q-ary translation subgroup of one coordinate) |
| `value_shifts` / `shifts` | all per-axis shifts — the full translation group |
| `coordinate_rotation` / `rot` | the cyclic rotation of coordinates |
| `coordinate_symmetric` / `sym` | the full symmetric group on coordinates |
| `monomial_group` / `mono` | the full Hamming automorphism group |
| `isometry_group(sp, family)` / `iso:FAMILY` | the isometry group of a shipped problem family |

Terms combine with `+`; a vector is written either as digits (`1011`) or
`x`-separated (`10x0x1x1`, matching `-p axes=9x9`). `translate:1011` and
`translate:1x0x1x1` are the same group.

```python
from coldcore.symmetry import parse_group
G = parse_group(sp, "translate:1011+translate:0112")   # ternary Hamming cosets
G.orbits()        # OrbitPartition(|X|=81, orbits=9, sizes 9..9)
G.order()         # 9
```

### Is my group an automorphism group?

`Group.is_automorphism_of(backend)` checks, by brute force over the whole
space, that every generator preserves the coverage relation
(`ball(g.w) == g.ball(w)`). It is a *diagnostic*, not a precondition:

> **Correctness of orbit search never depends on `G`.** Any union of
> orbits is an ordinary solution, scored by the backend's exact fields.
> A `G` that is not an automorphism group merely restricts the search to
> a bad subfamily — it can never produce a wrong answer.

The exhaustive module of §4 is the opposite: there `G` *must* be an
automorphism group, and the constructor refuses anything else.

A concrete example of the difference: translations are automorphisms of
`hamming`, `lee` and `torus_linf`, but **not** of `grid_linf` — the open
board has no wraparound, so edge balls are smaller and nothing is shift
invariant. Orbit search over a translation group on `grid_linf` still
returns valid dominating sets; they are just bad ones.

---

## 2. Orbit-restricted search

`SymmetricSearcher` runs the same three passes as `Searcher` — greedy
build, peel, notch descent — over different atoms. **A solution is a
union of chosen orbits.** Add and drop are whole orbits; a notch of the
descent is one orbit's worth of points, because a group-invariant
solution can only change size in orbit-sized chunks.

```python
from coldcore.reference import RefHammingBackend
from coldcore.symmetry import SymmetricSearcher, parse_group, orbit_solve

b = RefHammingBackend([3] * 4, 1)
s = SymmetricSearcher(b, parse_group(b.space, "translate:1011"), seed=1)
s.greedy_fill(10 ** 9)      # exact lazy greedy over orbits
s.peel()                    # drop orbits that feasibility does not need
code = s.descend(floor=1, notch_budget_s=30.0)

code, s = orbit_solve(b, parse_group(b.space, "translate:1011"))  # all three
```

### Why it is exact

Two facts, both resting on the existing exact fields:

**Orbit gain is an exact marginal.** `add_orbit(o)` applies the whole
orbit with one `ball_update(O, +1)` and reads the deficit back with
`uncovered()`. The returned gain is exactly what a from-scratch recount
says — no summing of per-word gains, so no overcount when two members'
balls overlap. `drop_orbit` is the mirror image.

**The gain map bounds it, for every orbit at once.** `gain_map()[w]` is
the exact per-word gain, so

```
bound(O) = sum_{w in O} gain_map[w]   >=   exact gain of adding O
```

(equality iff no word is newly covered by two members). One transform
plus one `segment_sum` gives `bound` for *all* orbits. And because each
per-word gain is non-increasing as the solution grows, so is `bound`.
An upper bound that only ever decreases is exactly what a lazy-greedy
bound heap needs, so `greedy_fill` — pop the best bound, score it
exactly, re-insert it when it no longer leads — is **exact greedy**
(Minoux), not a heuristic, with a single whole-space transform amortised
over the entire build.

The loss map is used differently and the docstrings say so: a zero
loss-map sum over an orbit is a *necessary* condition for the orbit to be
removable (any critical word in a member's ball makes the sum positive)
but not a sufficient one — a word covered twice from inside the same
orbit is not critical. So `peel` uses the sum as an exact filter and then
re-checks each candidate by actually dropping it.

### Hybrid: structured start, free polish

`SymmetricSearcher.free_searcher()` hands the group-invariant solution to
an ordinary `Searcher` (CLI: `--symmetry-polish`). The group-invariant
code is a plain index array that every verifier already accepts; the
word-level passes are then free to break the symmetry. This is the shape
the records paper's rigidity result predicts will be hard — free LNS
rarely improves a structured incumbent — and it is worth measuring per
cell rather than assuming.

---

## 3. When orbit search wins, and when it loses

Measured with `scripts/bench_symmetry.py --budget 30 --seeds 5` on the
pure-numpy reference backend (raw records in `docs/bench_symmetry.json`).
Both methods run greedy → peel → notch descent and stop the instant they
verify a solution of the target size `M*`; the time is the mean over the
seeds that got there, and `hit` says how many of 5 did — a mean over 1/5
seeds is a very different claim from a mean over 5/5, so both are shown.

**A methodology note, because it changed these numbers by 1000x.**
`Searcher.descend` will not start a notch it cannot finish and reserves
30 s of its deadline for that check. A free run given only its working
budget therefore never descends at all. The first version of this table
made exactly that mistake and reported "free search never reaches 12 on
K_2(6,1)"; with the reserve added, free search reaches it in 3.8 s on one
seed in five. The corrected numbers are below, and the reserve is a
named constant in both the benchmark and the tests.

| cell | \|X\| | orbits | M* | free s | hit | orbit s | hit | speedup | group |
|---|---|---|---|---|---|---|---|---|---|
| K_2²(6,1) | 64 | 16 | 20 | 14.16 | 3/5 | 0.0015 | 5/5 | **9438x** | `translate:010100+translate:111011` |
| Lee Z_5², μ=2 | 25 | 5 | 10 | 3.85 | 5/5 | 0.0004 | 5/5 | **9613x** | `translate:1x2` |
| K_4²(4,1) | 256 | 16 | 48 | 18.56 | 1/5 | 0.0065 | 5/5 | **2856x** | `translate:2x0x1x3+translate:1x3x0x3` |
| K_2³(8,1) | 256 | 64 | 96 | >98 | 0/5 | 0.0171 | 5/5 | **>1754x** | `translate:10000111+translate:01000110` |
| K_2(6,1) | 64 | 32 | 12 | 3.77 | 1/5 | 0.0035 | 5/5 | **1076x** | `translate:111111` |
| K_2⁴(8,1) | 256 | 64 | 128 | 10.30 | 4/5 | 0.0130 | 5/5 | **792x** | `translate:10000111+translate:00000110` |
| king 9x9 torus | 81 | 9 | 9 | 0.0054 | 5/5 | 0.0005 | 5/5 | 10.8x | `translate:3x1` |
| Lee Z_5² (GW) | 25 | 5 | 5 | 0.0011 | 5/5 | 0.0002 | 5/5 | 5.5x | `translate:1x2` |
| K_3(4,1) | 81 | 27 | 9 | 0.0044 | 5/5 | 0.0017 | 5/5 | 2.6x | `translate:1011` |
| K_3(5,1) | 243 | 81 | 27 | 0.0187 | 5/5 | 0.0114 | 5/5 | 1.6x | `translate:10111` |
| K_2²(3,1) | 8 | 4 | 4 | 0.0003 | 5/5 | 0.0004 | 5/5 | **0.7x** | `translate:111` |
| **K_4(4,1)** | 256 | 64 | 24 | 5.68 | 5/5 | >32 | 0/5 | **LOSS** | `translate:0x1x1x3` |
| **Lee 7x7 R=1** | 49 | 7 | 12 | 7.87 | 5/5 | >14 | 0/5 | **LOSS** | `translate:1x2` |
| **king 11x11** | 121 | 11 | 16 | 0.0102 | 5/5 | >22 | 0/5 | **LOSS** | `translate:10x2` |

`>M` = never reached `M*`; `M` is the smallest size that method did
verify.

**The honest reading.**

* Where the optimum is a union of cosets, the gap is three to four orders
  of magnitude: milliseconds against seconds, and reliably (5/5 seeds)
  against unreliably (1/5, 3/5, 4/5, and on `K_2³(8,1)` never inside
  30 s). Free search *can* get to most of these eventually — it is a good
  searcher — but it is rediscovering, one word at a time, a structure the
  group states outright.
* The μ-fold rows are the direct corroboration of the campaign finding:
  μ disjoint translates of a good code is exactly the kind of structure
  word-level LNS is worst at assembling, and it is the regime where the
  1990s construction-based bounds beat our free search.
* Where the optimum is **not** group-invariant, orbit search does not
  merely run slower — it cannot get there at all, and the gap is large:
  32 vs 24 on `K_4(4,1)`, 14 vs 12 on Lee 7x7, 22 vs 16 on the 11x11 king
  torus. The last two are the clearest failure mode: with a group of
  order 7 (resp. 11) acting freely, every orbit has 7 (resp. 11) points,
  so **no** union of orbits has 12 (resp. 16) points. The group has to
  divide the answer.
* On a trivially small cell (`K_2²(3,1)`, |X| = 8) orbit search is
  measurably *slower* — 0.7x — because the orbit bookkeeping is not
  repaid. The win needs a cell big enough for the reduction to matter.
* The wins are only as good as the group you supply. Choosing it is the
  modelling step, exactly as it was for the 1990s constructions; the
  layer mechanises the search over unions of cosets, not the choice of
  the coset structure.
* Free search stays the default. Orbit search is opt-in
  (`--symmetry SPEC`) and is best used as a *second* engine: milliseconds
  to try, and when it wins it wins by orders of magnitude.

Reproduce with:

```
python3 scripts/bench_symmetry.py --budget 30 --seeds 5 --json out.json
python3 scripts/bench_symmetry.py --only K_2 --budget 60
```

---

## 4. Isomorph-free exhaustive settlement

`src/coldcore/symmetry_exact.py` answers the question the search core
structurally cannot: *is there nothing smaller?* For a tiny cell,
`decide(M)` returns `"feasible"` with a witness, `"infeasible"` (an
exhaustive proof), or `"unknown"` (a node/time limit stopped it).

```python
from coldcore.symmetry_exact import IsomorphFreeSearch, settle
M, witness, trace = settle(RefHammingBackend([2] * 5, 1), hi=9)
# M = 7, and trace[-1] == (6, "infeasible", {...})
```

### The generator, and why this one

Three ingredients:

1. **Coverage branching (completeness).** At a node with partial code
   `P`, take `u* = min` of the words still short of their requirement.
   Every completion must contain a codeword whose ball reaches `u*`, so
   branching over `coverers(u*)` is exhaustive and never wastes a branch
   on a word that cannot help.

2. **Isomorph rejection.** Two exact rules, with a group `G` of
   automorphisms of the coverage relation:
   * *at the root*, the first codeword may be taken to be one `G`-orbit
     representative per orbit of the whole space (a non-empty solution
     contains some codeword, and `G` may be applied freely). For any
     translation group — transitive — that is **one** branch instead of
     `|coverers(u*)|`;
   * *at inner nodes*, with `H = Stab_G(P)` and `H* = Stab_H(u*)`, the
     branch set collapses to one representative per `H*`-orbit. The
     argument: if `C` completes `P` and `c` in `C ∩ coverers(u*)`, then
     for `h` in `H*` the image `h.C` also completes `P`, has the same
     size, and contains `h.c` in `coverers(u*)`. The stabiliser is
     maintained incrementally down the branch, so it costs almost
     nothing once it has collapsed.

3. **Counting bounds.** With `r = M - |P|` codewords left and deficit
   `D`, one codeword removes at most `|B|` of it, so `r*|B| >= D` is
   necessary; at `r <= 1` the exact test (is there one ball containing
   everything still missing?) replaces it.

**Why not orderly generation or canonical augmentation?** Orderly
generation — accept only partial codes that are lex-minimal in their
`G`-orbit — has the hereditary property it needs (if a sorted set is
lex-minimal, so is the set with its largest element removed), but
coverage branching does not add elements in increasing order, so the
canonical partial codes are not reachable along canonical paths. Making
the two fit requires McKay-style canonical augmentation over *all*
augmentations of each node, which throws away ingredient (1) — and (1) is
the pruning that actually pays on covering problems. Stabiliser-reduced
coverage branching keeps both, and is what the classical covering-design
enumerations use.

Two guardrails: `G` is brute-force verified to be an automorphism group
(a wrong group raises, it never yields an unsound proof), and at μ > 1
the branch set excludes words already in `P` — an incumbent *can* reach a
word that is merely short of its multiplicity, and picking it again would
build a multiset rather than a code. (That bug was caught by the μ-fold
test: it "settled" `K_2²(4,1) = 7` with the all-zero word twice; the
answer is 8, confirmed by a brute-force sweep of all 7-subsets.)

### Validated cells

Every one of these is settled from scratch by `settle()` in the test
suite (`tests/test_symmetry_exact.py`) and matches the literature:

| cell | optimum | cell | optimum |
|---|---|---|---|
| K_2(3,1) | 2 | K_2(4,1) | 4 |
| K_2(5,1) | 7 | K_3(3,1) | 5 |
| K_3(4,2) | 3 | K_4(3,1) | 8 |
| Lee Z_5² R=1 | 5 | king 6x6 torus | 4 |
| king 9x9 torus | 9 | king 8x8 board | 9 |
| K_2²(3,1) | 4 | K_2³(3,1) | 6 |
| K_2²(4,1) | 8 | K^2 Lee Z_5² | 10 |

Two larger settlements, run outside the suite for time:

| cell | optimum | nodes | wall |
|---|---|---|---|
| K_2(6,1) | 12 | 1.23e6 | 7.3 s (6.7 s of it enumerating \|G\|=46080) |
| K_3(4,1) | 9 | 3.7e4 | 4.0 s (mostly \|G\|=31104 setup) |

`K_2(6,1) = 12` is worth noting twice: §3 shows orbit search *finding*
the 12 that free search cannot, and this module *proving* that 11 is
impossible — the two halves of a settled cell, from the same layer.

### K_8(4,2): the verdict is "no"

`K_8(4,2)` is in `{22, 23}`; deciding it means deciding whether a
22-codeword cover of `Z_8^4` with radius 2 exists. Measured on this
machine (single core, pure Python/numpy):

| quantity | value |
|---|---|
| space, ball | \|X\| = 4096, \|B_2\| = 323 |
| group used | translations of `Z_8^4`, \|G\| = 4096 (transitive → 1 root branch) |
| node throughput | **1.15e6 nodes/s** = 4.1e9 nodes/hour |
| tree size at M=22, Knuth estimate (20000 dives) | mean **4.97e47**, median 4.74e47 nodes |
| implied wall time | 4.3e41 s ≈ **1.4e34 years** |

The estimate is Knuth's random-dive estimator (1975): a dive multiplies a
running weight by the surviving branching factor at each node and
descends into a uniform child; the sum of weights is an unbiased
estimator of the tree the real search traverses. It is calibrated in the
test suite against real, completed searches — mean 229 vs true 231 nodes
on `K_2(5,1)` M=6, 12.7 vs 13 on `K_3(3,1)` M=4, 3.33e5 vs 3.68e5 on
`K_2(6,1)` M=11.

**Would a better group rescue it?** Not this cell — though the group is
emphatically the right thing to reach for, and §6 shows it buying 64x on
a cell where it *can* be represented. Two separate obstacles here:

* the full monomial group of `Z_8^4` has order `8!^4 * 4! = 6.34e19`, so
  it cannot be represented at all — the implementation needs an
  index-permutation table, and even the trick that makes `Z_2^8`'s
  10.3M-element group tractable (`binary_monomial_table`) would need
  6.34e19 rows here;
* and even a hypothetical *perfect* isomorph rejection — every
  isomorphism class visited exactly once, which no practical scheme
  achieves — divides the tree by at most `|G|`, leaving `7.8e27` nodes,
  or 2.2e14 years at the measured rate.

So the gap is not a constant factor away from closing, and the reason is
not that symmetry is the wrong lever.

**What pruning would be needed.** Closing `1e34` needs strictly stronger
*bounds*. The cheapest one was tried and **measured as a wash** — see
`strong_bound` and §6 — so this list is now shorter and better informed:

* an LP relaxation of the residual set-cover at each node, or the excess
  / surplus counting bounds developed for covering codes
  (Habsieger-, Östergård-style), replacing the very weak `r*|B| >= D`
  used here. But note what the exact greedy bound did: it cut the
  estimated tree by ~1000x (4.97e47 -> 3.23e44) and cost ~820x in
  throughput (1.15e6 -> 1.4e3 nodes/s), for no net gain. A bound that
  helps has to be far cheaper per node than a whole-space rescan, or far
  stronger than greedy — probably incremental, maintained along the
  branch rather than recomputed;
* full-group canonical augmentation at every level, which needs a
  canonical-form algorithm for codes under the wreath product rather
  than an enumerated element list (the current implementation caps at
  `MAX_GROUP = 2^17` elements for exactly this reason);
* and even then, the honest expectation from the covering-code
  literature is that `K_8(4,2)` is settled by *analytic* lower bounds
  plus a construction, not by exhaustive generation.

**What the layer can still do for it.** The other half of the pincer is
an upper bound: search the group-invariant subfamily for a 22-codeword
witness. `scripts/probe_k8_42.py` sweeps every order-2 translation
group of `Z_8^4` — translations act freely, so every orbit has size
`|G|`, and `22 = 2 x 11` forces `|G| = 2` — with orbit greedy → peel →
descent. Over 528 passes (2 h, single core) it reaches **M=32**,
against **M=35** for a free greedy+peel on the same reference backend —
better, and nowhere near 22. `K_8(4,2)` needs a genuinely different idea;
this layer says so with numbers instead of guessing.

---

## 5. K(8,2,2): a worked settlement attempt

The binary μ-fold cell `K(8,2,2)` — `n = 8`, `R = 2`, `μ = 2`, so 256
words and `|B_2| = 37` — is an open entry in the μ-fold paper: our search
gives `K <= 18` (verified independently here: 18 distinct words, deficit
0, min count 2) against a published lower bound of 16, so
`K(8,2,2) ∈ {16, 17, 18}`.

**What was settled.** `M = 14` is **infeasible**, exhaustively — so
`K(8,2,2) >= 15`. Covers are monotone (any `M`-cover extends to an
`(M+1)`-cover by adding any unused word), so every `M <= 14` is excluded
with it. This is weaker than the published bound of 16, so it is a
confirmation of part of it rather than a new result; the value is that it
is machine-checkable end to end.

**What the group is worth.** This cell is the clean demonstration that
isomorph rejection is the dominant lever, not a footnote:

| group | order | M=14 nodes | wall |
|---|---|---|---|
| translations x C_8 | 2 048 | 7 802 251 | 12.1 s |
| **full monomial `S_2^8 ⋊ S_8`** | **10 321 920** | **120 982** | **0.31 s** |

a 64x cut in nodes and 39x in wall time. The full group is past what
`Group.elements()` can enumerate one Python object at a time, so
`binary_monomial_table(space)` builds its index-permutation table
directly — the `n!` bit permutations in one numpy expression, then the
`2^n` XOR translates by broadcasting (10.6 GB at `n = 8`).

**What is out of reach.** Knuth estimates under that group, at a measured
3.8e5 nodes/s (calibrated on this very cell: `M=14` estimated 5.83e6 vs
6.70e6 actual):

| M | mean nodes | median | mean cost | verdict |
|---|---|---|---|---|
| 15 | ~5e10 (heavy tail, still rising at 1600 dives) | 3.3e8 | ~1.5 core-days | **launched** |
| 16 | 3.3e14 | 5.9e11 | ~1e4 core-days (27 yr) | out of reach |
| 17 | 3.2e17 | 7.8e14 | ~1e7 core-days (27 000 yr) | out of reach |

So the cell is **not settled**: deciding `M = 17` is what would close it,
and it is seven orders of magnitude beyond a reasonable budget. `M = 15`
is inside budget and is running (`scratch_symexact/k822_m15.py`); it
proves the published `>= 16` independently, and nothing more.

**Heuristic evidence on M=17**, for what it is worth (it is not a proof):
seeded LNS from the verified 18-word cover, 8 seeds x 100 s with the
deficit operator and adaptive ruin selection, walls at deficit 12 on
*every* seed — a flat, reproducible wall rather than a near miss.

---

## 6. The four tight q-ary cells: estimate first, then decide

The same protocol applied to the four tightest open cells of the q-ary
μ-fold table, all of which had just survived a dedicated 3-hour
`break_wall` endgame pass with no movement. Every one is tiny, which is
exactly what makes them look like exhaustive-search targets — and only
one of them is.

Groups: the full monomial group `S_q^n ⋊ S_n` wherever its table fits
(`monomial_table`), and the `AGL(1,q)^n ⋊ S_n` fallback where it does
not. Rates are measured per cell, on this machine, with the other jobs
running:

| cell | bounds | \|X\| | \|B\| | μ | sphere | group | \|G\| | nodes/s |
|---|---|---|---|---|---|---|---|---|
| K_4(3,1;4) | 26–28 | 64 | 10 | 4 | 26 | full monomial | 82 944 | 2.8e5 |
| K_5(3,1;4) | 39–42 | 125 | 13 | 4 | 39 | full monomial | 1.04e7 | 3.2e5 |
| K_4(4,1;3) | 61–64 | 256 | 13 | 3 | 60 | full monomial | 7.96e6 | 3.4e5 |
| K_5(4,2;4) | 23–25 | 625 | 113 | 4 | 23 | AGL(1,5)⁴ ⋊ S₄ | 3.84e6 | 4.7e4 |

`K_5(4,2;4)` is the one that needs the fallback: its full monomial group
is `(5!)^4 · 4! = 4.98e9` elements, a 12.4 TB table. Note also the node
rate dropping by 7x on it — the state is `|X|`-bit masks, so a 625-word
space costs proportionally more per node than a 64-word one.

Knuth estimates (400 dives each; means, which is the readout the
calibration in §4 validated — the medians of this estimator run orders
of magnitude low on heavy-tailed trees and are *not* a size estimate):

| cell | M | mean nodes | core-days | verdict |
|---|---|---|---|---|
| **K_4(3,1;4)** | **26** | **3.34e9** | **0.14** | **tractable — launched** |
| K_4(3,1;4) | 27 | 1.70e14 | 7 060 | out of reach |
| K_5(3,1;4) | 39 | 2.99e15 | 1.08e5 | out of reach |
| K_5(3,1;4) | 40 | 2.44e19 | 8.8e8 | out of reach |
| K_5(3,1;4) | 41 | 4.31e24 | 1.6e14 | out of reach |
| K_4(4,1;3) | 61 | 1.22e22 | 4.1e11 | out of reach |
| K_4(4,1;3) | 62 | 2.75e26 | 9.2e15 | out of reach |
| K_4(4,1;3) | 63 | 2.23e30 | 7.5e19 | out of reach |
| K_5(4,2;4) | 23 | 3.58e23 | 8.8e13 | out of reach |
| K_5(4,2;4) | 24 | 7.75e30 | 1.9e21 | out of reach |

So exactly one of the four is decidable: `K_4(3,1;4)` at `M = 26`, its
sphere bound. **Either verdict improves the table** — feasible closes the
cell at 26, infeasible raises the lower bound to 27 and narrows
{26,27,28} to {27,28} — which is why it was worth the budget even though
`M = 27` behind it is 7 060 core-days away.

Note how fast this falls off. `K_4(3,1;4)` at M=26 is 3.3e9 nodes; one
codeword later it is 1.7e14, and the next cell up (25% more words) is
1e5 core-days at *its* sphere bound. Exhaustive settlement is not a
technique that scales into this table — it settles the corner cases and
nothing else.

**Heuristic evidence** (not proof), 6 seeds x 240 s of deficit-ruin LNS
per size: no witness at M=26 *or* M=27 for `K_4(3,1;4)`, and none at
M=39 for `K_5(3,1;4)`. The upper bounds in the table (28, 42) are what
sustained search produces; the sizes just below them are not merely
unproven, they are places the searcher does not go.

---

## 7. Two accelerations, one of which did not pay

**Layered bitsets for μ > 1** (kept). The μ-fold state is μ bitmasks,
`lay[k]` = the words already covered at least `k+1` times, updated by
`lay'[k] = lay[k] | (lay[k-1] & B)` — a word reaches `k+1` iff it was
already there, or was at `k` and the new ball covers it. That replaced a
whole-space numpy count array copied at every node. On `K(8,2,2)` M=14 it
is **2.3x** faster (28.7 s -> 12.3 s) for a
**bit-identical tree** (7 802 251 nodes both ways), which is also the
cleanest possible check that the two formulations agree.

**The exact greedy bound** (opt-in, `strong_bound=`, default **off**).
Adding a codeword reduces the deficit by exactly `|B ∩ short|`, where
`short` is the set of words still below their multiplicity requirement,
so the free bound `r * |B| >= D` can be sharpened by replacing `|B|` with
the largest `|B ∩ short|` any ball still achieves (valid because `short`
only shrinks). It works — and it does not pay:

| cell | nodes | throughput | net |
|---|---|---|---|
| `K(8,2,2)` M=14 | 7.80e6 -> 6.70e6 (-14%) | 13.1 s -> 13.8 s | wash |
| `K_8(4,2)` M=22 | est. 4.97e47 -> 3.23e44 (~1000x) | 1.15e6 -> 1.4e3 nodes/s (~820x) | wash |

Widening the trigger does nothing (the free bound is nearly binding
throughout these dense cells, so the strong bound already fires wherever
it can). The honest lesson is in §4: a whole-space rescan per node gives
back everything it wins.

---

## 8. CLI

```
# orbit-restricted solve
python3 -m coldcore solve --backend ref -p q=3,n=4,R=1 --symmetry translate:1011

# same, reached through the generic parameter dict
python3 -m coldcore solve --backend ref -p q=2,n=6,R=1 -p group=translate:111111

# orbit descent, then let word-level search try to break the symmetry
python3 -m coldcore solve --backend gpu -p q=2,n=8,R=1,mu=4 \
    --symmetry translate:10000111+translate:00000110 \
    --symmetry-polish --hours 2 --out best.txt
```

`--symmetry` prints the orbit count and the search-space reduction before
it starts, writes every verified improvement through the same
`--out` path as the free solver, and exits with a diagnostic if no union
of orbits covers the space at all (the group is too coarse for the cell).

## 9. Scaling notes

The orbit machinery is pure Python/numpy and drives the **existing**
backends — there is no new CUDA. Two size limits follow from that:

* `MAX_PARTITION = 2^24` — whole-space orbit partitions are materialised
  as index arrays. Above it, `Group.orbit_of` still computes individual
  orbits, and the per-orbit fields would need a device-side segment sum.
* `gain_bounds()` reads the whole map back with `map_read_at`; on a GPU
  backend that is one full device-to-host transfer per refresh, which is
  fine at these sizes and would want a device-side `segment_sum` for big
  cells.

The exhaustive module holds one Python big-int bitmask per word
(`|X|` bits each) and the group's index permutations as an `int32`
`(|G|, |X|)` array, so it is designed for `|X| <= ~2^12` and
`|G| <= 2^17` when the group is enumerated element by element. The
`table=` argument takes a precomputed `(|G|, |X|)` table instead, which
is what makes the 10.3M-element monomial group of `Z_2^8` usable
(`binary_monomial_table`, 10.6 GB); there the limit is memory, at
`|G| * |X| * 4` bytes.

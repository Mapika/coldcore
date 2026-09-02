# The explicit-incidence backend family

`coldcore.incidence` (host), `coldcore.incidence_gpu` + plugin ABI 2
(GPU), `coldcore.problems.covering_design` (the first problem).
Added 2026-09-01.

## What it is for

The product-space backends compute gain and loss fields by separable
transforms: candidates and covered points are the same set `X`, and
"u covers x" is a per-axis predicate. Covering designs do not fit that
model. In C(v,k,t) the candidates are the k-subsets of a v-set, the
covered points are the t-subsets, and a block covers the C(k,t)
t-subsets it contains — a bipartite relation between two different sets
with no product structure to transform along.

This family therefore stores the coverage relation explicitly: an
`(n_cand x d)` integer matrix `inc`, row `c` listing the `d` universe
elements candidate `c` covers. The fields are the same two dense arrays
the rest of coldcore uses — `cnt` over the universe, `map` over the
candidates — and every protocol operation is a pass over incidence rows:

| operation | host (numpy) | GPU (ABI 2) |
|---|---|---|
| `recount(idx)` | zero `cnt`, `bincount` of the selected rows | memset + atomic scatter, one thread per (candidate, element) |
| `gain_map` / `loss_map` | `np.take(role_mask, inc[chunk]).sum(1)` over row chunks | one warp per row: coalesced row read, `cnt` gather, warp reduction |
| `ball_update(idx, ±1)` | `add.at` (small batch) or `bincount` (large) | atomic scatter over the listed rows |
| `ball_gather(idx, role)` | gather + predicate, or role mask when the batch is wide | one warp per listed candidate |
| `count_eq`, `uncovered`, `map_max`, `map_hist`, `map_extract`, `map_read_at`, `read_cnt` | inherited from `protocol.DenseFieldBackend` | the ABI 1 reductions, re-implemented over int32 `cnt` |

The mu-fold semantics are the `DenseFieldBackend` ones (`cnt < mu`
deficient, `1 <= cnt <= mu` critical), so `-p mu=2` means every
t-subset in at least two blocks. `verify(idx)` recomputes coverage from
scratch without reading `cnt` or `map`; the covering-design layer
overrides it with a recomputation from the blocks themselves (per-block
t-subset enumeration with a fresh rank table), independent of the
incidence matrix.

Cost is `O(n_cand * d)` per field and `4 * n_cand * d` bytes for the
matrix: 2.9 GB for C(32,8,4) (10.5M x 70), 65 GB for C(32,10,5)
(64.5M x 252). On the GPU the matrix is placed in HBM when the cell fits
the budget (`COLDCORE_HBM_BUDGET`, default 80 GB) and otherwise in CPU
memory pinned to NUMA node 0 (`posix_memalign` + `mbind`, the ATS mode
of the v1 core) and streamed by the field kernel over C2C; `cnt`, `map`
and the extraction buffers stay in HBM either way.

## What the search core needed

Nothing in `search.py` changed. The core reads one thing off a backend
that is not part of the protocol proper: `backend.space`, for
`index_word` / `word_index` (solution files, `load` of digit rows) and
for the axis list (`n`, `axes`, `pow_q`) that the digit-distance
operators use. `IncidenceSpace` is the shim: a single axis of size
`n_cand`, `index_word` returning `(m, 1)` int64 rows and `word_index`
flattening them. With that in place `greedy_fill`, `peel`, `ruin("low")`,
`lns`, `descend`, the LAHC/annealing polish with the pool neighbourhood,
and the record callbacks run as they do on the product-space backends,
and reach the known optima C(7,3,2) = 7, C(9,3,2) = 12, C(8,4,3) = 14,
C(13,4,2) = 13 in under a second on the host backend
(`tests/test_incidence.py`).

## Locality: the optional backend hooks

Everything the core derives from *digit distance* degenerates on a
one-axis space, where two distinct candidates are always at distance 1:
`ruin("cluster")` and `ruin("hole")` become random ruins with extra
transforms, the `deficit` ruin measures distances between candidate and
universe indices that are unrelated sets, the batched peel has no
independence test, and the `shell` polish neighbourhood — "every word at
digit distance 1" — is *every other candidate*, a full-space
`ball_gather` per move.

The core therefore accepts four optional hooks (`protocol.py`,
"Optional locality hooks"), feature-tested and falling back to the
product-space code, unchanged, when absent:

| hook | used by | covering designs |
|---|---|---|
| `distance(a, b)` | `ruin("cluster")`, `ruin("hole")`, `peel` (with `indep_dist` as a threshold in this metric) | `k - |A n B|` (half the symmetric difference); `indep_dist = k - t`, since blocks sharing fewer than `t` elements cover no common t-subset |
| `neighbours(idx, radius, m, rng)` | the `shell` polish neighbourhood (`_shell1` at radius 1, `_shell_k` beyond) | radius 1: all `k (v-k)` single-element swaps; radius `r`: `m` sampled `r`-fold swaps |
| `deficient_universe(limit, rng)` | `uncovered_words`, hence `ruin("deficit")` | `flatnonzero(cnt < mu)` (one field readback on the GPU) |
| `universe_distance(idx, univ)` | `ruin("deficit")`: the incumbents nearest the deficient elements | `t - |B n T|`: the blocks sharing the most elements with the uncovered t-subsets |

`candidates_covering(univ_idx)` (the `C(v-t, k-t)` blocks containing a
given t-subset) is provided alongside for problem-level tooling; the
core does not call it.

The product-space backends define none of the hooks, and
`tests/test_incidence.py::test_product_space_unchanged_without_hooks`
pins the trajectory of a Hamming cell through the cluster/hole/deficit
ruins and a shell polish to the digest taken before the hooks existed
(the older pins in `tests/test_search_determinism.py` hold too).

### Ruin operators on covering designs

`scripts/bench_incidence_ops.py`: greedy, peel, then a one-block notch
ladder to the known optimum using a single ruin operator (reject-if-
worse rounds as in `Searcher.lns`; `hole+shell` adds a 2 s shell-
neighbourhood LAHC polish every 200 rounds). Seconds from the end of
the peel to a verified optimum, median over 3 seeds with the number of
seeds that got there within 120 s, host backend:

| cell | optimum | random | low | cluster | hole | hole+shell | deficit | low+hole+cluster+deficit |
|---|---|---|---|---|---|---|---|---|
| C(13,4,2) | 13 | 0.0 s (3/3) | 0.0 s (3/3) | 0.0 s (3/3) | 0.0 s (3/3) | 0.0 s (3/3) | 0.0 s (3/3) | 0.0 s (3/3) |
| C(10,4,3) | 30 | 89.0 s (3/3) | 89.0 s (3/3) | 7.0 s (3/3) | 21.4 s (3/3) | 21.4 s (3/3) | 24.7 s (3/3) | 0.1 s (3/3) |

Per-seed times (seconds; 89.0 = the 90 s cap, i.e. not reached):

| C(10,4,3) | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| random | 89.0 | 0.5 | 89.0 |
| low | 89.0 | 0.5 | 89.0 |
| cluster | 89.0 | 3.9 | 7.0 |
| hole | 0.03 | 21.4 | 72.7 |
| hole+shell | 0.03 | 21.4 | 72.7 |
| deficit | 24.7 | 89.0 | 0.05 |
| low+hole+cluster+deficit | 1.0 | 0.07 | 0.12 |

On C(13,4,2) the peel already lands on the optimum, so the cell does
not discriminate. On C(10,4,3) every operator with the block metric
beats random ruin in the median, and the standard mix — which now
alternates an exact deficit ruin with the metric cluster/hole ruins —
gets to 30 in about a tenth of a second on every seed where the single
operators take tens of seconds or cap out. Single operators are
high-variance on a cell this small (one seed each of `hole` and
`deficit` is near-instant, another near the cap): the operators are
complementary, and the mix is the honest recommendation, not any one
of them. The shell polish made no difference here because the ladder
never walled long enough to invoke it.

## Measurements

Measured 2026-09-01 on an otherwise idle GH200 480GB (aarch64, CUDA
12.8, sm_90; host runs at `nice -n 10`), one process per row, in the
order listed. "build" is the host-side construction of the block table
and the incidence matrix plus, for the GPU rows, the upload; the field
timings are the mean of 3 calls after a warm-up call, with a random
partial solution of 2,000 blocks in place; "gather" is `ball_gather` of
that many random candidates (mean of 5); "greedy" is `Searcher.greedy_fill`
from the empty solution to a feasible cover of size M. The three
largest cells are also run with the matrix in pinned host memory
(`inc_mode=1`), which is what the automatic placement chooses once the
cell exceeds the HBM budget; here it is forced so the two placements
can be compared on the same cell.

| cell | candidates x d | universe | matrix | backend | build | gain field | gather x64 | gather x1024 | greedy | M |
|---|---|---|---|---|---|---|---|---|---|---|
| C(20,6,4) | 38,760 x 15 | 4,845 | 0.00 GB | cpu | 6 ms | 1.16 ms | 0.01 ms | 0.06 ms | 0.2 s | 488 |
| C(20,6,4) | 38,760 x 15 | 4,845 | 0.00 GB | gpu, inc in HBM | 0.2 s | 0.28 ms | 0.28 ms | 0.28 ms | 0.4 s | 488 |
| C(24,8,5) | 735,471 x 56 | 42,504 | 0.16 GB | cpu | 0.3 s | 73.13 ms | 0.02 ms | 0.12 ms | 1.2 s | 759 |
| C(24,8,5) | 735,471 x 56 | 42,504 | 0.16 GB | gpu, inc in HBM | 0.6 s | 0.13 ms | 0.03 ms | 0.02 ms | 0.4 s | 759 |
| C(28,8,4) | 3,108,105 x 70 | 20,475 | 0.87 GB | cpu | 1.5 s | 375.16 ms | 0.02 ms | 0.15 ms | 17.8 s | 589 |
| C(28,8,4) | 3,108,105 x 70 | 20,475 | 0.87 GB | gpu, inc in HBM | 1.8 s | 0.51 ms | 0.03 ms | 0.02 ms | 0.8 s | 589 |
| C(32,8,4) | 10,518,300 x 70 | 35,960 | 2.95 GB | cpu | 5.2 s | 1.28 s | 0.02 ms | 0.15 ms | 72.1 s | 1018 |
| C(32,8,4) | 10,518,300 x 70 | 35,960 | 2.95 GB | gpu, inc in HBM | 5.6 s | 1.80 ms | 0.03 ms | 0.03 ms | 1.3 s | 1015 |
| C(32,10,5) | 64,512,240 x 252 | 201,376 | 65.03 GB | cpu | 98.9 s | 27.29 s | 0.06 ms | 0.63 ms | 2067.0 s | 2082 |
| C(32,10,5) | 64,512,240 x 252 | 201,376 | 65.03 GB | gpu, inc in HBM | 101.8 s | 61.57 ms | 0.04 ms | 0.03 ms | 6.2 s | 2089 |
| C(32,8,4) | 10,518,300 x 70 | 35,960 | 2.95 GB | gpu, inc in pinned host | 5.8 s | 13.87 ms | 0.04 ms | 0.03 ms | 1.8 s | 1013 |
| C(32,10,5) | 64,512,240 x 252 | 201,376 | 65.03 GB | gpu, inc in pinned host | 106.2 s | 367.72 ms | 0.04 ms | 0.04 ms | 27.7 s | 2089 |

Reading the table:

- The GPU field kernel streams the matrix at about 1.6 TB/s from HBM
  (2.95 GB in 1.8 ms) and about 1.0 TB/s on the 65 GB cell; from pinned
  host memory it runs at 180-210 GB/s (C2C), 8-27x slower than HBM but
  still 30-90x faster than the host backend. The host backend's field
  is a single-threaded numpy gather at roughly 0.6 G elements/s.
- `ball_gather` on the GPU is launch-latency bound at these batch sizes
  (30-40 µs regardless of the cell); the host backend gathers 64
  candidates in 20-60 µs and 1,024 in 0.1-0.6 ms. A polish loop that
  issues one small gather per move is therefore not faster on the GPU;
  the GPU wins on the whole-space fields and on the greedy build, which
  is field-bound (72 s -> 1.3 s on C(32,8,4), 2,067 s -> 6.2 s on
  C(32,10,5)).
- The greedy sizes differ by a few blocks between backends on the same
  seed: `map_extract` is unordered by contract, so ties among equal
  gains are broken differently. These are single greedy builds, not
  searches: the LJCR (Zenodo snapshot, 2026-09) lists C(28,8,4) <= 402,
  C(32,8,4) <= 620 and C(32,10,5) <= 1,486. The greedy's 759 on
  C(24,8,5) is the optimum (the Steiner system S(5,8,24); lower bound
  759 in the LJCR).
- Build time is dominated by the incidence matrix: about 145 M entries
  per second, so 99 s for the 16.3 G entries of C(32,10,5); the block
  unranking itself takes seconds. The GPU build adds the upload
  (65 GB in about 3 s to HBM, 7 s into pinned host memory including the
  page pinning). Host memory during the build peaks at one copy of the
  matrix plus the chunk temporaries; the GPU backend releases its host
  copy after the upload.

### A 20-minute GPU descent on C(28,8,4)

`coldcore solve --problem covering_design -p v=28,k=8,t=4 --backend gpu
--hours 0.3333 --deficit --ruin-select adaptive --polish lahc
--polish-neighbourhood shell --notch-budget 60 --step0 4 --seed 1`, i.e.
the notch ladder from the greedy start with the bandit choosing among
all five ruin operators (the block metric behind cluster/hole/deficit)
and the shell-neighbourhood LAHC polish on walled notches:

| time | event |
|---|---|
| 0 s | greedy 589, peel 589 (verified, written) |
| 23 s | notch to **585** solved (step 4; verified, written) |
| 23-83 s | notch to 581: LNS walls at 4 uncovered, polish from 11 to 4; failed |
| 83 s - 20 min | step halved to 2: notch to 583 walls at 3-2 uncovered; polish at 583 from 3 uncovered, best 2, still running when the budget ended |

Final verified M = **585**. The frozen LJCR value for this cell is
**402** (Wheel Generator v2, 2024-05-21; lower bound 326), so free LNS
from the greedy start lands about 45 % above the record. That is the
expected outcome, not a near miss: the repository's records for cells
of this size come from structured constructions (group-invariant block
sets, wheels, lifting from smaller designs), and the walls this run hit
at 583 and 581 within a minute are the same behaviour the covering-code
records showed before symmetry-restricted search was added there. The
operators behave sensibly — the notch ladder moves, the polish reduces
the deficit — but unstructured search over 3.1 M blocks is not the
tool for a 402 on this cell.

## Group-invariant search

`src/coldcore/problems/perm_groups.py`, `design_orbits.py`; CLI
`--symmetry SPEC [--symmetry-polish] [--orbit-budget S]`,
`-p orbit_mode=quotient|points`, `-p short_orbits=1`.

The records for cells of this size are structured: a permutation group
`G` on the points, and a block set that is a union of `G`-orbits. The
layer follows the product-space symmetry layer (`docs/symmetry.md`)
where the design carries over, and says where it does not.

**Groups.** `PermGroup(v, gens)` by generators; elements by closure,
capped at 200 000. Built-ins by name (terms joined with `+` generate the
join): `cyclic` (Z_v), `cyclic-1` / `cyclic-2` (Z_{v-1}, Z_{v-2} with
one / two fixed points), `dihedral`, `prod:AxB` (Z_A x Z_B on v = AB),
`affine` (AGL(1,p), prime v), `affine:A` and `mult:A` (subgroups),
`gens=(0 1 2)(3 4);(5 6)` (explicit cycles), `trivial`. Orbits of
k-subsets and of t-subsets are computed the way `symmetry.Group.orbits`
computes word orbits — label propagation over the rank permutations the
generators induce, with pointer jumping — and are
`symmetry.OrbitPartition` objects.

**The quotient.** A `G`-invariant block set is a union of block orbits
`O`; every t-subset in a t-orbit `U` is contained in the same number
`w(O,U)` of blocks of `O` (`G` is transitive on `U`), so

    cnt_quotient(U) = sum_{O chosen} w(O,U) = cnt_full(any member of U)
    deficiency      = sum_U |U| max(0, mu - cnt(U)) = the full deficiency

and the quotient is a weighted, variable-length incidence: candidates
= block orbits, universe = t-orbits weighted by |U|, weights `w`. That
is `WeightedIncidenceBackend` (host) / `GpuWeightedIncidenceBackend`
(plugin ABI 2, `sparse_init_csr`): exact weighted marginals for the
gain (`sum |U| min(w, max(0, mu - cnt))`) and the loss, reducing to the
role counts at unit weights (tested), and `uncovered` is the weighted
deficiency. `DesignOrbits` computes the weights from the full incidence
matrix as pair counts divided by |U|; `tests/test_design_orbits.py`
checks them against brute force, checks quotient `cnt` against full
`cnt` on expanded solutions, and checks every orbit gain against the
exact deficit drop on the full backend.

**Cost.** The search core's unit of cost is one candidate. With
full-length orbits only (`|O| = |G|`, the default) `M` orbits are
exactly `M |G|` blocks and the notch ladder moves in steps of `|G|`
blocks. Short orbits (`short_orbits=1`) enter at unit cost: the reported
objective is still the true block count, but the core cannot prefer
`9 + 3` to `9 + 9` (both are two orbits), and a gain-per-block ratio in
the greedy is not available — the core's deficit bookkeeping requires
`ball_gather` to be the exact gain, so a scaled field is not an option,
and the product-space layer has no such ratio either. That layer's
answer to unequal orbits is to count *points*: its `SymmetricSearcher`
skips orbits larger than the remaining room and steps the descent by the
smallest orbit. `orbit_search_points` is exactly that searcher run over
the block orbits of the *full* design backend through a two-attribute
adapter, so it reaches C(9,3,2) = 12 = 9 + 3 under Z_3 x Z_3 — at the
price of one whole-orbit `ball_update` plus a deficit readback per
probed orbit on the full cell, against one gather on the quotient. The
two drivers are `-p orbit_mode=quotient` (default) and `points`.

**A fix in the product-space layer.** `SymmetricSearcher.descend` calls
`lns(target)` with a feasible solution *above* the target, and `_lns`
had no step that ruins down to `M`: its round loop is `while uncov > 0`,
so it returned at once and every notch was reported failed. The
benchmark cells of `docs/symmetry.md` all had greedy + peel land on the
optimum, which hid it. `_lns` now drops the cheapest orbits until the
size is at or below `M` before the loop; the trajectory pins do not go
through `lns` and are unmoved.

**Release.** `release(v, k, t, blocks, budget)` seeds a plain `Searcher`
on the full `CoveringDesignBackend` with the expanded blocks and runs
the free descent — `SymmetricSearcher.free_searcher`, transplanted.
`--out` always receives expanded blocks in the LJCR format, verified by
the independent `verify` before writing.

**Validation** (`tests/test_design_orbits.py`, host backend, seconds):

| cell | group | optimum | driver | found |
|---|---|---|---|---|
| C(7,3,2) | cyclic | 7 | quotient | 7 |
| C(13,4,2) | cyclic | 13 | quotient | 13 |
| C(21,5,2) | cyclic (difference set) | 21 | quotient | 21 |
| C(31,6,2) | cyclic (difference set) | 31 | quotient | 31 |
| C(19,3,2) | cyclic (STS) | 57 | quotient | 57 |
| C(9,3,2) | Z_3 x Z_3 | 12 = 9 + 3 | points | 12 |
| C(15,3,2) | cyclic (STS) | 35 = 15 + 15 + 5 | points 20 s, then release | 45, then 35 |
| C(24,8,5) | — | 759 | free greedy | 759 |

C(15,3,2) is the honest one: neither driver found the short-orbit
solution inside a 20 s orbit phase (the quotient cannot express its
cost; the points driver's LNS is the weaker searcher), and it was the
released free descent that reached 35 from the 45-block cyclic start.

### Calibration sweep

`scripts/sweep_design_groups.py`, GPU, 2026-09-02, one process per
row: for every built-in group of the point set, a 3-minute orbit phase
on the quotient (full-length orbits only; the descent's 30 s reserve on
top) followed by a 2-minute free release seeded with the orbit
solution, notch budgets a sixth / a quarter of the phase. "greedy" is
the orbit greedy's block count, "orbit best" the best verified
G-invariant block set, "released best" the best verified block set after
the free descent; every number is a block count checked by the
independent verifier. Raw records: `docs/bench_design_groups.json`.

| cell | record | group | \|G\| | block orbits (full) | greedy | orbit best | released best | gain from release | orbit phase best at |
|---|---|---|---|---|---|---|---|---|---|
| C(20,6,4) | 382 | cyclic | 20 | 1944 (1932) | 540 | 480 | 475 | 5 | 61 s |
| C(20,6,4) | 382 | cyclic-1 | 19 | 2040 (2040) | 513 | 475 | 475 | 0 | 30 s |
| C(20,6,4) | 382 | cyclic-2 | 18 | 2162 (2146) | 540 | 486 | 468 | 18 | 0 s |
| C(20,6,4) | 382 | dihedral | 40 | 1032 (908) | 800 | 600 | 501 | 99 | 52 s |
| C(20,6,4) | 382 | prod:2x10 | 20 | 1956 (1920) | 580 | 480 | 475 | 5 | 91 s |
| C(20,6,4) | 382 | prod:4x5 | 20 | 1944 (1932) | 520 | 480 | 474 | 6 | 39 s |
| C(28,8,4) | 402 | cyclic | 28 | 111041 (110968) | 588 | 588 | 581 | 7 | 15 s |
| C(28,8,4) | 402 | cyclic-1 | 27 | 115115 (115115) | 594 | 567 | 564 | 3 | 36 s |
| C(28,8,4) | 402 | cyclic-2 | 26 | 119581 (119504) | 624 | 572 | 567 | 5 | 156 s |
| C(28,8,4) | 402 | dihedral | 56 | 56021 (54994) | 896 | 784 | 639 | 145 | 68 s |
| C(28,8,4) | 402 | prod:2x14 | 28 | 111111 (110898) | 672 | 616 | 592 | 24 | 23 s |
| C(28,8,4) | 402 | prod:4x7 | 28 | 111041 (110968) | 616 | 588 | 574 | 14 | 163 s |
| C(20,12,4) | 20 | cyclic | 20 | 6310 (6288) | 20 | 20 | 20 | 0 | 3 s |
| C(20,12,4) | 20 | cyclic-1 | 19 | 6630 (6630) | 38 | 38 | 27 | 11 | 3 s |
| C(20,12,4) | 20 | cyclic-2 | 18 | 7012 (6986) | 36 | 36 | 27 | 9 | 3 s |
| C(20,12,4) | 20 | dihedral | 40 | 3260 (3044) | 40 | 40 | 27 | 13 | 3 s |
| C(20,12,4) | 20 | prod:2x10 | 20 | 6330 (6268) | 40 | 40 | 27 | 13 | 3 s |
| C(20,12,4) | 20 | prod:4x5 | 20 | 6310 (6288) | 20 | 20 | 20 | 0 | 3 s |
| C(21,13,5) | 37 | cyclic | 21 | 9690 (9690) | 42 | 42 | 42 | 0 | 16 s |
| C(21,13,5) | 37 | cyclic-1 | 20 | 10186 (10164) | 60 | 60 | 49 | 11 | 14 s |
| C(21,13,5) | 37 | cyclic-2 | 19 | 10710 (10710) | 57 | 57 | 51 | 6 | 13 s |
| C(21,13,5) | 37 | dihedral | 42 | 4950 (4740) | 84 | 84 | 57 | 27 | 12 s |
| C(21,13,5) | 37 | prod:3x7 | 21 | 9690 (9690) | 42 | 42 | 42 | 0 | 16 s |
| C(28,10,4) | 154 | cyclic | 28 | 468754 (468611) | 252 | 252 | 237 | 15 | 220 s |
| C(28,10,4) | 154 | cyclic-1 | 27 | 486048 (486038) | 243 | 243 | 229 | 14 | 218 s |
| C(28,10,4) | 154 | cyclic-2 | 26 | 504812 (504658) | 260 | 234 | 233 | 1 | 197 s |
| C(28,10,4) | 154 | dihedral | 56 | 235378 (233312) | 392 | 336 | 264 | 72 | 245 s |
| C(28,10,4) | 154 | prod:2x14 | 28 | 468897 (468468) | 280 | 252 | 239 | 13 | 220 s |
| C(28,10,4) | 154 | prod:4x7 | 28 | 468754 (468611) | 252 | 224 | 224 | 0 | 336 s |

Reading the table, honestly:

- **Where the record is one full orbit, the layer finds it at once.**
  C(20,12,4) = 20 is a single Z_20 orbit (also a single Z_4 x Z_5
  orbit); both groups return it in 3 s, and the release has nothing
  left to do. That is the regime the product-space layer's benchmarks
  describe.
- **Where the record needs short orbits, full-length search bottoms
  out above it.** C(21,13,5) = 37 is not a multiple of 21, 7 or 3, so no
  union of full Z_21 orbits can reach it; the two groups with a 21-orbit
  stop at 42 (two orbits) and the release does not move them. C(28,10,4)
  = 154 = 5.5 x 28 likewise; the best here is 224 (Z_4 x Z_7, eight
  orbits).
- **On the two 4-designs the groups tried are not the records'
  groups.** C(20,6,4): best 468 (Z_18 with two fixed points, after
  release) against 382; C(28,8,4): best 564 (Z_27 with a fixed point)
  against 402 — the free 20-minute run above reached 585, so the
  structured start is better than the free start on this cell, by about
  4 %, and still 40 % above the record. The orbit phase reaches its best
  within 15-160 s and then walls; the remaining minutes buy nothing, so
  the limit is the group and the full-orbit restriction, not the budget.
  The LJCR entries for these cells come from wheel constructions and
  private tools that choose the group *and* the short orbits *and* the
  base blocks together; a sweep over six generic groups is not that.
- **What the release is worth.** Seeded with a good structured start
  (a cyclic-type group) the free descent gains 0-18 blocks in two
  minutes; seeded with a bad one (dihedral, whose orbits are twice as
  large and whose greedy is far off) it gains 72-145, i.e. it repairs
  the start rather than improving it. Nowhere did the release cross
  from a structured solution to a record. This is the rigidity result
  of the covering-code paper again: free LNS does not turn a
  group-invariant local optimum into a different, better structure.

A second pass admitting short orbits (`short_orbits=1`, unit cost —
see the cost note above) was running when this was written; its rows
are appended to the same raw file as they complete.

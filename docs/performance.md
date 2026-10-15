# Performance

Everything on this page is **measured**, paired (same cell, same seed,
same machine), and reproducible from a script in `scripts/`. Where an
idea did not pay, it is written down as not having paid — a negative
result that stops the next person re-implementing it is worth as much
as a speedup.

Measurement conditions for the 2026-08-25 CPU numbers: NVIDIA GH200
480GB host (aarch64, 64 cores), numpy 1.21.5, single-threaded, with
unrelated GPU jobs resident on the device. GPU numbers carry their own
date and conditions; the 2026-10-13 GPU numbers in sections 1 and 7 were
taken on an otherwise idle device (see section 7 for how that was
checked).

Contents:

1. [Where the time goes](#1-where-the-time-goes)
2. [The CPU backend](#2-the-cpu-backend)
3. [Search-core hot loops](#3-search-core-hot-loops)
4. [The orbit layer](#4-the-orbit-layer)
5. [What did not pay](#5-what-did-not-pay)
6. [GPU-side roadmap](#6-gpu-side-roadmap)
7. [The GPU backend in v0.3](#7-the-gpu-backend-in-v03)

---

## 1. Where the time goes

A representative campaign run — LNS ruin/recreate rounds at a fixed M,
then a shell-neighbourhood LAHC polish — profiled with `cProfile` on
the CPU backend (`scripts/profile_search.py --backend cpu`):

```
cell K5(8,3) |X|=390,625  M=395  25 LNS rounds + 10 s shell polish
    ncalls  tottime  cumtime  function
      6901    4.262    4.750  dp._balls                (ball offsets)
     23591    1.221    1.221  numpy ufunc reduce
      3717    1.086    7.004  dp.ball_gather
      1589    0.648    0.648  protocol._critical       (whole-space role mask)
       240    0.517    0.934  dp._axis_hamming         (the transform)
         9    0.360    0.360  search.py:272 <listcomp> (greedy heap build)
```

The shape is the finding: **on the CPU backend the search core's own
Python is a rounding error** (the heap build is the only search-core
line in the top ten, at 3.6%), and 70% of the clock is exact ball
gathers — `_balls` plus the mask it indexes. The optimisations below
follow that, not intuition.

On the **GPU** backend the first profile of this page (K_3(8,2), with an
unrelated tenant holding the device at 100% utilisation) measured about
4 ms per `ball_update` / `ball_gather` call. That was queueing behind the
other tenant, not the plugin. Retaken on an idle device (2026-10-13,
v0.2.2 plugin, `scripts/profile_gpu_ops.py`, 10 s shell-LAHC polish from
a saved greedy cover minus 2 words, share of polish wall time):

| cell | \|X\| | polish mv/s | ball_gather | ball_update | transforms | search Python |
|---|---|---|---|---|---|---|
| K_5(8,3) | 3.9e5 | 3,694 | 29% (40 µs/call) | 13% | <1% | 55% |
| K_8(9,4) | 1.3e8 | 817 | 54% (328 µs/call) | 7% | 12% (16.7 ms each) | 25% |
| K_10(9,5) | 1.0e9 | 25 | 87% (17.2 ms/call) | 1% | 10% (213 ms each) | 1% |

and the exact greedy build from the empty code at K_8(9,4) took 15.0 s,
of which 3.5 s were gathers, 2.2 s gain transforms and 8.8 s the search
core's heap handling. So the GPU picture depends on the cell: at
K_5(8,3) the per-call latency floor (about 20 µs for any call that
returns data) and the search core's Python share the time; at K_8(9,4)
and K_10(9,5) the support-walk gathers dominate, and the transform is
second. Section 7 is the v0.3 work that followed from this profile.

---

## 2. The CPU backend

`coldcore.dp.NumpyDPBackend` runs the CUDA plugin's own axis DP in
numpy: `O(n |X| (R+1))` instead of the reference's `O(|X|²)`. Full
tables and the end-to-end comparison are in
[benchmarks.md](benchmarks.md#cpu-backend-brute-force-vs-the-axis-dp);
the headline, on the gain transform:

| cell | \|X\| | ref | cpu | speedup |
|---|---|---|---|---|
| K_3(8,2) | 6,561 | 400 ms | 0.48 ms | 831× |
| K_2(12,2) | 4,096 | 238 ms | 0.59 ms | 404× |
| K_2(16,3) | 65,536 | 2,816 ms | 13.3 ms | 212× |

and cells the reference refuses outright (K_3(12,3), K_5(8,3),
K_7(8,3), torus 64³) now run at 1.7–460 ms per transform.

Reproduce: `python3 scripts/bench_backends.py --json docs/bench_backends.json`

### mask vs. values in `ball_gather`

`ball_gather` needs the coverage *role* (deficient / critical) of every
word in each candidate's ball. Two exact ways to get it:

* build the whole-space role mask (one `O(|X|)` pass) and gather
  **bytes** from it;
* gather the `cnt` **values** at the ball and apply the predicate there
  — no `O(|X|)` pass, but a `dtype`-sized gather.

Measured on this machine (|X| = 390,625, ball 4,065, batch 216): the
byte gather runs at ~0.85 ns/element against ~2.8 ns for the value
gather, and the mask pass costs ~1.0 ns/element of |X|. So the mask
pays only once the batch touches more than about half the space —
and the polish's inner loop is a **single-word** loss query, which
touches one ball. Scanning 390k words to answer a question about 4k of
them was pure waste.

Picking the cheaper path per call (`_MASK_BREAKEVEN`, identical answers
either way, gated by `tests/test_dp_backend.py::test_both_gather_paths_agree`):

| cell | mask always | cost-adaptive | gain |
|---|---|---|---|
| K_5(8,3) | 227.0 mv/s | 259.5 mv/s | +14.3% |
| K_7(8,3) | 40.4 mv/s | 44.0 mv/s | +8.9% |
| K_3(8,2) | 2,078 mv/s | 2,200 mv/s | +5.9% |
| K_3(12,3) | 284.8 mv/s | 300.5 mv/s | +5.5% |

(`mv/s` = shell-polish moves per second, 8 s of polish, same seed, same
move semantics — the move counts differ only because more moves fit in
the same wall clock.)

---

## 3. Search-core hot loops

Two changes, both provably decision-neutral:

* **greedy heap build.** `greedy_fill` turned a threshold extraction of
  up to 2¹⁸ map entries into a heap with a Python loop doing numpy
  *scalar* indexing per element — 0.36 s of a 9.9 s run, the only
  search-core line in the profile's top ten. One `.tolist()` per array
  and a `zip` produce the identical list of tuples in the identical
  order.
* **`_shell1`.** The polish's distance-1 shell was rebuilt from
  `np.arange` per axis on *every move*. The per-axis displacement
  tables are a function of the space alone, so they are built once.

Neither may move a decision, and that is enforced rather than asserted:
`tests/test_search_determinism.py` freezes SHA-256 digests of whole
move traces — greedy/peel/every ruin operator, the shell polish run to
feasibility, the two-swap endgame, and the orbit layer — taken from the
pre-optimisation core at commit `1710fb7`. The pass sequences are
deliberately **clock-free** (fixed round counts; a polish that
terminates on `uncov == 0` rather than on a budget), because a
time-budgeted pass runs *more* rounds on a faster machine and its trace
is then a stopwatch reading, not a decision record.

---

## 4. The orbit layer

`SymmetricSearcher`'s greedy probes an orbit by adding it, reading the
exact deficit back, and — usually — dropping it again. On the
`bench_symmetry` cells plus larger many-orbit cells
(`scripts/bench_orbit.py`, clock-free: partition + greedy + peel + 30
fixed ruin/recreate rounds), that probe was the whole cost.

Two changes, both exactly value-preserving:

* **the code array is materialised on demand.** A probe used to splice
  the point list twice (concatenate+sort on the way in, set-difference
  on the way out) although nothing between the two reads it — only its
  length, which is now tracked as an integer.
* **`_undo_add` / `_undo_drop`.** `ball_update` is its own exact
  inverse, so the deficit after undoing a probe is *known*: it is the
  value from before, and the probe already measured the difference.
  Reading it back off the field was a whole-space scan to learn a
  number already in hand.

Measured (best of 3, `--rounds 30`, CPU backend, identical move traces —
pinned by `tests/test_search_determinism.py`'s orbit digests):

| cell | orbits | before | after | speedup |
|---|---|---|---|---|
| K_2(14,3) / shift | 8,192 | 17,632 ms | 14,912 ms | 1.18× |
| K_5(6,2) / shift | 3,125 | 4,719 ms | 3,715 ms | **1.27×** |
| K_3(8,2) / shift | 2,187 | 2,468 ms | 2,091 ms | 1.18× |
| K_3(8,2) / diag | 2,187 | 1,307 ms | 1,099 ms | 1.19× |
| lee 11³ / translate | 121 | 113.4 ms | 92.6 ms | 1.22× |
| K_4(4,1) | 64 | 31.0 ms | 25.9 ms | 1.20× |
| K_2^3(8,1) (µ=3) | 64 | 32.2 ms | 28.3 ms | 1.14× |
| K_3(5,1) | 81 | 24.3 ms | 20.8 ms | 1.17× |
| K_2(6,1) | 32 | 22.8 ms | 19.9 ms | 1.15× |
| K_2^2(6,1) (µ=2) | 16 | 12.8 ms | 11.3 ms | 1.13× |
| torus 27×27 | 81 | 10.9 ms | 9.0 ms | 1.21× |
| K_4^2(4,1) (µ=2) | 16 | 9.0 ms | 7.6 ms | 1.18× |
| K_3(4,1) | 27 | 7.2 ms | 6.1 ms | 1.18× |
| torus 11×11 | 11 | 2.50 ms | 2.08 ms | 1.20× |
| king 9×9 | 9 | 0.44 ms | 0.41 ms | 1.07× |
| lee 5×5 | 5 | 0.39 ms | 0.38 ms | 1.03× |
| **all 16 cells** | | **26,393 ms** | **22,041 ms** | **1.20×** |

The two cells that barely move are the ones whose greedy accepts every
orbit on the first pass (1 probe): there is no probe overhead to remove.

Reproduce: `python3 scripts/bench_orbit.py --json docs/bench_orbit.json`

---

## 5. What did not pay

* **A symmetric prefilter for `ball_gather`.** Coverage is symmetric,
  so a candidate has zero gain unless it lies in some deficient word's
  ball; when few words are deficient, walking *those* balls settles the
  whole batch. Implemented, measured, **removed**: in the shell polish
  the gather happens *after* the incumbent has been dropped, so the
  deficient set is not `uncov` (≈ 5) but `uncov + loss(w)` (≈ 200) —
  the same order as the candidate batch. Measured difference on
  K_5(8,3), K_3(12,3), K_3(8,2): none, in either direction.
* **Block-structured ball offsets.** The ball is a union of blocks that
  are cartesian products of per-position delta sets, so the offsets are
  an outer sum and the random gathers collapse by ~16×. Prototyped:
  1,968 µs → 1,378 µs (1.4×), not the 3× the gather count suggested,
  because the cost is the `(candidates × ball)` output array's memory
  traffic and not the gathers. Not worth the complexity; noted here so
  the next person does not re-derive it.
* **int32 contribution tables.** Accumulating ball offsets in int32 and
  widening once measured 1,968 µs → 1,979 µs in one run and
  2,030 → 1,259 µs in another — i.e. inside the noise of a machine with
  other tenants. Not taken.
* **Vectorising the orbit-layer field folds** (`gain_bounds` /
  `loss_bounds` / the greedy heap build over orbits). Done — they are
  cleaner and allocate less — but measured on their own the total over
  the bench cells moved from 25.66 s to 25.98 s, i.e. not at all: at
  |X| ≤ 2¹⁴ those folds are dwarfed by the transform they follow. The
  1.20× in section 4 came entirely from cutting *calls* (the lazy code
  array and the arithmetic undo), not from vectorising them.
* **Search-core micro-allocations on the GPU path.** The obvious
  candidates — the per-move one-element `np.array` for `ball_update`,
  `np.isin` against the incumbent, the LAHC list bookkeeping — are all
  below the noise floor: the GPU profile above puts *everything* numpy
  does at 0.011 s of 6.55 s. Preallocating scratch there would be
  optimising 0.2% of the run.

---

## 6. GPU-side roadmap

This list predates the v0.3 GPU work (section 7). Status after it:

1. **Persistent-CTA transforms / fusing axis passes.** Done for HBM
   cells in a different form: two equal-size adjacent axes per kernel
   through shared memory, no layer fills, zero layers skipped (section
   7). Not measured on LPDDR-resident cells.
2. **A gather that does not re-derive `contrib` per block.** Not done.
   For the deficient role the walk is now usually replaced by the
   set-side match (section 7); the walk remains for role 1 and for
   large deficient sets, where it got 1.4-3.6x faster from the support
   table order alone.
3. **`map_extract` without the global atomic.** Done: one atomic per
   warp, and once the output is full the remaining hits are only
   counted. `thr = 1` at K_10(9,5): 24.6 ms -> 2.7 ms.
4. **µ-fold `ball_gather` fusion.** Unchanged (every shipped plugin has
   the fused kernel).
5. **The owner-trick loss pass** remains an honest negative in the
   dense covering regime (17 ms → 28 ms at K_8(9,4), 145 ms → 669 ms at
   K_8(10,4)); it is worth revisiting only for problems with
   `M·|ball| ≪ n·R·q^n`.

---

## 7. The GPU backend in v0.3

Measured 2026-10-13/14 on the GH200 (one H100-class GPU, 96 GB HBM3,
CUDA 12.8), v0.2.2 plugin and binding (`main`) against branch
`v0.3-gpu`, each pair back to back in one hold of the shared GPU lock. A
run started only after `nvidia-smi` reported at most 5% utilisation on
three consecutive one-second samples inside the lock; during each run
the SM share of every process on the device was sampled once a second,
and runs during which any other process used the device were discarded
and repeated. Every run starts from the same saved state (a greedy
cover; the polish removes 2 words from it, 4 at K_5(8,3), and the per-
call numbers use the cover minus 2 words), same seed. Polish throughput
depends on the trajectory (the GPU's extraction order is unspecified, so
two runs diverge after the first pool refresh, and accepted moves cost
more than rejected ones), so the polish rows below are three alternating
pairs and their medians.

### What changed

* **Calls.** One synchronisation per call that returns data, through
  pinned staging buffers; `ball_update` of up to 1,024 words returns
  without waiting for the device (the words travel as kernel
  parameters). Grids of whole-field passes are sized to the field.
* **Support-table order.** The binding sorts the table by unwrapped
  offset, so a warp's 32 lanes touch neighbouring `cnt` words. The walks
  are order-free, so results do not change; walk gathers got 1.2-1.7x
  faster at K_8(9,4) and 2.6-3.6x at K_10(9,5), the update walk 1.6x.
* **Set-side gather.** A deficient-role gather of nb candidates walks
  nb balls (nb·|B| random `cnt` reads). Coverage is the metric ball, so
  the same count is the number of deficient words within distance R of
  the candidate. The plugin lists the deficient words in one pass over
  `cnt` (cached until `cnt` changes, and produced as a side effect of
  `count_eq(0)`) and compares digits. It proves at init that the
  support table is the metric ball, and uses the list only while it has
  at most 2|B| words and the walk would read at least
  max(|X|/8, 2^21) counters.
* **Transform.** No layer fills (after axis pass j only layers
  0..min(j, R) can be nonzero), the gain/loss indicator computed inside
  the first pass, exact-Q kernels for all Q <= 10, R <= 6 (the v0.2
  fallback kernel, used for every (Q, R) without a template, kept its
  arrays in local memory; the K_3..K_6 transforms are 8-10x faster
  now), and two equal adjacent axes per kernel through shared memory.
* **Reductions and extraction.** 16-byte loads; one atomic per warp for
  extraction slots, and only a count once the output is full.
* **New optional operations** (plugin ABI unchanged, feature-tested):
  `swap_eval` (removal loss of k drop words and the post-removal gain of
  their candidates, read-only, one round trip), `ball_update_delta`
  (an update that returns the exact deficit change it caused).

### End to end

Shell-LAHC polish, 10 s (K_5: 6 s) from the saved cover, moves per
second, three alternating pairs (v0.2.2, v0.3, v0.2.2, ...):

| cell | \|X\| | v0.2.2 | v0.3 | median ratio |
|---|---|---|---|---|
| K_5(8,3) | 3.9e5 | 3,824 / 3,296 / 3,711 | 4,008 / 3,882 / 4,013 | 1.08x |
| K_8(9,4) | 1.3e8 | 777 / 824 / 798 | 1,988 / 2,054 / 2,051 | 2.57x |
| K_10(9,5) | 1.0e9 | 26 / 25 / 25 | 469 / 482 / 452 | 18.4x |

One pair each, 5 s polish (these cells are Python-bound; run-to-run
spread on this host is up to 15%):

| cell | v0.2.2 | v0.3 | gain transform v0.2.2 -> v0.3 |
|---|---|---|---|
| K_3(8,1) | 2,819 | 3,453 | 133 -> 16 µs |
| K_4(8,2) | 4,214 | 4,303 | 155 -> 18 µs |
| K_6(8,3) | 3,371 | 3,670 | 534 -> 51 µs |
| K_8(8,4) | 1,731 | 3,398 | 1,987 -> 627 µs |

With the polish calling `swap_eval` (the patch at the end of this
section; search decisions unchanged), two alternating pairs:

| cell | v0.3 | v0.3 + `swap_eval` | ratio |
|---|---|---|---|
| K_5(8,3) | 3,671 / 4,087 | 4,077 / 4,022 | 1.04x |
| K_8(9,4) | 2,150 / 1,969 | 2,383 / 2,572 | 1.20x |
| K_10(9,5) | 481 / 504 | 724 / 758 | 1.50x |

Exact greedy build from the empty code (same seed):

| cell | v0.2.2 | v0.3 |
|---|---|---|
| K_8(9,4) | 14.4 s (M = 2,758) | 11.7 s (M = 2,759) |
| K_10(9,5) | 713 words placed after 90 s, 154,035 words still uncovered | 34.5 s to the full cover (M = 1,056) |

At K_8(9,4) the build's gathers went from 290 to 224 µs and its
transforms from 16.5 to 6.1 ms, but 8.1-8.4 s of the 11.7-14.4 s are the
search core's heap handling, which the GPU does not touch.

### Per operation

Median µs per call at the saved cover minus 2 words. "after a change"
rows run the two-word update first, as the polish does; the plain
repeated rows see an unchanged `cnt`, which v0.3 answers from its cached
deficient-word list.

| operation | K_8(9,4) v0.2.2 | v0.3 | K_10(9,5) v0.2.2 | v0.3 |
|---|---|---|---|---|
| update pair (issue only) | 40.7 | 13.9 | 462 | 13.8 |
| update pair + 1-word read (completed) | 61.9 | 46.7 | 488 | 302 |
| gather role 1, 1 word | 29.8 | 25.0 | 211 | 125 |
| update pair + gather role 0, 64 words | 359 | 165 | 9,717 | 977 |
| update pair + `uncovered()` | 154 | 138 | 1,226 | 949 |
| gather role 0, 1,024 words, repeated | 4,170 | 25 | 192,401 | 27 |
| `gain_map` (transform) | 16,494 | 6,110 | 213,152 | 61,094 |
| `map_hist(1024)` | 348 | 277 | 2,540 | 2,114 |
| `map_extract(thr=max)` | 439 | 185 | 1,955 | 1,171 |
| `map_extract(thr=1, cap 2^18)` | 2,449 | 588 | 24,578 | 2,721 |

Device cost of one polish move (`scripts/bench_move.py`, 110
candidates, median µs): the four-call sequence (loss gather, removal,
candidate gather, re-insertion), the same as one `swap_eval`, and an
accepted move's state change with its check:

| cell | sequence v0.2.2 | sequence v0.3 | `swap_eval` v0.3 | accept + `uncovered()` v0.2.2 / v0.3 | accept + `ball_update_delta` v0.3 |
|---|---|---|---|---|---|
| K_5(8,3) | 82 | 84 | 68 | 75 / 43 | 90 |
| K_8(9,4) | 556 | 198 | 66 | 193 / 160 | 89 |
| K_10(9,5) | 17,347 | 1,112 | 169 | 1,648 / 1,243 | 649 |

Transform, median ms (gain map; recount and loss map within 4%):

| cell | v0.2.2 | v0.3, one axis per kernel | v0.3, two axes per kernel | bytes moved, v0.3 |
|---|---|---|---|---|
| K_8(9,4) | 16.6 | 12.0 | 6.1 | 18.5 GB, 3.0 TB/s |
| K_10(9,5) | 213.0 | 117.6 | 61.0 | 154 GB, 2.5 TB/s |

(v0.2.2 moves 44.8 GB at K_8(9,4) and 406 GB at K_10(9,5): R layer
fills, the indicator init, then R+1 planes read and R written per axis.)

### Attribution at K_8(9,4)

Same polish, one variable at a time (environment switches on one build,
paired runs): v0.2.2 about 800 mv/s; v0.3 with the builder's table order and
the set side off 915 (call overhead, transforms, reductions); plus the
locality order 1,155; plus the set side 1,731 (`84686f8`); the later
listing, extraction and transform changes 2,050. At K_10(9,5): v0.2.2
25; locality order, set side off 88; set side on 447; final 469.

### What did not pay, or paid less than expected

* **Fewer synchronisations alone.** Pinned staging and one wait per
  call lowered the floor of a data-returning call from about 21 µs to
  16-19 µs. The floor is the host side (Python, ctypes, launch), not the
  copies. What helped the small cells was not waiting at all
  (asynchronous updates) and the transform kernels.
* **The first two-axis kernel** issued one global load per loop
  iteration and staged the linear (stride-1) pair with a 32-way shared
  bank conflict: 4-11 ms per pair at K_8(9,4) against 1.4 ms per
  single-axis pass. All loads in flight plus a one-word pad fixed both.
  Loading one layer at a time (fewer registers) was then slower again
  (1.41 -> 1.86 ms per pair). These three were timed with other jobs on
  the device, so they are indicative only.
* **Pair tile shape at Q*R >= 50** (K_10(9,5), locked, idle): 32
  columns with two resident blocks 61 ms per transform; 16 columns 76
  ms; 16 columns with four blocks 79 ms; 8 columns 97 ms. The K_10
  pair kernel still moves its bytes at about 2.5 TB/s against 3.0 TB/s
  at K_8(9,4).
* **A large deficient set on the set side.** With the list allowed up
  to 32|B| words the K_8(9,4) greedy build's mean gather rose from 236
  µs (walk only) to 411 µs; listing hundreds of thousands of words costs
  more than walking 64 balls. Measured ratios 1/4/8/32: 212/226/256/411
  µs, hence the 2|B| default.
* **Four loads in flight in the listing kernel** (as in extraction):
  10% faster for a few thousand deficient words, 4-5x slower above
  1e5 (1,251 vs 297 µs at |Z| = 1.2e5, 4,259 vs 867 µs at 6.2e5).
  Reverted.
* **`count_eq(0)` as a listing pass** before the per-warp vote cost
  163 µs against 120 µs for the plain count at K_8(9,4); with the vote
  it is 100-120 µs and leaves the list for the next gather.
* **Persistent `map_extract` buffers for every cap** made large
  extractions slower (a copy of the found entries: 172 vs 122 µs for
  2e5 entries); now only caps of 2^20 and up reuse buffers.
* **`ball_update_delta` on small cells.** Four synchronised updates (90
  µs at K_5(8,3)) cost more than two asynchronous updates and an
  `uncovered()` (43 µs). It pays from K_8(9,4) up.
* **`swap_eval` on small cells.** At K_5(8,3) a move is 68 µs as one
  call against 84 µs as the four-call sequence; the polish there is
  host-bound (search Python 55-60% of the wall clock), so the patch
  gains 4% end to end.
* **The greedy build** at K_8(9,4) is 58-69% search-core Python (heap
  handling of up to 2^18 extracted entries per refresh, 64 heap pops
  and pushes around each of ~11,300 batch gathers); its gathers got
  1.3x and its transforms 2.7x faster, the build 1.2x.
* **Not changed:** the Lee, torus and grid transforms (still the v0.2
  kernels with layer fills), the explicit-incidence plugin (ABI 2), the
  µ-fold literal-sum fallback, and the support walk's per-block
  `build_contrib`.

### For the search core

`swap_eval` replaces the polish's four calls per move (loss gather,
removal, candidate gather, re-insertion) by one read-only call; an
accepted move then issues the two updates, which do not wait for the
device. The patch is 15 lines in `_shift_anneal`
(`coldcore.gpu.swap_eval_generic` keeps the CPU backends on the same
code path); with it the trace-pinning tests in
`tests/test_search_determinism.py` pass unchanged, i.e. every decision
is the same. Measured effect in the table above.

---

## Reproducing

```bash
python3 scripts/bench_backends.py --json docs/bench_backends.json  # §2
python3 scripts/profile_search.py --backend cpu --cell 'K5(8,3)'   # §1
python3 scripts/bench_orbit.py    --json docs/bench_orbit.json     # §4
python3 -m pytest tests/test_search_determinism.py                 # §3 gate
# §1 GPU rows and §7 (hold the shared-GPU lock; states/*.npy are saved
# covers, see --save-code):
python3 scripts/profile_gpu_ops.py --cell 'K8(9,4)' --save-code s.npy --polish-s 0
python3 scripts/profile_gpu_ops.py --cell 'K8(9,4)' --load-code s.npy --polish-s 10
python3 scripts/bench_move.py      --cell 'K8(9,4)' --load-code s.npy
python3 scripts/bench_transform.py --cell 'K8(9,4)' --load-code s.npy
python3 scripts/parity_vs_baseline.py --base /path/to/v0.2.2/libcoldcore_covering.so
```

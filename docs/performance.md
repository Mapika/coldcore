# Performance

Everything on this page is **measured**, paired (same cell, same seed,
same machine), and reproducible from a script in `scripts/`. Where an
idea did not pay, it is written down as not having paid — a negative
result that stops the next person re-implementing it is worth as much
as a speedup.

Measurement conditions for the 2026-08-25 CPU numbers: NVIDIA GH200
480GB host (aarch64, 64 cores), numpy 1.21.5, single-threaded, with
unrelated GPU jobs resident on the device. GPU numbers carry their own
date and conditions.

Contents:

1. [Where the time goes](#1-where-the-time-goes)
2. [The CPU backend](#2-the-cpu-backend)
3. [Search-core hot loops](#3-search-core-hot-loops)
4. [The orbit layer](#4-the-orbit-layer)
5. [What did not pay](#5-what-did-not-pay)
6. [GPU-side roadmap](#6-gpu-side-roadmap)

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

The same workload on the **GPU** backend, at K_3(8,2)
(`scripts/profile_search.py --backend gpu`), says the same thing more
bluntly:

```
6.55 s wall, 332 polish moves (55.3 mv/s)
    ncalls  tottime  function
       692    2.868   gpu.ball_update      (4.1 ms/call)
       705    2.835   gpu.ball_gather      (4.0 ms/call)
       105    0.438   gpu.uncovered        (4.2 ms/call)
     11411    0.011   numpy dispatch       (everything the search core does)
```

99% of the clock is inside the ctypes calls and **2% is the search
core**, so there is no Python lever on this path either.

> **Caveat, stated because it changes what the numbers mean.** These GPU
> figures were taken with an unrelated tenant holding the device at 100%
> utilisation, so ~4 ms per call is queueing, not launch overhead (a
> launch is tens of microseconds on an idle device). On this cell the
> CPU backend runs the same polish at 2,200 mv/s — 40× the *contended*
> GPU — which says nothing about the GPU and everything about sharing
> it. Re-take these on an idle device before drawing any conclusion
> about the plugin.

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

CUDA kernels were deliberately **not** touched in this pass (the device
was fully occupied by an unrelated tenant, and a kernel change measured
under contention is not a measurement). The opportunities below are
carried over from [benchmarks.md](benchmarks.md) and the covering
paper's Appendix A, in the order their evidence supports:

1. **Persistent-CTA transforms for LPDDR-resident cells.** Pinned-ATS
   streaming runs at ~72 GB/s effective and is insensitive to grid
   size, which says the axis passes are bandwidth-bound and latency
   is hidden — but the *layer* layout still re-streams every plane per
   axis. Fusing the last two axis passes would remove one full pass
   over `(R+1)·|X|` words. Measure against the K_8(10,4) 0.143 s HBM
   baseline and the K_10(10,5) 73–110 s LPDDR baseline.
2. **A gather that does not re-derive `contrib` per block.** The
   split-block geometry already gave 29–37× on single-word updates;
   the remaining per-block cost is `build_contrib`, which is identical
   for every block working on the same word. Shared-memory reuse across
   the split is a pure win if the block count per word is > 1.
3. **`map_extract` without the global atomic.** The extraction kernel
   serialises on one `atomicAdd` slot counter; a two-pass
   count-then-scatter (or per-block sub-allocation) is the standard fix
   and matters at `thr = 1`, which is what the polish's pool refresh
   uses.
4. **µ-fold `ball_gather` fusion.** With `mu > 1` and a plugin lacking
   `dct_ball_gather_mu`, the binding sums `mu` separate literal-level
   gathers. Every shipped plugin has the fused kernel; the fallback is
   there for old builds and should be measured before anyone relies on
   it for a µ-fold campaign.
5. **The owner-trick loss pass** remains an honest negative in the
   dense covering regime (17 ms → 28 ms at K_8(9,4), 145 ms → 669 ms at
   K_8(10,4)); it is worth revisiting only for problems with
   `M·|ball| ≪ n·R·q^n`.

---

## Reproducing

```bash
python3 scripts/bench_backends.py --json docs/bench_backends.json  # §2
python3 scripts/profile_search.py --backend cpu --cell 'K5(8,3)'   # §1
python3 scripts/bench_orbit.py    --json docs/bench_orbit.json     # §4
python3 -m pytest tests/test_search_determinism.py                 # §3 gate
```

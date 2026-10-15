# Benchmarks

Measured **2026-08-22** on an idle NVIDIA **GH200 480GB** (96 GB HBM3 +
480 GB LPDDR5X, cache-coherent C2C/ATS; aarch64, CUDA 12.8, sm_90).
All comparisons are **paired**: same cell, same seed, run sequentially on
an otherwise idle machine. Cells are covering-code instances K_q(n,R)
(`q^n` words, radius R); M is the solution size during the measurement.

## Reproduction-critical caveat: LPDDR requires `mbind` pinning

**All LPDDR ("ATS") numbers below are valid only with the allocation
pinned to the CPU NUMA node** — `posix_memalign` followed by
`mbind(MPOL_BIND, node0)`, as done by the engine's allocator since
2026-08-22.

Without the pin, GH200's HMM access counters silently **migrate GPU-hot
pages into HBM**. The failure mode is subtle in both directions:

- **small cells**: the working set migrates entirely, so an "LPDDR"
  benchmark quietly measures HBM and reports impossibly good numbers;
- **large cells** (bigger than HBM): migration fills HBM and the
  transform thrashes, orders of magnitude slower than honest LPDDR
  streaming.

With the pin in place, HBM usage stays flat during ATS transforms and
results are bit-identical to the unpinned build. If you re-run these
benchmarks on unpinned allocations, the placement columns are
meaningless.

## Support-walk kernels: split-block geometry vs. one-block-per-word

Incremental ball updates and exact per-candidate gathers, before and
after switching the launch geometry from one block per word to
split-block (many blocks cooperating on one large ball).

| cell | ball size | single-word update | gather ×64 | gather ×1024 |
|---|---|---|---|---|
| K_8(9,4), M=940 | 333,166 | 0.798 → 0.027 ms (**29.6×**) | 0.99 → 0.23 ms (4.3×) | 5.3 → 4.1 ms |
| K_8(10,4), M=11,776 | 547,646 | 1.335 → 0.036 ms (**37×**) | 1.68 → 0.45 ms (3.7×) | 11.3 → 9.1 ms |

## Support-table formats: parity

The v1 packed-byte pattern table vs. the optional "ox" format
(position mask + delta nibbles): **parity**. Single-word updates are
0.027 / 0.036 ms on both formats for the two cells above; gather ×1024
is 9.1 ms (v1) vs. 10.0 ms (ox) on K_8(10,4). v1 remains the default.

## Full-space transforms by memory placement

One full coverage transform (solution set → exact multiplicity field
over all `q^n` words), per placement of the layer arrays.

| cell | q^n | arrays | HBM | LPDDR, ATS + mbind | LPDDR, managed |
|---|---|---|---|---|---|
| K_8(10,4) | 1.1·10⁹ | 24 GB | 0.143 s | 3.4 s (23.7×) | 8.7 s (60×) |
| K_9(10,5) | 3.5·10⁹ | 91 GB | (exceeds HBM pairing) | 22–26 s | — |
| K_10(10,5) | 1.0·10¹⁰ | 260 GB | impossible | **73–110 s** | — |

The K_10(10,5) row is the first exact whole-space transform over a
10¹⁰-word cell on this machine: 260 GB of dense layer arrays streamed
from pinned LPDDR over C2C, initialization 10.7 s. Pinned-ATS streaming
runs at roughly 72 GB/s effective and is insensitive to launch grid
size; managed memory with preferred-location CPU is consistently the
slowest coherent mode (~30× HBM under contention vs. ~19× for
ATS + mbind), which is why pinned ATS is the committed host-memory mode.

The refactored plugin library in pure-HBM mode matches the pre-refactor
baseline exactly (0.143 s on both): the plugin seam costs nothing.

## Owner-trick loss pass: an honest negative

The "owner trick" (tag each covered word with its unique owner during
the transform, then compute all losses in one scan) **loses** to the
plain loss transform in the dense regime that covering codes live in:

| cell | transform loss | owner loss |
|---|---|---|
| K_8(9,4) | 17 ms | 28 ms |
| K_8(10,4) | 145 ms | 669 ms |

The owner pass wins only when `M·|ball| ≪ n·R·q^n`, i.e. for sparse
solutions. It remains available as an optional ABI capability
(`dct_loss_owner`) for problems in that regime.

## End-to-end search

Exact greedy construction of a full cover of K_8(9,4), old vs. new
kernels: **40.4 s → 28.7 s** (1.41×), with matching solution quality
(cover size 2758 vs. 2759).

## GPU backend: v0.2.2 vs. v0.3 (2026-10-13/14)

Idle GH200 (checked per run, see [performance.md](performance.md#7-the-gpu-backend-in-v03)),
v0.2.2 plugin and binding against branch `v0.3-gpu`, paired in one lock
hold, same saved state and seed. Results are bit-identical between the
two (`scripts/parity_vs_baseline.py`).

| measure | cell | v0.2.2 | v0.3 |
|---|---|---|---|
| shell-LAHC polish, mv/s (median of 3) | K_5(8,3) | 3,711 | 4,008 |
| | K_8(9,4) | 798 | 2,051 |
| | K_10(9,5) | 25 | 469 |
| gain transform, ms | K_8(9,4) | 16.6 | 6.1 |
| | K_10(9,5) | 213.0 | 61.0 |
| greedy build from empty | K_8(9,4) | 14.4 s | 11.7 s |
| | K_10(9,5) | 713 of 1,056 words after 90 s | 34.5 s, complete |
| one polish move, device side (µs) | K_8(9,4) | 556 | 198 (66 with `swap_eval`) |
| | K_10(9,5) | 17,347 | 1,112 (169 with `swap_eval`) |

Reproduce with `scripts/profile_gpu_ops.py`, `scripts/bench_transform.py`
and `scripts/bench_move.py` (commands at the end of performance.md).

## CPU backend: brute force vs. the axis DP

Measured **2026-08-25**, same machine, on CPU only (the GPU was busy
with an unrelated tenant; every number below is single-threaded numpy).
`ref` is the O(|X|²) brute-force reference; `cpu` is
`coldcore.dp.NumpyDPBackend`, which runs the plugin's own axis DP in
numpy. Both produce **bit-identical fields**, so this is purely a clock
comparison. Reproduce with `python3 scripts/bench_backends.py`
(raw records in `docs/bench_backends.json`).

| cell | \|X\| | gain transform, ref | gain transform, cpu | speedup |
|---|---|---|---|---|
| king15×15 (torus) | 225 | 0.83 ms | 0.04 ms | 20× |
| lee 9×9×9, R=2 | 729 | 9.2 ms | 0.12 ms | 74× |
| K_2(12,2) | 4,096 | 238 ms | 0.59 ms | **404×** |
| K^2_2(12,2) (μ=2) | 4,096 | 320 ms | 0.58 ms | **552×** |
| K_3(8,2) | 6,561 | 400 ms | 0.48 ms | **831×** |
| K_2(16,3) | 65,536 | 2,816 ms | 13.3 ms | **212×** |

The `recount` transform (from the solution set rather than from an
indicator) is only linear in M for brute force, so its gap is smaller
but still large: 395 ms → 13.4 ms at K_2(16,3) (30×).

Cells the reference **refuses outright** (`\|X\| > 2^16`), on the CPU
backend alone — one coverage transform and one gain transform:

| cell | \|X\| | recount | gain map | ball size |
|---|---|---|---|---|
| torus 64×64×64, R=2 | 262,144 | 1.7 ms | 1.6 ms | 125 |
| K_5(8,3) | 390,625 | 26.8 ms | 26.3 ms | 4,065 |
| K_3(12,3) | 531,441 | 73.5 ms | 73.1 ms | 2,049 |
| K_7(8,3) | 5,764,801 | 459 ms | 455 ms | 13,153 |

End to end (greedy build → peel → notch descent, 60 s budget, seed 1):
at K_2(16,3) the reference cannot even finish the greedy build inside
the budget, while the CPU backend completes the whole descent to
M = 256 in **3.1 s**. At K_3(12,3), K_5(8,3) and torus 64³ — all
impossible for the reference — the CPU backend descends to M = 991,
398 and 2,205 respectively inside the same budget.

The two small cells (\|X\| ≤ 729) are the honest floor: at that size the
brute force fits in cache and the DP's per-axis pass overhead is
comparable, so the win is only 20-74× rather than hundreds.

## Benchmarking rules for contributions

- **Paired runs only.** A vs. B on the same cells, same seeds, same
  wall-clock budget, same machine state; report both quality and
  throughput. No best-of-N.
- Report the exact commit, `nvcc --version`, driver, and GPU.
- Separate *kernel* benchmarks (transform / ball-walk throughput) from
  *search* benchmarks (time-to-target-M): kernel wins that do not move
  search outcomes are reported as such.
- On GH200-class hardware, state the allocator/pinning mode for every
  LPDDR number (see the caveat above).

# Reproducing the covering-code records

The framework's founding credential: **26 new best-known upper bounds**
on the K_q(n,R) covering-code tables (alongside 58 improved lower
bounds), produced in August 2026 on a single NVIDIA GH200 by the
upstream record-hunting engine from which coldcore is extracted.

- Paper: [arXiv:2608.19872](https://arxiv.org/abs/2608.19872) —
  *New upper and lower bounds on covering codes K_q(n,R) for alphabets
  of size 5 ≤ q ≤ 21*.
- Code files and verification tooling:
  [github.com/Mapika/coldcase](https://github.com/Mapika/coldcase).

## What "verified" means

Every claimed record passed an **independent verification gate**, not
the search engine's own bookkeeping:

1. the candidate code file (M rows of n digits) is re-read from disk;
2. coverage is recomputed exactly by an implementation independent of
   the transform engine;
3. only then is the result recorded. The search core never writes the
   records table.

coldcore preserves this separation: `Searcher.descend` fires an
`on_solved(M, code)` callback; verification gates hook there.

## Reproducing a cell with coldcore

```bash
cmake -B build && cmake --build build -j
python -m coldcore solve --backend gpu \
    -p q=7 -p n=8 -p R=3 \
    --out live_K7_8_3.txt --hours 2
```

Seeds, per-cell wall-clock budgets, and the exact engine revisions used
for the published records are documented in the upstream repository and
the paper's ancillary files.

## Scale notes

- Cells up to q^n ≈ 3.5·10⁹ run entirely in 96 GB HBM.
- Larger cells run host-coherent on GH200: transform-streamed layer
  arrays in pinned LPDDR (ATS/C2C), the random-access `cnt` field in HBM
  (automatic placement; `COLDCORE_HBM_BUDGET` overrides the budget).
  Measured placement numbers — including why the LPDDR pinning matters —
  are in [benchmarks.md](benchmarks.md).

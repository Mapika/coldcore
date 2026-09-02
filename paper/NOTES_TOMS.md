# NOTES: what a TOMS-weight version adds beyond the JOSS paper

Half-page outline for a future ACM Transactions on Mathematical Software
submission ("coldcore: exhaustive-coverage transforms for combinatorial
search on coherent-memory GPUs"). The JOSS paper announces the software;
the TOMS paper would carry the engineering science.

## 1. Formal ABI specification

- The v1 contract as a specification, not a header: state machine
  (init/set/transform/update lifecycle), field semantics (cnt/A0
  invariants after each op), error model, singleton rationale, v2
  handle/context reservation.
- Support-table formats (v1 packed bytes, ox mask+nibbles) as formal
  encodings with their capability bounds (ax_j <= 16, pattern weight <= 8)
  and proof that both enumerate the same coverage set.
- Conformance test suite as an artifact: what a plugin must pass
  (reference-parity gates, all three transform init modes, incremental
  vs. recount drift bounds).

## 2. Transform complexity and correctness analysis

- The separable axis-DP recurrence per plugin family; cost model
  O(L * n * |X|) with layer count L (R+1 for Hamming grade truncation,
  1 for windowed sums); grade-truncation correctness proof.
- Exactness of lazy greedy from submodularity of coverage gain: stale
  map values are upper bounds; proof that pop/re-gather/commit realizes
  the true argmax; conditions a plugin must satisfy to inherit this.
- Incremental ball updates vs. full transforms: when M*|ball| beats
  n*L*|X| (the measured owner-trick crossover generalized into a model).

## 3. Memory-placement model

- Analytic placement policy: streaming vs. random-access arrays,
  HBM budget, LPDDR coherent (ATS + mbind pinning) vs. managed; predicted
  vs. measured transform times across the placement matrix (0.143 s /
  3.4 s / 8.7 s on K8(10,4); 73-110 s at 10^10 words).
- The GH200 HMM page-migration pitfall as a reproducible measurement
  hazard: access-counter migration silently converts "LPDDR" benchmarks
  into HBM benchmarks; mbind(MPOL_BIND) methodology.
- Roofline-style analysis: effective ~72 GB/s C2C streaming vs. HBM
  bandwidth; where transforms are bandwidth-bound vs. launch-bound.
- Search economics at scale: transform-dominated regime at 10^10 words
  (8 LNS rounds / 8 h), implications for pass scheduling (gain-refresh
  thinning, gather-only endgames).

## 4. Multi-GPU and portability roadmap

- Axis-sliced decomposition of the transform across devices (all axes
  but one are embarrassingly parallel per slab; the swept axis needs a
  halo/all-to-all per pass); projected scaling for NVL72-class coherent
  domains.
- Portable fallbacks: managed-memory path on non-coherent devices
  (measured 60x), CPU reference backend as the semantics anchor.
- v2 ABI: context handles for multi-cell/multi-device residency,
  R=0 degenerate transforms, streamed extraction beyond extract_cap.

## 5. Expanded evaluation

- Full paired-benchmark matrix (kernel + search) across q, n, R, and
  placement; time-to-target-M distributions over seeds, not single runs.
- Third-problem case study (e.g. Lee-metric covering / quasi-perfect
  Lee codes) quantifying plugin-authoring effort (LOC, hours) as the
  reuse claim's evidence.

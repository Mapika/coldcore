# The Lee-metric plugin (`PROB_LEE`, problem id 2)

The third problem family, and the framework's problem-agnostic proof:
three distinct neighborhoods (Hamming, Chebyshev-torus, Lee) run through
one v1 ABI and one untouched search core.

## Problem

Covering codes on a mixed-radix product space X = prod_j Z_{ax_j} under
the **Lee metric**: per coordinate the distance is circular,

    d_j(a, b) = min(|a - b|, ax_j - |a - b|),

and the total distance is the **sum** over coordinates. It is the
sum-metric analogue of Hamming (which counts each differing coordinate
as 1); q = 2, 3 axes degenerate to Hamming per coordinate.

## Transform

The Hamming grade-truncated axis DP generalizes directly. State: layer
planes A_0..A_R, where after processing axes 0..j, A_d(x) = number of
S-elements at Lee distance exactly d from x on axes <= j and equal on
axes > j. Per axis of size Q, with W = floor(Q/2), the fiber recurrence
is a circulant kernel per grade:

    newA_d[a] = A_d[a] + sum_{w=1..min(d,W)} K_w(a)
    K_w(a)    = A_{d-w}[a-w] + A_{d-w}[a+w]      (indices mod Q)

with the two terms counted **once** iff Q is even and w = Q/2 (the
antipodal digit has multiplicity 1). Grades above R are truncated —
they can never re-enter the ball. After all n axes, coverage
(gain/loss/cnt) = sum_{d<=R} A_d, exactly as in the Hamming plugin.

Implementation: `src/plugins/covering/dctcore_lee.cuh` (`axis_lee` kernel +
`lee_axes` driver), wired into `dct_init3` validation and the
`dct_transform` dispatch in `src/plugins/covering/dct_hc.cu`. Each fiber is
buffered into registers/local before writing, so the in-place update
order is trivially safe; the descending-grade write discipline of the
Hamming plugin is kept.

Guards: `2 <= ax_j <= 16` (nibble-packed support deltas), `R <= 6`
(layer planes). Support patterns change at most `min(n, R) <= 8`
positions, since every changed position costs Lee weight >= 1.

## Support table (host side)

`coldcore.patterns.support_patterns(axes, R, "lee")`: all circular
shift vectors with total per-axis Lee weight <= R, packed in the v1
byte format (also available in ox format). Ball sizes come from
`coldcore.patterns.lee_ball_size(axes, R)` — coefficient sum (grades
0..R) of the product of per-axis weight enumerators

    L_a(z) = 1 + 2z + ... + 2z^(ceil(a/2)-1)  [+ z^(a/2) for even a].

## Reference backend

`coldcore.reference.RefLeeBackend` — an independent brute-force oracle
(numpy double loop over words vs centers, **not** a port of the DP), so
GPU parity tests are meaningful. `indep_dist = 2R` stays valid for
batched peeling because Lee distance dominates digit distance.

Registration: `-p problem=lee` in the CLI, `GpuBackend(axes, R,
problem="lee")` on the GPU.

## Verification (tests/test_lee.py, tests/test_gpu_parity_lee.py)

- Ball sizes, three independent ways (enumerator / brute force / packed
  support table), including even axes:
  |B_1|([5,5]) = 5, |B_1|([4,4]) = 5, |B_2|([4,4]) = 11,
  |B_2|([6,6]) = 13, |B_2|([7,7]) = 13, |B_1|([3,4,5]) = 7,
  |B_4|([4,4]) = 16 (whole space).
- Field identities and incremental-vs-recount on even-axis cells.
- Known optima through the untouched search core: the perfect
  Golomb-Welch code {(i, 2i mod 5)} of Z_5^2, R=1 (size 5 = the sphere
  bound, every word covered exactly once), on CPU and GPU; and the
  exhaustively brute-forced optimum of the even-q cell [4,4], R=1.
- GPU parity vs the reference on [5,5] R=1, [4,6] R=2 (antipodal
  multiplicity-1 edge), [3,4,5] R=1, [4,4,4] R=3 (deep grades).

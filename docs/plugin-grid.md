# `grid_linf` — open-boundary grid domination (problem id 3)

Dominating sets of the n-dimensional grid board `prod_j [0, m_j)` under
Chebyshev (L-infinity) balls of radius `R`, **without wraparound**:

```
cnt(x) = #{ u in S : max_j |x_j - u_j| <= R }
```

`R = 1` on a 2-D board is king-graph domination of a chessboard. This is
the torus plugin (`torus_linf`) minus the wrap — and that minus is the
point: the coverage relation is **not shift invariant**, so balls at the
edges are smaller than balls in the middle. Every other family in the
engine is circulant, which makes this one the framework's generality
test: it exercises both plugin seams, not just the axis pass.

## The two seams

**Axis pass** (`grid_axes` in `src/plugins/covering/dctcore_grid.cuh`) — a
clamped window sum over each fiber, evaluated as a sliding window:

```
newA0[a] = sum_{b = max(0, a-R)}^{min(Q-1, a+R)} A0[b]
```

One plane, `n` passes, `O(|X|)` work per transform, exactly as for the
torus. Coverage stays symmetric (`|a-b| <= R` either way), so the
transform is still self-adjoint and the gain / loss maps come out of the
same pass with the indicator inputs.

**Support walk** — this is the part the circulant families never needed.
The packed v1 table stores *circular* shifts, so near a boundary a
pattern wraps to the far side of the board, onto a word that is not in
the ball. `grid_linf` keeps the torus-shaped table (per-axis deltas `d`
with `min(d, Q-d) <= R`, deduplicated) and resolves each delta against
the actual centre digit at walk time:

```
offset = (a + d < Q) ? d : d - Q          # true board displacement
keep the pattern iff every position has |offset| <= R
```

Balls therefore shrink automatically at the edges while the table stays
the single dense max-size table the state core allocates and streams —
no per-centre tables, no masking plane. `dct_ball_update` /
`dct_ball_gather` dispatch to the clamped kernels for this problem id.

Consequence: the ox support format (`dct_use_fmt(1)`) is **rejected**
for `grid_linf` — its register-only walk is circulant by construction.

## Parameters and guards

| | |
|---|---|
| axes | `2 <= m_j <= 16` (nibble-packed deltas), `n <= 8` (patterns change up to `n` positions) |
| R | `>= 1`; `2R+1 <= m_j` is **not** required — an oversized window simply clamps to the whole axis |
| layer planes | 1 (as for the torus) |
| `indep_dist` | `None` — batched peeling is off, the metric is not the digit metric |
| mu | supported, like every family ([multi-coverage.md](multi-coverage.md)) |

## Usage

```python
from coldcore.gpu import GpuBackend
from coldcore.reference import RefGridBackend      # brute-force oracle

b = GpuBackend([8, 8], 1, problem="grid_linf")     # 8x8 king graph
```

## Verification

- `tests/test_grid.py` — balls are exactly the clamped boxes at every
  centre; shift invariance is *observably broken* (corner 4, edge 6,
  interior 9 on the 8x8 king board); an independent numpy replica of the
  walk rule reproduces the brute-force ball for **every** centre of five
  boards; field identities; and the optima below.
- `tests/test_gpu_parity_grid.py` — GPU vs. brute force on six boards
  (non-square, 3-D, `2R+1 > axis`, `R` beyond the board, max axis size)
  at `mu = 1, 2`: `cnt`, deficit, both maps over the whole space, both
  gather roles; single-codeword corner updates (a circulant walk would
  light up the far edge); the ox-format rejection; and end-to-end runs.

Optima pinned in-test, both bounds established in-test rather than
quoted: the domination number of the **8x8 king graph is 9** (upper:
kings at rows x cols `{1,4,6}^2`, verified to dominate; lower: the nine
cells `{0,3,6}^2` have pairwise disjoint closed neighbourhoods, verified
cell by cell — so nine dominators are needed), and of the **5x9 board is
6** (`{1,4} x {1,4,7}` dominates; `{0,3} x {0,3,6}` is a disjoint
packing of size 6). Both equal the classic
`ceil(m/(2R+1)) * ceil(n/(2R+1))`. The 2-fold domination number of the
**5x5 board is 8** (union of the two disjoint dominating sets
`{0,3}^2` and `{1,4}^2`; packing bound `4 * mu`).

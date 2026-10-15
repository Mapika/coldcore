#!/usr/bin/env python3
"""Example 3 — teach coldcore a brand-new problem, in one file, no CUDA.

    python examples/03_custom_problem.py

What this shows
---------------
Examples 1 and 2 used problems that ship with coldcore.  This one adds a
problem the framework has never heard of — **toroidal queen domination**
— without touching a single line of the search core, the CLI, or any
compiled code.  Everything new lives in the ~90 lines of
`QueenTorusBackend` below.

The problem: place queens on an n x n board wrapped into a torus so that
every square is occupied or attacked.  A queen at (x, y) attacks its
whole row, its whole column, and both wrapped diagonals.  For n = 7 the
minimum is 4 — and this script *proves* that by exhaustive enumeration
(there is no dominating set of 3 queens among the C(49,3) triples), so
the number the search core reports is checkable against ground truth.

Why it is a good test of the seam
---------------------------------
The two problems that ship with coldcore are *metric balls* (Hamming,
Chebyshev).  A queen's attack set is **not** a metric ball: it is a
union of four lines, an arbitrary symmetric neighborhood.  The search
core never assumes otherwise — it only asks the backend the questions in
`src/coldcore/protocol.py`:

    recount(idx)          rebuild the exact multiplicity field `cnt`
    gain_map()            write, for EVERY point, the exact number of
                          currently-uncovered points it would cover
    loss_map()            same for privately-covered points (cnt == 1)
    ball_update(idx, d)   incremental +1 / -1 of `cnt`
    ball_gather(idx, t)   exact per-candidate count of cnt == t in its
                          neighborhood — never stale, this is what makes
                          the lazy greedy exact
    count_eq(t)           how many points have cnt == t (t=0: uncovered)
    map_max / map_hist / map_extract / map_read_at   queries on the map

A CUDA plugin answers exactly these questions with separable axis-DP
sweeps over the whole space; this file answers them with dense numpy
over a precomputed 49x49 neighborhood matrix.  The layer above cannot
tell the difference.  That is the plugin seam, exercised without a GPU.

Scaling note: this reference-style backend is O(|X|^2) in memory and
time, which is exactly why real problems get a CUDA plugin.  For a toy
board it is instant, and it is the right first step when adding a
problem: write the brute-force backend first, then make the GPU plugin
match it (CONTRIBUTING.md: "no reference, no merge").
"""
import itertools
import sys

import numpy as np

try:
    from coldcore.protocol import Backend
    from coldcore.search import Searcher
    from coldcore.space import ProductSpace
except ImportError:                                    # running from a checkout
    import os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from coldcore.protocol import Backend
    from coldcore.search import Searcher
    from coldcore.space import ProductSpace

SIDE = 7
OPTIMUM = 4                                            # proven below


def queen_neighborhood(side):
    """(size, size) boolean: does a queen on square i attack square j?

    Squares are indexed the way `ProductSpace` indexes them, so the
    matrix lines up with the index space the search core works in.
    Row, column, and both wrapped diagonals; symmetric, reflexive.
    """
    digits = ProductSpace(side, 2).all_words().astype(np.int64)
    x, y = digits[:, 0], digits[:, 1]
    dx = (x[:, None] - x[None, :]) % side
    dy = (y[:, None] - y[None, :]) % side
    same_row = dx == 0
    same_col = dy == 0
    diag = dx == dy                                    # "/" direction, wrapped
    anti = (dx + dy) % side == 0                       # "\" direction, wrapped
    return same_row | same_col | diag | anti


class QueenTorusBackend(Backend):
    """A complete coldcore backend for queen domination on a torus.

    State it owns, mirroring what a GPU plugin keeps in device memory:
      `cnt` — for every square, how many queens attack it (the exact
              multiplicity field);
      `map` — a scratch field written by gain_map()/loss_map() and read
              by the map_* queries.
    """

    def __init__(self, side):
        self.side = side
        self.space = ProductSpace(side, 2)             # the index <-> (x,y) map
        self.extract_cap = 1 << 20                     # max map_extract results
        # `indep_dist`: removals farther apart than this cannot affect each
        # other's loss, which lets peel() drop several queens at once. For a
        # metric ball it is 2R; a queen's lines reach across the whole board,
        # so there is no such distance -> None -> peel one at a time.
        self.indep_dist = None
        self._nbr = queen_neighborhood(side)
        self.cnt = np.zeros(self.space.size, dtype=np.int64)
        self.map = np.zeros(self.space.size, dtype=np.int64)

    # -- the neighborhood ------------------------------------------------
    def ball_indices(self, w):
        """Squares attacked by a queen on square `w` (including `w`)."""
        return np.flatnonzero(self._nbr[int(w)])

    # -- exact fields ----------------------------------------------------
    def recount(self, idx):
        """Rebuild `cnt` from scratch: the ground truth after any drift."""
        self.cnt[:] = 0
        for w in np.atleast_1d(np.asarray(idx, dtype=np.int64)):
            self.cnt[self.ball_indices(w)] += 1

    def _whole_space_field(self, target):
        """map[w] = #{u attacked by w : cnt[u] == target}, for every w.

        This is the operation a GPU plugin implements as a separable
        axis-by-axis DP sweep in O(n * q^n); here it is one dense
        matrix-vector product.  The point is that it is *exact and for
        the whole space at once* — no sampling, no incumbent-only
        evaluation.
        """
        self.map[:] = self._nbr[:, self.cnt == target].sum(axis=1)

    def gain_map(self):
        """Marginal gain of adding each square: uncovered points it wins."""
        self._whole_space_field(0)
        # A constrained variant ("some squares may not hold a queen") is a
        # one-liner here: self.map[~self.allowed] = 0 hides them from the
        # greedy, because candidates only ever enter the search through
        # map_extract() on this field.

    def loss_map(self):
        """Marginal loss: read at incumbents, the points they cover alone."""
        self._whole_space_field(1)

    # -- incremental updates and exact per-candidate queries -------------
    def ball_update(self, idx, delta):
        """Apply +1 (place) / -1 (remove) to `cnt`. Does not touch `map`."""
        for w in np.atleast_1d(np.asarray(idx, dtype=np.int64)):
            self.cnt[self.ball_indices(w)] += delta

    def ball_gather(self, idx, target):
        """Exact count of `cnt == target` in each candidate's neighborhood.

        The lazy greedy uses a *stale* threshold slice of the gain map as
        a bound, then re-checks the popped candidates here, and only
        accepts a placement whose exact gain still beats the next bound.
        That is what keeps "lazy" and "exact" compatible.
        """
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        return np.array([(self.cnt[self.ball_indices(w)] == target).sum()
                         for w in idx], dtype=np.int32)

    # -- whole-space queries ---------------------------------------------
    def count_eq(self, target):
        return int((self.cnt == target).sum())         # uncovered() = count_eq(0)

    def map_max(self):
        return int(self.map.max(initial=0))

    def map_hist(self, nbins, vmax):
        """Positive map values binned at (v-1)*nbins//vmax, clipped.

        Optional capability: set `supports_hist = False` and raise here
        and the greedy falls back to threshold escalation.  The exact
        binning rule matters only for parity with a GPU implementation.
        """
        out = np.zeros(nbins, dtype=np.int32)
        v = self.map[self.map > 0]
        if len(v) and vmax > 0:
            bins = np.minimum((v - 1) * nbins // vmax, nbins - 1).astype(np.int64)
            np.add.at(out, bins, 1)
        return out

    def map_extract(self, thr, cap=None):
        """(idx, val, found) for map >= thr; `found` may exceed len(idx)."""
        cap = min(cap or self.extract_cap, self.extract_cap)
        idx = np.flatnonzero(self.map >= thr).astype(np.int64)
        found = len(idx)
        idx = idx[:cap]
        return idx, self.map[idx].astype(np.int32), found

    def map_read_at(self, idx):
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        return self.map[idx].astype(np.int32)


# ---------------------------------------------------------------------
# Ground truth, computed independently of the framework.
# ---------------------------------------------------------------------
def verify_domination(cells, side):
    """Independent check that every square is attacked or occupied."""
    hit = np.zeros((side, side), dtype=bool)
    for x, y in cells:
        for k in range(side):
            hit[x, k] = hit[k, y] = True               # row and column
            hit[(x + k) % side, (y + k) % side] = True  # "\" wrapped diagonal
            hit[(x + k) % side, (y - k) % side] = True  # "/" wrapped diagonal
    return bool(hit.all())


def exhaustive_minimum(side, upto):
    """Smallest dominating set found by brute force over all subsets <= upto."""
    nbr = queen_neighborhood(side)
    for k in range(1, upto + 1):
        for combo in itertools.combinations(range(side * side), k):
            if nbr[list(combo)].any(axis=0).all():
                return k
    return None


def board(cells, side):
    grid = [["." for _ in range(side)] for _ in range(side)]
    for x, y in cells:
        grid[x][y] = "Q"
    return "\n".join("      " + " ".join(row) for row in grid)


def main():
    print(f"Toroidal queen domination, {SIDE}x{SIDE} — a problem coldcore "
          f"has never seen")

    # 0. ground truth by exhaustive enumeration (cheap at this size).
    exact = exhaustive_minimum(SIDE, upto=OPTIMUM)
    print(f"    brute force:       minimum is {exact} queens "
          f"(no smaller subset dominates)")
    assert exact == OPTIMUM

    backend = QueenTorusBackend(SIDE)
    solutions = []
    with backend:
        s = Searcher(
            backend,
            seed=7,
            # the LNS logs one line per 20 rounds; on a 49-square board that
            # is thousands of lines a second, so only keep the milestones
            on_log=lambda m: None if m.startswith("lns ") else print(f"    | {m}"),
            on_solved=lambda M, idx: solutions.append((M, idx.copy())),
        )

        # 1. Start from a deliberately WASTEFUL feasible solution — random
        #    queens until the board is dominated — so the machinery below
        #    has something to strip.  (Examples 1 and 2 start from empty
        #    and let greedy_fill build one instead; both are supported.)
        rng = np.random.default_rng(0)
        start = []
        while True:
            backend.recount(np.array(start, dtype=np.int64))
            if backend.uncovered() == 0:
                break
            start.append(int(rng.integers(backend.space.size)))
        s.load(np.array(start, dtype=np.int64))
        print(f"    random start:      M={len(s.code)} (uncovered={s.uncov})")

        # 2. peel(): remove every queen whose exact private coverage is 0.
        #    Feasibility is preserved by construction.
        dropped = s.peel()
        print(f"    peel:              dropped {dropped} redundant "
              f"{'queen' if dropped == 1 else 'queens'} -> M={len(s.code)}")
        assert s.verify_full() == 0

        # 3. descend(): notch M downwards, repairing each notch with
        #    ruin-and-recreate LNS.  Every feasible size fires on_solved.
        #    step0=1 keeps the descent from overshooting past the optimum
        #    and burning a full budget on an infeasible target.
        s.descend(floor=OPTIMUM, notch_budget_s=3.0, step0=1)

    M, idx = min(solutions, key=lambda t: t[0])
    cells = [tuple(int(d) for d in row) for row in backend.index_word(idx)]
    print(f"    best size:         M={M}")
    ok = verify_domination(cells, SIDE)
    print(f"    independent check: {'DOMINATES the torus' if ok else 'GAPS LEFT'}")
    print("    queens at:         " + ", ".join(f"({x},{y})" for x, y in cells))
    print(board(cells, SIDE))

    assert ok, "solution does not dominate the torus"
    assert M == OPTIMUM, f"expected the proven optimum {OPTIMUM}, got {M}"
    print("\nOK — a brand-new problem solved to proven optimality by the "
          "stock search core,\n     with one new class and zero changes "
          "to coldcore itself.")


if __name__ == "__main__":
    main()

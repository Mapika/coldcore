#!/usr/bin/env python3
"""Example 2 — the same search core, a different problem: king domination.

    python examples/02_domination.py

What this shows
---------------
Example 1 solved a covering code.  This one solves a *graph domination*
problem: place as few kings as possible on a 9x9 chessboard **wrapped
into a torus** so that every square is occupied or attacked.  A king at
(x, y) dominates the 3x3 block around it, wrapped modulo 9.

Different problem, different metric — and yet nothing in
`coldcore.search` changes.  The problem lives entirely behind the
backend protocol: `RefTorusBackend` differs from `RefCoveringBackend` by
one coverage predicate (Chebyshev distance on circular axes instead of
Hamming distance).  Swapping `--backend ref` for `--backend gpu` swaps
in the CUDA `torus_linf` plugin with the same result and the same code
path above the seam.

The answer is known: the 3x3 blocks tile the 9x9 torus exactly, so 9
kings suffice and (81 squares / 9 squares per king) 9 are necessary.
The domination number is exactly **9**, and the searcher has to find it.

The equivalent CLI invocation is:

    coldcore solve --backend ref -p problem=torus_linf -p axes=9x9 -p R=1
"""
import sys
import time

import numpy as np

try:
    from coldcore.reference import RefTorusBackend
    from coldcore.search import Searcher
except ImportError:                                    # running from a checkout
    import os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from coldcore.reference import RefTorusBackend
    from coldcore.search import Searcher

SIDE = 9                                               # torus is SIDE x SIDE
R = 1                                                  # king move = L-inf ball 1
OPTIMUM = 9


def verify_domination(cells, side, R):
    """Independent check: does every square lie in some king's wrapped box?"""
    dominated = np.zeros((side, side), dtype=bool)
    for x, y in cells:
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                dominated[(x + dx) % side, (y + dy) % side] = True
    return bool(dominated.all())


def board(cells, side):
    """Pretty-print the torus with K for kings and . for empty squares."""
    grid = [["." for _ in range(side)] for _ in range(side)]
    for x, y in cells:
        grid[x][y] = "K"
    return "\n".join("      " + " ".join(row) for row in grid)


def main():
    backend = RefTorusBackend([SIDE, SIDE], R)
    print(f"{SIDE}x{SIDE} king torus — dominating set (optimum {OPTIMUM})")
    print(f"    space:             {backend.space.size} squares, "
          f"axes={backend.space.axes}, R={R}")

    solutions = []
    with backend:
        s = Searcher(
            backend,
            seed=1,
            on_log=lambda m: print(f"    | {m}"),
            on_solved=lambda M, idx: solutions.append((M, idx.copy())),
        )
        s.load(np.zeros(0, dtype=np.int64))            # start from nothing
        t0 = time.time()
        assert s.greedy_fill(target_m=10 ** 9) == 0, "greedy left squares free"
        print(f"    greedy cover:      M={len(s.code)}  ({time.time()-t0:.2f}s)")

        # peel() alone often finishes this instance: it drops every king
        # whose *private* coverage (the exact loss field, evaluated at the
        # incumbents) is zero.  descend() then pushes below that.
        s.descend(floor=OPTIMUM, notch_budget_s=10.0)

    M, idx = min(solutions, key=lambda t: t[0])
    cells = [tuple(int(d) for d in row) for row in backend.index_word(idx)]
    print(f"    best size:         M={M}")
    ok = verify_domination(cells, SIDE, R)
    print(f"    independent check: {'DOMINATES the torus' if ok else 'GAPS LEFT'}")
    print("    kings at:          " + ", ".join(f"({x},{y})" for x, y in cells))
    print(board(cells, SIDE))

    assert ok, "solution does not dominate the torus"
    assert M == OPTIMUM, f"expected the known optimum {OPTIMUM}, got {M}"
    print(f"\nOK — {OPTIMUM} kings, the known optimum, with an unchanged "
          f"search core.")


if __name__ == "__main__":
    main()

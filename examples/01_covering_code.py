#!/usr/bin/env python3
"""Example 1 — a covering code from scratch, on the CPU reference backend.

    python examples/01_covering_code.py

What this shows
---------------
The founding problem of coldcore: a *covering code* K_q(n,R) is the
smallest set C of words in the alphabet cube Z_q^n such that every word
of Z_q^n is within Hamming distance R of some codeword.

We solve the cell **K_3(4,1)**: length-4 words over a 3-letter alphabet,
radius 1.  The space has 3^4 = 81 words and each ball covers
1 + n(q-1) = 9 of them, so 81/9 = 9 is a trivial lower bound — and it is
attained by the perfect ternary Hamming code.  So the answer is exactly
**K_3(4,1) = 9**, which makes this a good end-to-end correctness check:
the search core must find 9, not 10.

Nothing here is GPU-specific.  `RefCoveringBackend` is a pure-numpy
brute-force implementation of the same `coldcore.protocol.Backend`
surface the CUDA plugins implement, so the *identical* search core runs
on it — that is the whole point of the seam.  It is O(|X|^2), i.e. for
toy instances only.

At the bottom we also run the radius-2 cell K_3(4,2) for contrast; its
exact value is 3 (verified by exhaustive enumeration over all
C(81,3) triples).
"""
import sys
import time

import numpy as np

try:
    from coldcore.reference import RefCoveringBackend
    from coldcore.search import Searcher
except ImportError:                                    # running from a checkout
    import os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from coldcore.reference import RefCoveringBackend
    from coldcore.search import Searcher


# ---------------------------------------------------------------------
# An INDEPENDENT verifier.  House rule (CONTRIBUTING.md): the search core
# reports solutions, a separate check decides what is real.  This one
# recomputes coverage directly from the digit rows and never touches the
# backend that produced them.
# ---------------------------------------------------------------------
def verify_covering(words, q, n, R):
    """True iff every word of Z_q^n is within Hamming distance R of `words`."""
    all_words = np.stack(np.meshgrid(*([np.arange(q)] * n), indexing="ij"), -1)
    all_words = all_words.reshape(-1, n)
    dist = (all_words[:, None, :] != np.asarray(words)[None, :, :]).sum(-1)
    return bool((dist.min(axis=1) <= R).all())


def solve(q, n, R, floor, seed=1, notch_budget_s=5.0):
    """Greedy build -> peel -> notch descent; returns the best solution seen.

    `on_solved` fires once per *verified feasible* size, so collecting
    from it is the supported way to read the answer out of a run: the
    searcher's own `code` attribute is scratch space that may be mid-ruin
    when the descent stops on a budget."""
    backend = RefCoveringBackend(q, n, R)
    solutions = []                                     # (M, indices) per notch

    with backend:                                      # Backend is a ctx manager
        s = Searcher(
            backend,
            seed=seed,
            on_log=lambda m: print(f"    | {m}"),
            on_solved=lambda M, idx: solutions.append((M, idx.copy())),
        )

        # 1. start empty and let the exact lazy greedy build a first cover.
        #    "Exact" = every candidate's marginal gain is the true number of
        #    still-uncovered points it would cover, read off a whole-space
        #    gain field — no sampling, no estimate.
        s.load(np.zeros(0, dtype=np.int64))
        t0 = time.time()
        uncovered = s.greedy_fill(target_m=10 ** 9)
        assert uncovered == 0, "greedy failed to produce a feasible cover"
        print(f"    greedy cover:      M={len(s.code)}  ({time.time()-t0:.2f}s)")

        # 2. descend: try M-step, repair by ruin-and-recreate LNS, halve the
        #    step on failure.  Stops at `floor` (the value we are trying to
        #    reach) or when three consecutive single-step notches fail.
        s.descend(floor=floor, notch_budget_s=notch_budget_s)

    M, idx = min(solutions, key=lambda t: t[0])
    return M, backend.index_word(idx)                  # -> (M, n) digit rows


def main():
    print("K_3(4,1) — ternary, length 4, radius 1 (optimum 9)")
    M, code = solve(q=3, n=4, R=1, floor=9)
    print(f"    best size:         M={M}")
    ok = verify_covering(code, 3, 4, 1)
    print(f"    independent check: {'COVERS Z_3^4' if ok else 'NOT A COVER'}")
    print("    codewords:")
    for row in code:
        print("      " + " ".join(str(int(d)) for d in row))
    assert ok, "solution does not cover the space"
    assert M == 9, f"expected the optimum K_3(4,1) = 9, got {M}"

    print()
    print("K_3(4,2) — same cube, radius 2 (optimum 3)")
    M2, code2 = solve(q=3, n=4, R=2, floor=3)
    print(f"    best size:         M={M2}")
    ok2 = verify_covering(code2, 3, 4, 2)
    print(f"    independent check: {'COVERS Z_3^4' if ok2 else 'NOT A COVER'}")
    print("    codewords:")
    for row in code2:
        print("      " + " ".join(str(int(d)) for d in row))
    assert ok2, "radius-2 solution does not cover the space"
    assert M2 == 3, f"expected the optimum K_3(4,2) = 3, got {M2}"

    print("\nOK — both cells solved to their known optima.")


if __name__ == "__main__":
    main()

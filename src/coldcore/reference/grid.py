"""Brute-force reference backend for open-boundary grid domination.

Coverage predicate: Chebyshev (L-inf) distance <= R with NO wraparound
— a point dominates the (2R+1)^n box around it, CLAMPED to the board.
R=1 on a 2-D board is king-graph domination of a chessboard, whose
optimum is the classic ceil(m/(2R+1)) * ceil(n/(2R+1)).

The difference from RefTorusBackend is exactly the missing wrap, which
is what breaks shift invariance: balls at the edges are smaller.
"""
import numpy as np

from .base import BruteForceBackend


class RefGridBackend(BruteForceBackend):

    def __init__(self, axes, R, **kw):
        super().__init__(axes, R, **kw)
        self.indep_dist = None          # peel one at a time (metric differs)

    def _covers(self, a, b):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        return d.max(axis=-1) <= self.R

"""Brute-force reference backend for dominating sets on torus grids.

Coverage predicate: Chebyshev (L-inf) distance <= R with circular axes
— a point dominates the (2R+1)^n box around it, wrapped. R=1 on a 2-D
torus is king-graph domination.
"""
import numpy as np

from .base import BruteForceBackend


class RefTorusBackend(BruteForceBackend):

    def __init__(self, axes, R, **kw):
        super().__init__(axes, R, **kw)
        if any(2 * R + 1 > a for a in self.space.axes):
            raise ValueError("need 2R+1 <= ax_j on every axis")
        self.indep_dist = None          # peel one at a time (metric differs)
        self._axarr = np.array(self.space.axes, dtype=np.int16)

    def _covers(self, a, b):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        d = np.minimum(d, self._axarr - d)
        return d.max(axis=-1) <= self.R

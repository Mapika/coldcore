"""Brute-force reference backend for Lee-metric covering codes.

Coverage predicate: SUM over coordinates of the circular per-axis
distance min(|a-b|, ax_j - |a-b|) <= R (mixed-radix axes allowed) — the
sum-metric analogue of Hamming on Z_q alphabets.
"""
import numpy as np

from .base import BruteForceBackend


class RefLeeBackend(BruteForceBackend):

    def __init__(self, axes, R, **kw):
        super().__init__(axes, R, **kw)
        # Lee distance dominates digit (Hamming) distance, so removals at
        # digit distance > 2R cannot interact: batched peeling stays valid
        self.indep_dist = 2 * R
        self._axarr = np.array(self.space.axes, dtype=np.int16)

    def _covers(self, a, b):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        d = np.minimum(d, self._axarr - d)
        return d.sum(axis=-1) <= self.R

"""Brute-force reference backend for covering codes K(n; ax; R).

Coverage predicate: Hamming distance <= R (mixed-radix axes allowed;
RefCoveringBackend is the homogeneous K_q(n,R) shorthand).
"""
from .base import BruteForceBackend


class RefHammingBackend(BruteForceBackend):

    def __init__(self, axes, R, **kw):
        super().__init__(axes, R, **kw)
        self.indep_dist = 2 * R

    def _covers(self, a, b):
        return (a != b).sum(axis=-1) <= self.R


class RefCoveringBackend(RefHammingBackend):
    """Homogeneous covering-code cell K_q(n,R)."""

    def __init__(self, q, n, R, **kw):
        super().__init__([q] * n, R, **kw)
        self.q, self.n = q, n

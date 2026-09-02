"""Shared brute-force machinery for reference backends.

Subclasses define one thing: the coverage predicate between digit
arrays. Everything else (fields, maps, queries) is generic and matched
operation-for-operation to the GPU contract, including map_hist binning
and unordered map_extract with found > cap.

mu-fold multiple covering (docs/multi-coverage.md) lives here too, and
is likewise problem-blind: it only changes which cnt levels count as
deficient (cnt < mu) and critical (1 <= cnt <= mu).

O(space^2 * n) — tiny instances only, by design.
"""
import numpy as np

from ..protocol import DenseFieldBackend
from ..space import ProductSpace

_MAX_SPACE = 1 << 16


class BruteForceBackend(DenseFieldBackend):

    def __init__(self, axes, R, extract_cap=1 << 22, seed_shuffle=None, mu=1):
        self.R = R
        self.mu = int(mu)
        if self.mu < 1:
            raise ValueError("mu must be >= 1")
        self.space = ProductSpace(axes)
        size = self.space.size
        if size > _MAX_SPACE:
            raise ValueError(
                f"reference backend is brute force; |X|={size} too large")
        self.extract_cap = extract_cap
        self._digs = self.space.all_words().astype(np.int16)   # (size, n)
        self.cnt = np.zeros(size, dtype=np.int64)
        self.map = np.zeros(size, dtype=np.int64)
        # optional shuffling of extract order, to mimic the GPU's
        # nondeterministic slot order in tests
        self._rng = (np.random.default_rng(seed_shuffle)
                     if seed_shuffle is not None else None)

    # ---------------- the problem-specific part ----------------
    def _covers(self, a, b):
        """Boolean coverage predicate for broadcastable digit arrays
        a (..., n) vs b (..., n)."""
        raise NotImplementedError

    def ball_indices(self, w):
        """Indices of all points covered by a move at word index w."""
        m = self._covers(self._digs, self._digs[int(w)][None, :])
        return np.flatnonzero(m)

    # ---------------- solution / fields ----------------
    def recount(self, idx):
        self.cnt[:] = 0
        for w in np.atleast_1d(np.asarray(idx, dtype=np.int64)):
            self.cnt[self.ball_indices(w)] += 1

    # ---------------- fields (the mu-fold predicates and every
    # cnt/map query live in protocol.DenseFieldBackend) ----------------
    def _eval_field(self, mask):
        """map[w] = #{u in ball(w): mask[u]} for every w."""
        size = self.space.size
        sel = np.flatnonzero(mask)
        self.map[:] = 0
        if not len(sel):
            return
        chunk = max(1, (1 << 22) // len(sel))
        for lo in range(0, size, chunk):
            hi = min(size, lo + chunk)
            m = self._covers(self._digs[lo:hi, None, :],
                             self._digs[None, sel, :])
            self.map[lo:hi] = m.sum(axis=1)

    def gain_map(self):
        self._eval_field(self._deficient())

    def loss_map(self):
        self._eval_field(self._critical())

    # ---------------- incremental / exact ----------------
    def ball_update(self, idx, delta):
        for w in np.atleast_1d(np.asarray(idx, dtype=np.int64)):
            self.cnt[self.ball_indices(w)] += delta

    def ball_gather(self, idx, target):
        """target is a coverage ROLE: 0 = deficient (cnt < mu), 1 =
        critical (1 <= cnt <= mu), anything else = literal cnt ==
        target. At mu=1 all three are the literal count."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        mask = self._role_mask(target)
        out = np.empty(len(idx), dtype=np.int32)
        for k, w in enumerate(idx):
            out[k] = int(mask[self.ball_indices(w)].sum())
        return out


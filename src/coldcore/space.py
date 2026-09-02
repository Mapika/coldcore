"""Finite product space X = prod_j Z_{ax_j}: index <-> digit conversions.

Heterogeneous (mixed-radix) axes are supported; ``ProductSpace(q, n)``
is the homogeneous shorthand. Indexing is little-endian: index =
sum_j digit_j * prod_{i<j} ax_i.
"""
import numpy as np


class ProductSpace:

    def __init__(self, q_or_axes, n=None):
        if n is not None:
            axes = [int(q_or_axes)] * int(n)
        else:
            axes = [int(a) for a in q_or_axes]
        self.axes = axes
        self.n = len(axes)
        self.q = axes[0] if len(set(axes)) == 1 else None   # homogeneous only
        self.size = 1
        pows = []
        for a in axes:
            pows.append(self.size)
            self.size *= a
        self.pow_q = np.array(pows, dtype=np.int64)
        self._ax = np.array(axes, dtype=np.int64)

    def word_index(self, words):
        """(k, n) digit array -> int64 indices."""
        w = np.asarray(words, dtype=np.int64)
        return w @ self.pow_q

    def index_word(self, idx):
        """int64 indices -> (k, n) uint8 digit array."""
        idx = np.asarray(idx, dtype=np.int64).ravel()
        out = np.empty((len(idx), self.n), dtype=np.uint8)
        x = idx.copy()
        for i in range(self.n):
            out[:, i] = x % self._ax[i]
            x //= self._ax[i]
        return out

    def all_words(self):
        """(size, n) digit array of the whole space (tiny spaces only)."""
        return self.index_word(np.arange(self.size, dtype=np.int64))

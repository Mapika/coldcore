"""The backend protocol: the narrow surface the search core runs on.

A backend is one problem instance resident somewhere (GPU plugin, or
pure-numpy reference). It maintains:

  - ``cnt``   — the exact multiplicity field over the whole space
  - a *map*   — a dense work field written by :meth:`gain_map` /
                :meth:`loss_map` and read by the ``map_*`` queries

Semantics contract (matching the proven covering engine):

  - ``recount(idx)`` rebuilds ``cnt`` exactly from the solution set.
  - ``gain_map()``  writes, for every point w, the exact marginal gain of
    adding w to the current solution.
  - ``loss_map()``  writes the exact marginal loss field; evaluated at
    incumbent points it is the number of points they privately cover.
  - ``ball_update(idx, delta)`` applies incremental adds (+1) / removes
    (-1) of solution points to ``cnt`` (does not touch the map).
  - ``ball_gather(idx, target)`` returns, per candidate, the exact count
    of points in its neighborhood with ``cnt == target`` (target 0 =
    fresh gain, 1 = private coverage) — always current, never stale.
  - ``map_extract(thr)`` returns (idx, val, found) for map values >= thr,
    unordered; ``found`` may exceed the returned length (capped).
  - ``map_hist(nbins, vmax)`` bins positive values v at (v-1)*nbins//vmax
    (clipped to the last bin).
  - ``count_eq(t)`` counts points with ``cnt == t`` over the whole space.

GPU note: bindings follow the ctypes precedent of the prototype record
engine. The C-side ABI is documented in docs/PLUGIN.md; this Python
protocol is the stable surface.
"""
import numpy as np


class Backend:
    """Abstract base. Attributes every backend must set:

    space        : ProductSpace instance
    extract_cap  : max entries a single map_extract can return
    indep_dist   : removals farther apart than this (digit/Hamming
                   distance) cannot affect each other's loss (covering:
                   2R); None disables batched peeling.
    """

    space = None
    extract_cap = 1 << 22
    indep_dist = None
    supports_hist = True     # map_hist is an optional v1 capability;
                             # the search core falls back gracefully

    # -- convenience passthroughs -------------------------------------
    def word_index(self, words):
        return self.space.word_index(words)

    def index_word(self, idx):
        return self.space.index_word(idx)

    # -- lifecycle ------------------------------------------------------
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- required operations ---------------------------------------------
    def recount(self, idx):
        raise NotImplementedError

    def gain_map(self):
        raise NotImplementedError

    def loss_map(self):
        raise NotImplementedError

    def ball_update(self, idx, delta):
        raise NotImplementedError

    def ball_gather(self, idx, target):
        raise NotImplementedError

    def count_eq(self, target):
        raise NotImplementedError

    def uncovered(self):
        return self.count_eq(0)

    def map_max(self):
        raise NotImplementedError

    def map_hist(self, nbins, vmax):
        raise NotImplementedError

    def map_extract(self, thr, cap=None):
        raise NotImplementedError

    def map_read_at(self, idx):
        raise NotImplementedError

    def map_all(self):
        """The whole map field, as one array (treat as read-only).

        The orbit layer needs the entire field per transform to fold it
        into per-orbit sums; the generic form is a full ``map_read_at``,
        and a backend holding the field in host memory overrides it with
        a view."""
        return self.map_read_at(np.arange(self.space.size, dtype=np.int64))


class DenseFieldBackend(Backend):
    """Host backend whose ``cnt`` and ``map`` fields are dense arrays.

    Everything that only *reads* those two arrays — the mu-fold coverage
    role predicates and every ``map_*`` / ``count_*`` query — is
    problem-blind and byte-for-byte identical between the brute-force
    reference backends and the axis-DP CPU backend, so it lives here
    once. A subclass supplies ``space``, ``mu``, ``extract_cap``, the two
    field arrays, and the transforms that fill them.

    ``_rng`` (optional) shuffles ``map_extract`` output, mimicking the
    GPU's nondeterministic slot order in tests.
    """

    cnt = None
    map = None
    mu = 1
    _rng = None

    # ---------------- mu-fold coverage predicates ----------------
    def _deficient(self):
        """Words still short of the multiplicity requirement."""
        return self.cnt < self.mu

    def _critical(self):
        """Words that would fall below mu if one covering codeword left
        (words with cnt == 0 are in no codeword's ball, so excluding
        them is exact and reproduces [cnt==1] at mu=1)."""
        return (self.cnt >= 1) & (self.cnt <= self.mu)

    def _role_mask(self, target):
        if target == 0:
            return self._deficient()
        if target == 1:
            return self._critical()
        return self.cnt == target

    # ---------------- queries ----------------
    def count_eq(self, target):
        """Literal count of words with cnt == target (mu-blind)."""
        return int((self.cnt == target).sum())

    def uncovered(self):
        """Total deficit sum_x max(0, mu - cnt(x)); 0 iff feasible."""
        return int(np.maximum(self.mu - self.cnt, 0).sum(dtype=np.int64))

    def map_max(self):
        return int(self.map.max(initial=0))

    def map_hist(self, nbins, vmax):
        """Positive v binned at (v-1)*nbins//vmax, clipped (GPU parity)."""
        out = np.zeros(nbins, dtype=np.int32)
        v = np.asarray(self.map[self.map > 0], dtype=np.int64)
        if len(v) and vmax > 0:
            b = np.minimum((v - 1) * nbins // vmax, nbins - 1).astype(np.int64)
            np.add.at(out, b, 1)
        return out

    def map_extract(self, thr, cap=None):
        cap = min(cap or self.extract_cap, self.extract_cap)
        idx = np.flatnonzero(self.map >= thr).astype(np.int64)
        found = len(idx)
        if self._rng is not None:
            self._rng.shuffle(idx)
        idx = idx[:cap]
        return idx, self.map[idx].astype(np.int32), found

    def map_read_at(self, idx):
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        return self.map[idx].astype(np.int32)

    def map_all(self):
        return self.map

    def read_cnt(self, off=0, length=None):
        length = length if length is not None else self.space.size - off
        return self.cnt[off:off + length].astype(np.uint16)

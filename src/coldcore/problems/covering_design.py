"""Covering designs C(v,k,t) on the explicit-incidence backends.

A (v,k,t) covering design is a family of k-subsets ("blocks") of a
v-set such that every t-subset lies in at least one block; C(v,k,t) is
the minimum number of blocks, and the La Jolla Covering Repository
(LJCR) tabulates the best known values. With mu > 1 every t-subset must
lie in at least mu blocks.

Lowered to the incidence model of ``coldcore.incidence``:

  candidates  all k-subsets of {0..v-1}, indexed by colex rank
              (the combinatorial number system: rank of sorted
              ``s_0 < ... < s_{k-1}`` is ``sum_j C(s_j, j+1)``)
  universe    all t-subsets, indexed the same way
  incidence   block -> the C(k,t) t-subsets it contains

The block list and the incidence matrix are built vectorised: the
blocks by unranking ``0..C(v,k)-1`` (k searchsorted passes over the
binomial table), the incidence by a depth-first walk over the t-subsets
of block *positions* that shares the partial rank sums of common
prefixes, chunked over rows so temporaries stay small. The matrix is
int32 and ``4 * C(v,k) * C(k,t)`` bytes: 2.9 GB for C(32,8,4), 65 GB for
C(32,10,5).

File format: one block per line, elements 1-based, space separated —
the LJCR download format. ``read_cover`` / ``write_cover`` translate to
and from candidate indices.

``CoveringDesignBackend(v, k, t, mu=1, device="cpu")`` returns the host
backend (``device`` ``"cpu"`` or ``"ref"`` — the numpy implementation
*is* the reference) or the GPU one (``"gpu"``, needs the ``sparse``
plugin built; see docs/PLUGIN.md, ABI 2).
"""
import os
from itertools import combinations
from math import comb

import numpy as np

from ..incidence import SparseIncidenceBackend
from ..incidence_gpu import GpuIncidenceBackend
from ..protocol import Backend

#: Rows per chunk when building the incidence matrix.
_BUILD_ROWS = 1 << 20


# ------------------------------------------------------------ ranking
def binom_table(v, r):
    """``C[a, b] = comb(a, b)`` for ``0 <= a <= v``, ``0 <= b <= r``."""
    C = np.zeros((v + 1, r + 1), dtype=np.int64)
    C[:, 0] = 1
    for a in range(1, v + 1):
        for b in range(1, min(a, r) + 1):
            C[a, b] = C[a - 1, b - 1] + C[a - 1, b]
    return C


def rank_subsets(S, v=None, C=None):
    """Colex ranks of the subsets in ``S`` (``(m, r)`` integer array,
    0-based elements, any order within a row)."""
    S = np.sort(np.asarray(S, dtype=np.int64), axis=1)
    if S.ndim != 2:
        raise ValueError("S must be (m, r)")
    m, r = S.shape
    if C is None:
        if v is None:
            v = int(S.max()) + 1 if m else r
        C = binom_table(v, r)
    out = np.zeros(m, dtype=np.int64)
    for j in range(r):
        out += C[S[:, j], j + 1]
    return out


def unrank_subsets(ranks, v, r, C=None):
    """Inverse of :func:`rank_subsets`: ``(m, r)`` sorted subsets of
    ``{0..v-1}`` with the given colex ranks. Vectorised: one
    ``searchsorted`` per position, from the top."""
    ranks = np.asarray(ranks, dtype=np.int64).ravel()
    if C is None:
        C = binom_table(v, r)
    rem = ranks.copy()
    out = np.empty((len(ranks), r), dtype=np.int64)
    for j in range(r - 1, -1, -1):
        col = C[:, j + 1]                         # nondecreasing in a
        s = np.searchsorted(col, rem, side="right") - 1
        out[:, j] = s
        rem -= col[s]
    if (rem != 0).any() or (out[:, -1] >= v).any():
        raise ValueError("rank out of range for the given (v, r)")
    return out


def all_blocks(v, k):
    """All k-subsets of ``{0..v-1}`` in colex order, as a ``(C(v,k), k)``
    uint8 array (row ``r`` has rank ``r``)."""
    if not (0 < k <= v <= 255):
        raise ValueError("need 0 < k <= v <= 255")
    n = comb(v, k)
    return unrank_subsets(np.arange(n, dtype=np.int64), v, k).astype(np.uint8)


def build_incidence(blocks, v, t, rows=_BUILD_ROWS):
    """``(C(v,k), C(k,t))`` int32 matrix: row ``c`` = colex ranks of the
    t-subsets of block ``c``.

    Depth-first over the t-subsets of block positions in lex order,
    reusing the partial sums of a common prefix; each step is one table
    gather plus one add over the row chunk."""
    blocks = np.asarray(blocks)
    n_cand, k = blocks.shape
    if not (1 <= t <= k):
        raise ValueError("need 1 <= t <= k")
    if comb(v, t) > np.iinfo(np.int32).max:
        raise ValueError("universe too large for int32 indices")
    subs = list(combinations(range(k), t))
    Ct = binom_table(v, t).astype(np.int32)
    cols = [np.ascontiguousarray(Ct[:, j]) for j in range(t + 1)]
    inc = np.empty((n_cand, len(subs)), dtype=np.int32)
    for lo in range(0, n_cand, rows):
        blk = blocks[lo:lo + rows]
        part = [np.zeros(len(blk), dtype=np.int32)] + [None] * t
        prev = ()
        for j, s in enumerate(subs):
            p = 0
            while p < len(prev) and prev[p] == s[p]:
                p += 1
            for depth in range(p, t):
                part[depth + 1] = part[depth] + np.take(cols[depth + 1],
                                                        blk[:, s[depth]])
            inc[lo:lo + len(blk), j] = part[t]
            prev = s
    return inc


# ------------------------------------------------------------ files
def read_cover(path, v, k):
    """LJCR-format file -> int64 candidate indices (colex ranks).

    One block per line, k elements in ``1..v``, space separated; blank
    lines and ``#`` comments are skipped."""
    rows = []
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [int(x) for x in line.split()]
            if len(vals) != k:
                raise ValueError(f"{path}:{ln}: {len(vals)} elements, want k={k}")
            if min(vals) < 1 or max(vals) > v:
                raise ValueError(f"{path}:{ln}: element out of 1..{v}")
            if len(set(vals)) != k:
                raise ValueError(f"{path}:{ln}: repeated element")
            rows.append(vals)
    if not rows:
        return np.zeros(0, dtype=np.int64)
    S = np.asarray(rows, dtype=np.int64) - 1
    return rank_subsets(S, C=binom_table(v, k))


def write_cover(path, blocks):
    """Write ``(M, k)`` 0-based blocks in LJCR format (1-based, one block
    per line), via tmp + rename."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for row in np.asarray(blocks):
            fh.write(" ".join(str(int(x) + 1) for x in sorted(row)) + "\n")
    os.replace(tmp, path)


# ------------------------------------------------------------ backends
class CoveringDesignBackend(Backend):
    """C(v,k,t) with multiplicity ``mu`` on a host (``device="cpu"`` /
    ``"ref"``) or GPU (``"gpu"``) incidence backend.

    Instantiating this class returns :class:`CpuCoveringDesignBackend` or
    :class:`GpuCoveringDesignBackend`; both carry the problem attributes
    ``v, k, t, mu, blocks, n_univ`` and the helpers below.
    """

    def __new__(cls, v, k, t, mu=1, device="cpu", **kw):
        if cls is CoveringDesignBackend:
            if device == "gpu":
                cls = GpuCoveringDesignBackend
            elif device in ("cpu", "ref"):
                cls = CpuCoveringDesignBackend
            else:
                raise ValueError(f"unknown device {device!r}")
        return object.__new__(cls)

    def _design(self, v, k, t):
        v, k, t = int(v), int(k), int(t)
        if not (1 <= t <= k <= v):
            raise ValueError("need 1 <= t <= k <= v")
        self.v, self.k, self.t = v, k, t
        self.blocks = all_blocks(v, k)
        self.n_univ = comb(v, t)
        self._Ck = binom_table(v, k)
        # blocks sharing fewer than t elements cover no common t-subset,
        # so their removals are independent: |A n B| < t  <=>  dist > k-t
        self.indep_dist = k - t
        return build_incidence(self.blocks, v, t)

    # -- locality hooks (protocol.py): the block metric --
    @staticmethod
    def _shared(A, B):
        """``|A n B|`` for broadcastable ``(..., a)`` / ``(..., b)`` rows
        of distinct elements."""
        return (A[..., :, None] == B[..., None, :]).any(axis=-1).sum(axis=-1)

    def distance(self, a, b):
        """Block distance ``|A - B| = |A ^ B| / 2 = k - |A n B|``,
        elementwise over broadcastable candidate index arrays."""
        A = self.blocks[np.asarray(a, dtype=np.int64)]
        B = self.blocks[np.asarray(b, dtype=np.int64)]
        return (self.k - self._shared(A, B)).astype(np.int64)

    def universe_distance(self, idx, univ):
        """``t - |B n T|`` for every block in ``idx`` and t-subset in
        ``univ``: ``(len(idx), len(univ))``, 0 iff the block covers it."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        univ = np.atleast_1d(np.asarray(univ, dtype=np.int64))
        B = self.blocks[idx][:, None, :]
        T = unrank_subsets(univ, self.v, self.t)[None, :, :]
        return (self.t - self._shared(B, T)).astype(np.int64)

    def neighbours(self, idx, radius=1, m=None, rng=None):
        """Blocks at distance ``radius`` from block ``idx``: all
        ``k (v-k)`` single swaps at radius 1 (``m`` of them, sampled,
        when ``m`` is given); a sample of ``m`` (default 64) distinct
        ``radius``-fold swaps beyond that."""
        B = self.blocks[int(idx)].astype(np.int64)
        v, k = self.v, self.k
        out = np.setdiff1d(np.arange(v, dtype=np.int64), B)
        r = int(radius)
        if r < 1 or r > min(k, v - k):
            return np.zeros(0, dtype=np.int64)
        if r == 1:
            n = k * (v - k)
            rows = np.repeat(B[None, :], n, axis=0)
            rows[np.arange(n), np.repeat(np.arange(k), v - k)] = np.tile(out, k)
            res = self.block_rank(rows)
            if m is not None and int(m) < n:
                rng = np.random.default_rng() if rng is None else rng
                res = rng.choice(res, size=int(m), replace=False)
            return res
        m = 64 if m is None else int(m)
        if m <= 0:
            return np.zeros(0, dtype=np.int64)
        rng = np.random.default_rng() if rng is None else rng
        drop = np.argsort(rng.random((m, k)), axis=1)[:, :r]
        add = np.argsort(rng.random((m, v - k)), axis=1)[:, :r]
        rows = np.repeat(B[None, :], m, axis=0)
        rows[np.arange(m)[:, None], drop] = out[add]
        return np.unique(self.block_rank(rows))

    def candidates_covering(self, univ_idx):
        """All ``C(v-t, k-t)`` blocks containing the t-subset ``univ_idx``
        (sorted candidate indices)."""
        T = unrank_subsets([int(univ_idx)], self.v, self.t)[0]
        rest = np.setdiff1d(np.arange(self.v, dtype=np.int64), T)
        tails = np.array(list(combinations(rest.tolist(), self.k - self.t)),
                         dtype=np.int64).reshape(-1, self.k - self.t)
        rows = np.concatenate([np.repeat(T[None, :], len(tails), axis=0),
                               tails], axis=1)
        return np.sort(self.block_rank(rows))

    # -- ranks <-> blocks --
    def block_rank(self, blocks):
        """``(m, k)`` 0-based blocks (any element order) -> candidate
        indices."""
        S = np.asarray(blocks, dtype=np.int64)
        if S.ndim == 1:
            S = S[None, :]
        if S.shape[1] != self.k or (S.size and (S.min() < 0 or S.max() >= self.v)):
            raise ValueError(f"blocks must be (m, {self.k}) with elements in "
                             f"0..{self.v - 1}")
        return rank_subsets(S, C=self._Ck)

    def rank_block(self, idx):
        """Candidate indices -> ``(m, k)`` 0-based sorted blocks."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64)).ravel()
        return self.blocks[idx].astype(np.int64)

    # -- files (LJCR format) --
    def read_cover(self, path):
        return read_cover(path, self.v, self.k)

    def write_cover(self, path, idx):
        write_cover(path, self.rank_block(idx))

    # the CLI's generic hooks (--seed-file / --out use these when present)
    read_solution_file = read_cover
    write_solution_file = write_cover

    # -- independent verification --
    def verify(self, idx):
        """True iff the blocks ``idx`` cover every t-subset at least mu
        times, recomputed from the blocks themselves: per-block
        enumeration of its t-subsets with a fresh rank table, never the
        incidence matrix or the fields."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64)).ravel()
        Ct = binom_table(self.v, self.t)
        cov = np.zeros(self.n_univ, dtype=np.int64)
        for b in self.blocks[idx]:
            T = np.array(list(combinations(sorted(int(x) for x in b), self.t)),
                         dtype=np.int64)
            np.add.at(cov, rank_subsets(T, C=Ct), 1)
        return bool((cov >= self.mu).all())

    @property
    def label(self):
        tail = f"mu{self.mu}" if self.mu > 1 else ""
        return f"C({self.v},{self.k},{self.t}){tail}"

    def __repr__(self):
        return (f"{type(self).__name__}({self.label}: {len(self.blocks)} "
                f"blocks, {self.n_univ} t-subsets)")


class CpuCoveringDesignBackend(CoveringDesignBackend, SparseIncidenceBackend):
    """C(v,k,t) on :class:`~coldcore.incidence.SparseIncidenceBackend`."""

    def __init__(self, v, k, t, mu=1, device="cpu", **kw):
        inc = self._design(v, k, t)
        SparseIncidenceBackend.__init__(self, inc, self.n_univ, mu=mu, **kw)


class GpuCoveringDesignBackend(CoveringDesignBackend, GpuIncidenceBackend):
    """C(v,k,t) on :class:`~coldcore.incidence_gpu.GpuIncidenceBackend`.

    The host copy of the incidence matrix is released once uploaded
    (``keep_host=True`` retains it as ``inc``)."""

    def __init__(self, v, k, t, mu=1, device="gpu", keep_host=False, **kw):
        inc = self._design(v, k, t)
        GpuIncidenceBackend.__init__(self, inc, self.n_univ, mu=mu, **kw)
        self.inc = inc if keep_host else None

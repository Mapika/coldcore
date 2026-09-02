"""Explicit bipartite incidence: a second backend family.

The product-space backends (``coldcore.dp``, the CUDA plugin) compute
gain/loss fields by separable transforms over a space in which
candidates and covered points are the *same* set. This module serves
problems where they are not: a finite set of **candidates** (indexed
``0..n_cand-1``), a finite **universe** (indexed ``0..n_univ-1``), and an
explicit incidence list — every candidate covers exactly ``d`` universe
elements, given as one row of an ``(n_cand, d)`` integer matrix. Covering
designs C(v,k,t) are the first such problem
(``coldcore.problems.covering_design``): candidates are k-subsets,
universe elements are t-subsets, a block covers the C(k,t) t-subsets it
contains.

Nothing in the search core changes. It asks the backend only the
protocol questions (``recount``, ``gain_map``, ``ball_gather``, the
``map_*`` queries — see ``coldcore.protocol``), and here each is a gather
over the incidence rows instead of an axis transform. The one thing it
does read off the backend's ``space`` is the index/digit conversion and
the axis list, for which :class:`IncidenceSpace` stands in: a single
axis whose "digits" are the candidate indices themselves.

Semantics are the ``DenseFieldBackend`` ones, mu-fold included:
``cnt[u] < mu`` is deficient, ``1 <= cnt[u] <= mu`` is critical,
``gain_map`` counts deficient elements per candidate, ``loss_map`` counts
critical ones, ``ball_gather`` answers the same per-candidate counts
exactly and never stale. The fields are two dense arrays — ``cnt`` over
the universe, ``map`` over the candidates — so all the query code is the
shared implementation in ``protocol.DenseFieldBackend``.

Cost: a field is one pass over the incidence matrix, ``O(n_cand * d)``;
memory for the matrix is ``4 * n_cand * d`` bytes (int32 universe
indices; int64 when the universe exceeds 2^31). The GPU implementation
of the same operations is ``coldcore.incidence_gpu``.
"""
import numpy as np

from .protocol import DenseFieldBackend

#: Elements per (rows x d) gather temporary in the chunked passes; bounds
#: memory at a few tens of MB regardless of the incidence size.
_CHUNK_ELEMS = 1 << 24

#: ``ball_gather`` builds the whole-universe role mask only when the batch
#: touches at least this fraction of the universe; below that it gathers
#: cnt values and applies the predicate to those (same answer, cheaper).
_MASK_BREAKEVEN = 2


class IncidenceSpace:
    """Stand-in for :class:`~coldcore.space.ProductSpace` on a plain index
    set: one axis of size ``size``.

    The search core reads ``size``, ``n``, ``axes``, ``pow_q`` and calls
    ``word_index`` / ``index_word``; here a "word" is a one-digit row
    ``(idx,)``, so ``index_word`` returns an ``(m, 1)`` int64 array and
    ``word_index`` flattens such rows back to indices. Anything the core
    derives from digit distance (cluster/hole ruins, the shell
    neighbourhood, batched peel independence) therefore degenerates to
    "equal or not", which is the honest answer for a set without
    geometry; the exact passes — greedy, low-loss ruin, LNS, descent —
    are unaffected.
    """

    def __init__(self, size):
        self.size = int(size)
        if self.size < 1:
            raise ValueError("IncidenceSpace needs at least one candidate")
        self.n = 1
        self.axes = [self.size]
        self.q = self.size
        self.pow_q = np.array([1], dtype=np.int64)
        self._ax = np.array([self.size], dtype=np.int64)

    def word_index(self, words):
        """``(m, 1)`` (or 1-D) index rows -> int64 indices."""
        return np.asarray(words, dtype=np.int64).reshape(-1)

    def index_word(self, idx):
        """int64 indices -> ``(m, 1)`` int64 rows."""
        return np.asarray(idx, dtype=np.int64).reshape(-1, 1)

    def all_words(self):
        return np.arange(self.size, dtype=np.int64)[:, None]

    def __repr__(self):
        return f"IncidenceSpace({self.size})"


class SparseIncidenceBackend(DenseFieldBackend):
    """Host backend over an explicit incidence matrix.

    Parameters
    ----------
    inc : (n_cand, d) integer array
        Row ``c`` lists the universe elements candidate ``c`` covers.
        Entries within a row are expected distinct (a candidate covers an
        element once); rows need not be sorted. Kept as given (int32 when
        the universe allows it), no copy when already contiguous int32.
    n_univ : int
        Universe size; every entry of ``inc`` must lie in ``[0, n_univ)``.
    mu : int
        Covering multiplicity (``cnt < mu`` is deficient).
    extract_cap, seed_shuffle
        As in the other backends.
    chunk_elems : int
        Elements per gather temporary in the chunked passes.
    """

    indep_dist = None
    supports_hist = True

    def __init__(self, inc, n_univ, mu=1, extract_cap=1 << 22,
                 seed_shuffle=None, chunk_elems=_CHUNK_ELEMS):
        inc = np.asarray(inc)
        if inc.ndim != 2 or not np.issubdtype(inc.dtype, np.integer):
            raise ValueError("inc must be a 2-D integer array (n_cand, d)")
        self.n_univ = int(n_univ)
        if self.n_univ < 1:
            raise ValueError("n_univ must be >= 1")
        if inc.size and (int(inc.min()) < 0 or int(inc.max()) >= self.n_univ):
            raise ValueError("incidence entries must lie in [0, n_univ)")
        dtype = np.int32 if self.n_univ <= np.iinfo(np.int32).max else np.int64
        self.inc = np.ascontiguousarray(inc, dtype=dtype)
        self.n_cand, self.d = (int(x) for x in self.inc.shape)
        self.mu = int(mu)
        if self.mu < 1:
            raise ValueError("mu must be >= 1")
        self.space = IncidenceSpace(self.n_cand)
        self.extract_cap = extract_cap
        self.cnt = np.zeros(self.n_univ, dtype=np.int32)
        self.map = np.zeros(self.n_cand, dtype=np.int32)
        self._rows = max(1, int(chunk_elems) // max(self.d, 1))
        self._rng = (np.random.default_rng(seed_shuffle)
                     if seed_shuffle is not None else None)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _idx(idx):
        return np.atleast_1d(np.asarray(idx, dtype=np.int64)).ravel()

    def ball_indices(self, w):
        """Universe elements covered by candidate ``w`` (sorted)."""
        return np.sort(self.inc[int(w)].astype(np.int64))

    def _scatter(self, idx, delta, out):
        """``out[u] += delta`` for every incidence ``(c, u)``, ``c`` in idx.

        A small batch touches few elements and goes through ``add.at``;
        a large one is a ``bincount`` over the universe, which is what a
        full recount amounts to."""
        rows = self._rows
        for lo in range(0, len(idx), rows):
            flat = self.inc[idx[lo:lo + rows]].ravel()
            if len(flat) * 8 < self.n_univ:
                np.add.at(out, flat, delta)
            else:
                bc = np.bincount(flat, minlength=self.n_univ)
                np.add(out, bc * delta, out=out, casting="unsafe")

    def _field(self, mask):
        """``map[c] = #{u in row c : mask[u]}`` for every candidate."""
        m8 = mask.view(np.uint8) if mask.dtype == np.bool_ else mask.astype(
            np.uint8)
        rows = self._rows
        for lo in range(0, self.n_cand, rows):
            hi = min(self.n_cand, lo + rows)
            self.map[lo:hi] = np.take(m8, self.inc[lo:hi]).sum(
                axis=1, dtype=np.int32)

    def _role_of(self, values, target):
        """The coverage-role predicate on gathered cnt values (identical
        to indexing :meth:`_role_mask` at those positions)."""
        if target == 0:
            return values < self.mu
        if target == 1:
            return (values >= 1) & (values <= self.mu)
        return values == target

    # ------------------------------------------------------------ fields
    def recount(self, idx):
        self.cnt[:] = 0
        self._scatter(self._idx(idx), 1, self.cnt)

    def gain_map(self):
        self._field(self._deficient())

    def loss_map(self):
        self._field(self._critical())

    # ------------------------------------------------------- incremental
    def ball_update(self, idx, delta):
        self._scatter(self._idx(idx), int(delta), self.cnt)

    def ball_gather(self, idx, target):
        """target is a coverage ROLE: 0 = deficient (cnt < mu), 1 =
        critical (1 <= cnt <= mu), anything else = literal cnt ==
        target. At mu=1 all three are the literal count."""
        idx = self._idx(idx)
        out = np.empty(len(idx), dtype=np.int32)
        if not len(idx):
            return out
        wide = len(idx) * self.d * _MASK_BREAKEVEN >= self.n_univ
        m8 = self._role_mask(int(target)).view(np.uint8) if wide else None
        rows = self._rows
        for lo in range(0, len(idx), rows):
            sub = self.inc[idx[lo:lo + rows]]
            hit = (np.take(m8, sub) if wide
                   else self._role_of(np.take(self.cnt, sub), int(target)))
            out[lo:lo + rows] = hit.sum(axis=1, dtype=np.int32)
        return out

    # ---------------------------------------------------------- queries
    def read_cnt(self, off=0, length=None):
        """The multiplicity field is over the *universe*, so ``off`` and
        ``length`` address universe elements, not candidates."""
        length = length if length is not None else self.n_univ - off
        return self.cnt[off:off + length].astype(np.uint16)

    def deficient_universe(self, limit, rng=None):
        """Up to ``limit`` universe elements with ``cnt < mu`` (sorted; a
        random subset of them when ``rng`` is given)."""
        u = np.flatnonzero(self.cnt < self.mu)
        if rng is not None and len(u) > limit:
            u = np.sort(rng.choice(u, size=int(limit), replace=False))
        return u[:limit].astype(np.int64)

    # ----------------------------------------------------- verification
    def coverage(self, idx):
        """Fresh multiplicity vector of the solution ``idx``, computed
        from the incidence list alone (no incremental state)."""
        cov = np.zeros(self.n_univ, dtype=np.int64)
        self._scatter(self._idx(idx), 1, cov)
        return cov

    def verify(self, idx):
        """True iff ``idx`` is a feasible mu-fold cover, recomputed from
        scratch and without reading ``cnt`` or ``map``."""
        return bool((self.coverage(idx) >= self.mu).all())

    def __repr__(self):
        return (f"{type(self).__name__}(n_cand={self.n_cand}, d={self.d}, "
                f"n_univ={self.n_univ}, mu={self.mu})")


class WeightedIncidenceBackend(DenseFieldBackend):
    """Host backend over a weighted, variable-length incidence (CSR).

    Candidate ``c`` covers universe element ``indices[j]`` with weight
    ``w[j]`` for ``j`` in ``indptr[c]:indptr[c+1]``; universe element
    ``u`` has weight ``uw[u]``. Choosing ``c`` adds ``w`` to ``cnt[u]``,
    and the deficiency is ``sum_u uw[u] * max(0, mu - cnt[u])`` — the
    quotient of a covering design by a group (``problems.design_orbits``)
    is exactly this: candidates are block orbits, universe elements are
    t-subset orbits, ``w`` the number of blocks of the orbit through a
    fixed representative t-subset, ``uw`` the t-orbit size.

    With ``w >= 1`` a single candidate may satisfy several units of a
    mu-fold requirement at once, so the fields are the exact weighted
    marginals rather than role counts:

      gain(c) = sum_j uw * min(w_j, max(0, mu - cnt))            (target 0)
      loss(c) = sum_j uw * (max(0, mu - cnt + w_j) - max(0, mu - cnt))
                       over the elements with cnt >= w_j             (target 1)

    both of which reduce to the deficient / critical role counts when
    every weight is 1 (the ``cnt >= w`` gate is what makes the loss of a
    candidate *outside* the solution — elements it never covered — agree
    with the critical role; at an incumbent every element has
    ``cnt >= w`` and the gate is void). Other ``ball_gather`` targets are literal
    (unweighted) ``cnt == target`` counts. ``count_eq`` is literal;
    ``uncovered`` is the weighted deficiency.
    """

    indep_dist = None
    supports_hist = True

    def __init__(self, indptr, indices, w, n_univ, uw=None, mu=1,
                 extract_cap=1 << 22, seed_shuffle=None,
                 chunk_elems=_CHUNK_ELEMS):
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64).ravel()
        self.indices = np.ascontiguousarray(indices, dtype=np.int32).ravel()
        self.w = np.ascontiguousarray(w, dtype=np.int32).ravel()
        self.n_univ = int(n_univ)
        if self.n_univ < 1 or (self.indices.size and (
                int(self.indices.min()) < 0
                or int(self.indices.max()) >= self.n_univ)):
            raise ValueError("indices must lie in [0, n_univ)")
        if len(self.indices) != len(self.w) or len(self.indptr) < 1:
            raise ValueError("indices / w length mismatch")
        if self.indptr[0] != 0 or self.indptr[-1] != len(self.indices) or (
                np.diff(self.indptr) < 0).any():
            raise ValueError("indptr is not a valid CSR offset array")
        if (self.w < 1).any():
            raise ValueError("weights must be >= 1")
        self.uw = (np.ones(self.n_univ, dtype=np.int32) if uw is None
                   else np.ascontiguousarray(uw, dtype=np.int32).ravel())
        if len(self.uw) != self.n_univ or (self.uw < 1).any():
            raise ValueError("uw must have n_univ entries >= 1")
        self.n_cand = len(self.indptr) - 1
        self.nnz = len(self.indices)
        self.row_len = np.diff(self.indptr)
        self.d = None
        self.mu = int(mu)
        if self.mu < 1:
            raise ValueError("mu must be >= 1")
        self.space = IncidenceSpace(self.n_cand)
        self.extract_cap = extract_cap
        self.cnt = np.zeros(self.n_univ, dtype=np.int32)
        self.map = np.zeros(self.n_cand, dtype=np.int32)
        self._chunk_elems = int(chunk_elems)
        self._rng = (np.random.default_rng(seed_shuffle)
                     if seed_shuffle is not None else None)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _idx(idx):
        return np.atleast_1d(np.asarray(idx, dtype=np.int64)).ravel()

    def ball_indices(self, w):
        s, e = self.indptr[int(w)], self.indptr[int(w) + 1]
        return np.sort(self.indices[s:e].astype(np.int64))

    def _flat(self, idx):
        """Flat (indices, weights) of the rows in ``idx``."""
        if not len(idx):
            return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))
        lens = self.row_len[idx]
        starts = self.indptr[idx]
        pos = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])),
                        lens) + np.arange(int(lens.sum()))
        return self.indices[pos], self.w[pos]

    def _scatter(self, idx, delta, out):
        step = max(1, self._chunk_elems // max(int(self.row_len.max(initial=1)), 1))
        for lo in range(0, len(idx), step):
            flat, wf = self._flat(idx[lo:lo + step])
            if len(flat) * 8 < self.n_univ:
                np.add.at(out, flat, wf.astype(np.int64) * delta)
            else:
                bc = np.bincount(flat, weights=wf, minlength=self.n_univ)
                np.add(out, np.rint(bc).astype(np.int64) * delta, out=out,
                       casting="unsafe")

    def _row_values(self, flat, wf, target):
        """Per-incidence contribution for a coverage role / literal target."""
        c = self.cnt[flat].astype(np.int64)
        if target == 0:
            return self.uw[flat] * np.minimum(wf, np.maximum(0, self.mu - c))
        if target == 1:
            return (self.uw[flat] * (np.maximum(0, self.mu - c + wf)
                                     - np.maximum(0, self.mu - c))
                    * (c >= wf))
        return (c == target).astype(np.int64)

    def _segment_sums(self, idx, target):
        """Row sums of :meth:`_row_values` over the rows ``idx``."""
        out = np.zeros(len(idx), dtype=np.int64)
        step = max(1, self._chunk_elems // max(int(self.row_len.max(initial=1)), 1))
        for lo in range(0, len(idx), step):
            sub = idx[lo:lo + step]
            lens = self.row_len[sub]
            flat, wf = self._flat(sub)
            vals = self._row_values(flat, wf, target)
            starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
            nz = lens > 0
            if nz.any():
                seg = np.add.reduceat(vals, starts[nz]) if len(vals) else []
                out[lo:lo + step][nz] = seg
        return out

    # ------------------------------------------------------------ fields
    def recount(self, idx):
        self.cnt[:] = 0
        self._scatter(self._idx(idx), 1, self.cnt)

    def _field(self, target):
        self.map[:] = self._segment_sums(np.arange(self.n_cand), target)

    def gain_map(self):
        self._field(0)

    def loss_map(self):
        self._field(1)

    def ball_update(self, idx, delta):
        self._scatter(self._idx(idx), int(delta), self.cnt)

    def ball_gather(self, idx, target):
        idx = self._idx(idx)
        return self._segment_sums(idx, int(target)).astype(np.int32)

    # ---------------------------------------------------------- queries
    def uncovered(self):
        return int((self.uw.astype(np.int64)
                    * np.maximum(self.mu - self.cnt, 0)).sum())

    def read_cnt(self, off=0, length=None):
        length = length if length is not None else self.n_univ - off
        return self.cnt[off:off + length].astype(np.uint16)

    def deficient_universe(self, limit, rng=None):
        u = np.flatnonzero(self.cnt < self.mu)
        if rng is not None and len(u) > limit:
            u = np.sort(rng.choice(u, size=int(limit), replace=False))
        return u[:limit].astype(np.int64)

    # ----------------------------------------------------- verification
    def coverage(self, idx):
        cov = np.zeros(self.n_univ, dtype=np.int64)
        self._scatter(self._idx(idx), 1, cov)
        return cov

    def verify(self, idx):
        return bool((self.coverage(idx) >= self.mu).all())

    def __repr__(self):
        return (f"{type(self).__name__}(n_cand={self.n_cand}, nnz={self.nnz}, "
                f"n_univ={self.n_univ}, mu={self.mu})")

"""Group-invariant covering designs: the quotient problem and its release.

Given C(v,k,t) and a permutation group G on the points
(``perm_groups``), a G-invariant block set is a union of block orbits.
The search over such sets is the ordinary search core run on a
**quotient** incidence backend:

  candidates   block orbits O (by default only the full-length ones,
               |O| = |G|; ``short_orbits=True`` admits the others)
  universe     t-subset orbits U, weighted by |U|
  weight       w(O,U) = number of blocks of O containing a fixed
               representative of U (the same for every member of U,
               since G is transitive on U)

so that ``cnt(U)`` in the quotient equals the full ``cnt`` at any member
of U, and the weighted deficiency ``sum_U |U| max(0, mu - cnt(U))``
equals the full deficiency. The search core's unit of cost is one
candidate, i.e. one orbit: with full-length orbits only, ``M`` orbits are
exactly ``M |G|`` blocks and the notch ladder moves in steps of |G|
blocks — the same step the product-space orbit layer takes
(``symmetry.SymmetricSearcher.descend`` steps by the smallest orbit).
With short orbits admitted, the reported objective is still the true
block count ``sum |O|``, but the core weighs a short orbit like a full
one (no gain-per-block ratio in the greedy: the product-space layer has
no such ratio either, and the core's deficit bookkeeping needs
``ball_gather`` to be the exact gain, so a scaled field is not an
option). That is the documented simplification.

Release: ``expand`` turns selected orbits into block indices, and
``free_searcher`` hands them to a plain :class:`~coldcore.search.Searcher`
on the full :class:`CoveringDesignBackend`, exactly as
``SymmetricSearcher.free_searcher`` does for product spaces. The plain
block list is what every verifier and the LJCR file format accept.
"""
import time
from math import comb

import numpy as np

from ..incidence import WeightedIncidenceBackend
from .covering_design import (CoveringDesignBackend, all_blocks,
                              build_incidence)
from .perm_groups import PermGroup, parse_design_group

#: ``Searcher.descend`` will not start a notch it cannot finish and keeps
#: this many seconds of its deadline back for that check (see
#: docs/symmetry.md, "A methodology note"); the drivers here add it to
#: the budget they are given so the budget is working time.
DESCENT_RESERVE_S = 30.0


class DesignOrbits:
    """Block orbits, t-orbits and the quotient weights of C(v,k,t) under
    a group ``G`` (a :class:`PermGroup` or a spec string)."""

    def __init__(self, v, k, t, group, blocks=None, inc=None):
        self.v, self.k, self.t = int(v), int(k), int(t)
        if isinstance(group, str):
            group = parse_design_group(v, group)
        if not isinstance(group, PermGroup) or group.v != self.v:
            raise ValueError("group must be a PermGroup on v points")
        self.group = group
        self.order = group.order()
        self.blocks = all_blocks(v, k) if blocks is None else blocks
        self.block_part = group.subset_orbits(k, self.blocks)
        self.univ_part = group.subset_orbits(t)
        self.n_univ = comb(v, t)
        # pair counts P(O,U) = #{(B,T): B in O, T in U, T subset B} from
        # the full incidence; w(O,U) = P / |U| is exact by transitivity
        inc = build_incidence(self.blocks, v, t) if inc is None else inc
        n_uorb = len(self.univ_part)
        ulab = self.univ_part.labels
        blab = self.block_part.labels
        step = max(1, (1 << 24) // max(inc.shape[1], 1))
        parts, cnts = [], []
        for lo in range(0, len(inc), step):
            kk = (blab[lo:lo + step, None] * n_uorb
                  + ulab[inc[lo:lo + step]]).ravel()
            u, c = np.unique(kk, return_counts=True)
            parts.append(u)
            cnts.append(c)
        keys, inv = np.unique(np.concatenate(parts), return_inverse=True)
        P = np.bincount(inv.ravel(), weights=np.concatenate(cnts),
                        minlength=len(keys))
        P = np.rint(P).astype(np.int64)
        orb = keys // n_uorb
        U = keys % n_uorb
        w = P // self.univ_part.sizes[U]
        if (w * self.univ_part.sizes[U] != P).any():
            raise AssertionError("pair counts not divisible by |U|")
        order = np.argsort(orb, kind="stable")
        self._O, self._U = orb[order], U[order]
        self._w = w[order].astype(np.int32)
        self._indptr = np.concatenate(
            ([0], np.cumsum(np.bincount(self._O, minlength=len(self.block_part))))
        ).astype(np.int64)

    # -- the quotient --
    def candidate_orbits(self, short_orbits=False):
        """Orbit ids admitted as candidates (full-length unless asked)."""
        if short_orbits:
            return np.arange(len(self.block_part), dtype=np.int64)
        return np.flatnonzero(self.block_part.sizes == self.order)

    def quotient_csr(self, short_orbits=False):
        """``(orbit_ids, indptr, indices, w, uw)`` of the quotient
        incidence restricted to the candidate orbits."""
        cand = self.candidate_orbits(short_orbits)
        lens = np.diff(self._indptr)[cand]
        starts = self._indptr[cand]
        pos = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])),
                        lens) + np.arange(int(lens.sum()))
        indptr = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
        return (cand, indptr, self._U[pos].astype(np.int32), self._w[pos],
                self.univ_part.sizes.astype(np.int32))

    def expand(self, cand_idx, cand=None):
        """Quotient candidate indices -> sorted block indices."""
        cand = self.candidate_orbits() if cand is None else cand
        oid = cand[np.asarray(cand_idx, dtype=np.int64)]
        if not len(oid):
            return np.zeros(0, dtype=np.int64)
        return np.sort(np.concatenate([self.block_part.orbit(o) for o in oid]))

    def block_count(self, cand_idx, cand=None):
        cand = self.candidate_orbits() if cand is None else cand
        return int(self.block_part.sizes[cand[np.asarray(cand_idx,
                                                          dtype=np.int64)]].sum())

    def summary(self):
        bs, us = self.block_part.sizes, self.univ_part.sizes
        return (f"{self.group.name} |G|={self.order}: {len(bs)} block orbits "
                f"({int((bs == self.order).sum())} full), {len(us)} t-orbits, "
                f"quotient nnz={len(self._w)}")


class QuotientDesignBackend(WeightedIncidenceBackend):
    """The quotient of C(v,k,t) by ``G`` as a host backend.

    Attributes: ``orbits`` (the :class:`DesignOrbits`), ``cand`` (orbit
    id of every candidate), ``expand(idx)`` / ``block_count(idx)``."""

    def __init__(self, v, k, t, group, mu=1, short_orbits=False,
                 orbits=None, **kw):
        self.orbits = (DesignOrbits(v, k, t, group) if orbits is None
                       else orbits)
        cand, indptr, indices, w, uw = self.orbits.quotient_csr(short_orbits)
        self.cand = cand
        self.short_orbits = bool(short_orbits)
        WeightedIncidenceBackend.__init__(self, indptr, indices, w,
                                          len(self.orbits.univ_part), uw=uw,
                                          mu=mu, **kw)
        self.v, self.k, self.t = self.orbits.v, self.orbits.k, self.orbits.t

    def expand(self, idx):
        return self.orbits.expand(idx, self.cand)

    def block_count(self, idx):
        return self.orbits.block_count(idx, self.cand)

    @property
    def label(self):
        return (f"C({self.v},{self.k},{self.t})/{self.orbits.group.name}"
                + (f"mu{self.mu}" if self.mu > 1 else ""))


class _GroupAdapter:
    """What :class:`coldcore.symmetry.SymmetricSearcher` reads off a group:
    the backend's space and an ``orbits()`` partition — here the block
    orbits over the candidate index set of a full design backend."""

    def __init__(self, space, part, name):
        self.space, self._part, self.name = space, part, name

    def orbits(self):
        return self._part


def orbit_search_points(v, k, t, group, budget_s=60.0, mu=1, seed=1,
                        device="cpu", on_log=None, on_solved=None, floor=1,
                        notch_budget_s=None, orbits=None, max_orbit=None):
    """The product-space precedent, verbatim: ``SymmetricSearcher`` over
    the block orbits of the *full* design backend. It counts blocks, not
    orbits — the greedy skips orbits larger than the remaining room, the
    descent steps by the smallest orbit size — so short orbits carry
    their true cost (C(9,3,2) = 12 = 9 + 3 under Z_3 x Z_3, C(15,3,2) =
    35 = 15 + 15 + 5 under Z_15). Every orbit gain is one whole-orbit
    ``ball_update`` on the full backend plus a deficit readback, so it is
    the slower of the two drivers on large cells. Returns
    ``(blocks, searcher, backend)`` with the backend closed."""
    from ..symmetry import SymmetricSearcher
    orb = DesignOrbits(v, k, t, group) if orbits is None else orbits
    b = CoveringDesignBackend(v, k, t, mu=mu, device=device)
    G = _GroupAdapter(b.space, orb.block_part, orb.group.name)
    best = {"blocks": np.zeros(0, dtype=np.int64), "n": None}

    def solved(M, idx):
        if best["n"] is None or M < best["n"]:
            best["blocks"], best["n"] = np.asarray(idx, dtype=np.int64).copy(), M
            if on_solved is not None:
                on_solved(M, best["blocks"])

    s = SymmetricSearcher(b, G, seed=seed, deadline=time.time() + budget_s,
                          on_log=on_log, on_solved=solved, max_orbit=max_orbit)
    try:
        s.greedy_fill(10 ** 9)
        if s.uncov != 0:
            return best["blocks"], s, b
        s.peel()
        nb = notch_budget_s or max(1.0, budget_s / 6)
        s.descend(floor=floor, notch_budget_s=nb)
        solved(len(s.code), s.code)
    finally:
        b.close()
    return best["blocks"], s, b


def orbit_search(v, k, t, group, budget_s=60.0, mu=1, seed=1,
                 short_orbits=False, device="cpu", on_log=None,
                 on_solved=None, floor=1, notch_budget_s=None, orbits=None):
    """Greedy -> peel -> notch descent over block orbits, on the quotient
    backend (``mode="quotient"`` of the CLI).

    Returns ``(blocks, searcher, quotient)``: the best *verified*
    G-invariant block set found (sorted block indices, may be empty when
    no union of admitted orbits covers), the quotient searcher and the
    quotient backend (closed). ``on_solved(n_blocks, blocks)`` is called
    for every verified improvement, with the block count. ``budget_s`` is
    working time (the descent's reserve is added on top)."""
    from ..search import Searcher
    if device == "gpu":
        from ..incidence_gpu import GpuWeightedIncidenceBackend
        orb = DesignOrbits(v, k, t, group) if orbits is None else orbits
        cand, indptr, indices, w, uw = orb.quotient_csr(short_orbits)
        q = GpuWeightedIncidenceBackend(indptr, indices, w,
                                        len(orb.univ_part), uw=uw, mu=mu)
        q.orbits, q.cand = orb, cand
        q.expand = lambda idx: orb.expand(idx, cand)
        q.block_count = lambda idx: orb.block_count(idx, cand)
    else:
        q = QuotientDesignBackend(v, k, t, group, mu=mu,
                                  short_orbits=short_orbits, orbits=orbits)
    log = on_log or (lambda m: None)
    best = {"blocks": np.zeros(0, dtype=np.int64), "n": None}

    def solved(M, idx):
        blocks = q.expand(idx)
        n = len(blocks)
        if best["n"] is None or n < best["n"]:
            best["blocks"], best["n"] = blocks, n
            if on_solved is not None:
                on_solved(n, blocks)

    deadline = time.time() + budget_s + DESCENT_RESERVE_S
    s = Searcher(q, seed=seed, deadline=deadline, on_log=log, on_solved=solved)
    try:
        s.load(np.zeros(0, dtype=np.int64))
        s.greedy_fill(10 ** 9)
        if s.uncov != 0:
            log(f"orbit search {q.label}: no union of admitted orbits covers "
                f"(deficit {s.uncov})")
            return best["blocks"], s, q
        nb = notch_budget_s or max(1.0, budget_s / 6)
        s.descend(floor=floor, notch_budget_s=nb, step0=1)
    finally:
        q.close()
    return best["blocks"], s, q


def release(v, k, t, blocks, budget_s, mu=1, seed=1, device="cpu",
            on_log=None, on_solved=None, floor=1, notch_budget_s=None,
            **descend_kw):
    """Free polish: seed a plain :class:`~coldcore.search.Searcher` on the
    full :class:`CoveringDesignBackend` with the expanded blocks (the
    structured-start / free-polish hybrid of ``SymmetricSearcher
    .free_searcher``) and run the notch descent for ``budget_s``.
    Returns ``(blocks, searcher)``; the backend is closed."""
    from ..search import Searcher
    b = CoveringDesignBackend(v, k, t, mu=mu, device=device)
    best = {"blocks": np.asarray(blocks, dtype=np.int64)}

    def solved(M, idx):
        best["blocks"] = np.asarray(idx, dtype=np.int64).copy()
        if on_solved is not None:
            on_solved(M, best["blocks"])

    s = Searcher(b, seed=seed, deadline=time.time() + budget_s
                 + DESCENT_RESERVE_S,
                 on_log=on_log or (lambda m: None), on_solved=solved)
    try:
        s.load(best["blocks"])
        if s.uncov != 0:
            raise ValueError("released solution is not feasible")
        nb = notch_budget_s or max(1.0, budget_s / 6)
        s.descend(floor=floor, notch_budget_s=nb, **descend_kw)
    finally:
        b.close()
    return best["blocks"], s

"""Isomorph-free exhaustive settlement of tiny cells.

The search core is a *heuristic*: it finds covers, never proves that none
smaller exists. This module closes the other half for cells small enough
to enumerate — it decides ``K(...) <= M`` exhaustively, and returns
either a witness or a proof of infeasibility.

The generator
-------------
Three ingredients, in the order they matter:

1. **Coverage branching (completeness).** At a node with partial code P,
   take ``u* = min`` of the words that are still short of their coverage
   requirement. *Every* completion of P must contain a codeword whose
   ball reaches ``u*``, so branching over ``coverers(u*)`` is exhaustive.
   No ordering assumption, no wasted branches on words that cannot
   possibly help.

2. **Stabiliser-reduced branching (isomorph rejection).** Let
   ``H = Stab_G(P)`` be the setwise stabiliser of the partial code inside
   an automorphism group ``G`` of the coverage relation, and
   ``H* = Stab_H(u*)``. If ``C`` is any completion of P and
   ``c in C ∩ coverers(u*)``, then for every ``h in H*`` the image
   ``h.C`` is *also* a completion of P of the same size, and it contains
   ``h.c in coverers(u*)``. So the branch set may be cut down to one
   representative per ``H*``-orbit — exactly, with no completion lost.
   This is the classical isomorph-free generation of covering designs
   (Östergård et al.), and it is the reason the whole thing is tractable
   at all: at the root ``H = G`` and a transitive ``G`` collapses the
   first level from ``|X|`` branches to one.

   *Orderly generation* (accept only lex-minimal partial codes) was the
   alternative. It is hereditary under "drop the largest element", but
   coverage branching does not add elements in increasing order, so a
   canonical partial code is not reachable along a canonical path;
   making the two fit needs McKay-style canonical augmentation over
   *all* augmentations, which throws away pruning (1) — the pruning that
   actually pays here. Stabiliser reduction keeps both.

3. **Counting bounds (pruning).** With ``r = M - |P|`` codewords left and
   a deficit ``D``, one codeword removes at most ``|B|`` of it, so
   ``r*|B| >= D`` is necessary; at ``r <= 1`` the exact test (is there a
   single ball containing everything still missing?) is run instead.

Guarantees
----------
``decide(M)`` returns ``"feasible"`` with a witness, ``"infeasible"``
(an exhaustive proof, up to the isomorph rejection above), or
``"unknown"`` when a node/time limit stopped it first. ``G`` must be a
group of automorphisms of the coverage relation; the constructor
verifies that by brute force on the space, so a wrong group is an error,
never a wrong answer.
"""
import time

import numpy as np

from .symmetry import Group, isometry_group, monomial_group

#: Refuse to enumerate a group bigger than this by default.
MAX_GROUP = 1 << 17


#: Refuse to build an index-permutation table bigger than this.
MAX_TABLE_BYTES = 24 << 30


def all_perms(q):
    """Every permutation of ``Z_q`` — the full ``S_q``, ``q!`` rows."""
    from itertools import permutations
    return np.array(list(permutations(range(q))), dtype=np.int64)


def affine_perms(q):
    """``x -> a*x + b`` on ``Z_q``, ``gcd(a, q) = 1`` — ``AGL(1,q)``.

    ``q*phi(q)`` rows (20 for q=5) instead of ``q!`` (120). The subgroup
    to fall back on when the full monomial table will not fit: every row
    is still a genuine alphabet permutation, hence a Hamming isometry."""
    from math import gcd
    x = np.arange(q, dtype=np.int64)
    units = [a for a in range(1, q) if gcd(a, q) == 1]
    return np.array([(a * x + b) % q for a in units for b in range(q)],
                    dtype=np.int64)


def monomial_table(space, per_axis=None, coord=None,
                   max_bytes=MAX_TABLE_BYTES):
    """Index-permutation table of a monomial group of a product space.

    The monomial (wreath) group of ``Z_q^n`` is ``S_q^n |x| S_n``, of
    order ``(q!)^n * n!`` — 10 321 920 for ``Z_2^8``, 7 962 624 for
    ``Z_4^4``. That is far past what :meth:`Group.elements` can build one
    Python object at a time, so the table of induced *index*
    permutations is constructed directly, in two broadcast steps:

    * the coordinate part, ``G_p[x]`` = the index whose digit ``j`` is
      digit ``p_j`` of ``x``;
    * the value part, ``V_v[x]`` = the index whose digit ``j`` is
      ``v_j`` applied to digit ``j`` of ``x``, accumulated one axis at a
      time over the product of the per-axis permutation sets;

    and the group element ``(g.x)_j = v_j[x_{p_j}]`` is their
    composition, ``table[(v, p)] = V_v[G_p]``.

    ``per_axis`` is a list of ``(k_j, a_j)`` permutation arrays, one per
    axis (default: all ``a_j!``); ``coord`` is a ``(m, n)`` array of
    coordinate permutations (default: all ``n!``, and only the ones that
    preserve axis sizes). Pass :func:`affine_perms` for ``per_axis`` when
    the full table will not fit.

    Returns ``(table, reps)`` with ``reps`` the orbit representatives of
    the group on the space, which is what the root of the exhaustive
    search branches over. Raises when the table would exceed
    ``max_bytes``; the cost is ``|G| * |X| * 4`` bytes.
    """
    from itertools import permutations
    n, size = space.n, space.size
    axes = list(space.axes)
    pw = np.asarray(space.pow_q, dtype=np.int64)
    x = np.arange(size, dtype=np.int64)
    digits = (x[:, None] // pw[None, :]) % np.array(axes)[None, :]

    if per_axis is None:
        per_axis = [all_perms(a) for a in axes]
    per_axis = [np.asarray(v, dtype=np.int64) for v in per_axis]
    for j, v in enumerate(per_axis):
        if v.ndim != 2 or v.shape[1] != axes[j]:
            raise ValueError(f"per_axis[{j}] must be (k, {axes[j]})")

    if coord is None:
        coord = np.array([p for p in permutations(range(n))
                          if all(axes[p[j]] == axes[j] for j in range(n))],
                         dtype=np.int64)
    coord = np.asarray(coord, dtype=np.int64).reshape(-1, n)

    n_val = 1
    for v in per_axis:
        n_val *= len(v)
    total = n_val * len(coord)
    need = total * size * 4
    if need > max_bytes:
        raise ValueError(
            f"monomial table would need {need / 2**30:.1f} GiB "
            f"(|G|={total}, |X|={size}); pass a smaller per_axis/coord "
            f"(see affine_perms) or raise max_bytes")

    # value part, accumulated axis by axis
    V = np.zeros((1, size), dtype=np.int64)
    for j, v in enumerate(per_axis):
        add = v[:, digits[:, j]] * int(pw[j])           # (k_j, size)
        V = (V[:, None, :] + add[None, :, :]).reshape(-1, size)
    # coordinate part
    G = (digits[:, coord].transpose(1, 0, 2) * pw[None, None, :]).sum(-1)
    table = V[:, G].reshape(-1, size).astype(np.int32)

    # orbit representatives: the orbit of a word is just the column's
    # distinct images, so this is a handful of uniques, not a partition
    # sweep. A group containing all translations is transitive and this
    # returns after one.
    reps, seen = [], np.zeros(size, dtype=bool)
    while not seen.all():
        w = int(np.flatnonzero(~seen)[0])
        reps.append(w)
        seen[np.unique(table[:, w])] = True
    return table, np.array(reps, dtype=np.int64)


def binary_monomial_table(space):
    """The full monomial group of ``Z_2^n`` — :func:`monomial_table`
    specialised to the binary case (order ``n! * 2^n``)."""
    if list(space.axes) != [2] * space.n:
        raise ValueError("binary spaces only")
    return monomial_table(space)


def _family_of(backend):
    """Best-effort problem family name for a reference backend."""
    cls = type(backend).__name__.lower()
    for fam in ("lee", "torus", "grid"):
        if fam in cls:
            return {"torus": "torus_linf", "grid": "grid_linf"}.get(fam, fam)
    return "hamming"


class IsomorphFreeSearch:
    """Exhaustive decision procedure for one tiny cell.

    Parameters
    ----------
    backend : a reference backend (needs ``ball_indices``)
    group   : automorphism group used for isomorph rejection. ``None``
              picks the full isometry group of the family and falls back
              to the translation subgroup when that is too large.
    iso_depth : reduce branch sets only while ``|P| < iso_depth``; below
              that the stabiliser is almost always trivial and testing
              for it costs more than it saves.
    verify  : brute-force check that every generator really is an
              automorphism (O(|X|^2) per generator; on by default).
    strong_bound : run the exact greedy bound whenever the free counting
              bound is within this many codewords of failing. 0 (the
              default) disables it; see the note on ``_best_drop`` for
              why it does not pay.
    """

    def __init__(self, backend, group=None, iso_depth=4,
                 max_group=MAX_GROUP, verify=True, strong_bound=0,
                 table=None):
        if not hasattr(backend, "ball_indices"):
            raise TypeError("exhaustive settlement needs a reference backend")
        self.b = backend
        self.space = backend.space
        self.size = self.space.size
        self.mu = int(getattr(backend, "mu", 1))
        self.iso_depth = int(iso_depth)
        self.strong_bound = float(strong_bound)
        # ---- balls, as bitmasks and as index lists (bit i of balls[w] is
        # set iff w covers word i; coverers is the transpose relation)
        self.ball_idx = [np.asarray(backend.ball_indices(w), dtype=np.int64)
                         for w in range(self.size)]
        self.balls = [_mask(ix, self.size) for ix in self.ball_idx]
        ws = np.repeat(np.arange(self.size, dtype=np.int64),
                       [len(ix) for ix in self.ball_idx])
        us = np.concatenate(self.ball_idx) if self.size else ws
        order = np.argsort(us, kind="stable")
        us, ws = us[order], ws[order]
        cuts = np.searchsorted(us, np.arange(self.size + 1, dtype=np.int64))
        self.coverers = [np.sort(ws[cuts[u]:cuts[u + 1]])
                         for u in range(self.size)]
        self.maxball = max(int(m.bit_count()) for m in self.balls)
        self.full = (1 << self.size) - 1
        # ---- group
        if table is not None:
            # precomputed (index-permutation table, root orbit reps) --
            # the escape hatch for groups too large to enumerate as
            # Python objects, e.g. binary_monomial_table(). The caller
            # owns the promise that these really are automorphisms.
            self.gperm, self._root_reps = table
            self.gperm = np.ascontiguousarray(self.gperm, dtype=np.int32)
            self.group = group
            self.elements = range(len(self.gperm))
        else:
            self.group = self._pick_group(group, max_group)
            if verify and not self.group.is_automorphism_of(backend):
                raise ValueError(
                    "group is not an automorphism group of this coverage "
                    "relation; isomorph rejection would be unsound")
            self.elements = self.group.elements(cap=max_group)
            self.gperm = (np.array([g.permutation(self.space)
                                    for g in self.elements], dtype=np.int32)
                          if self.elements
                          else np.zeros((0, self.size), dtype=np.int32))
            self._root_reps = None
        self.stats = {"nodes": 0, "leaves": 0, "pruned_bound": 0,
                      "pruned_strong": 0, "pruned_iso": 0}

    def _pick_group(self, group, max_group):
        if group is not None:
            return group
        fam = _family_of(self.b)
        for cand in (isometry_group(self.space, fam),
                     monomial_group(self.space) if fam == "hamming" else None):
            if cand is None:
                continue
            try:
                cand.elements(cap=max_group)
                return cand
            except ValueError:
                continue
        # last resort: translations only (always small enough here)
        from .symmetry import value_shifts
        g = value_shifts(self.space)
        try:
            g.elements(cap=max_group)
            return g
        except ValueError:
            return Group(self.space, [], name="trivial")

    # ---------------------------------------------------------- the search
    def decide(self, M, node_limit=None, time_limit=None, log_every=0,
               on_log=None):
        """Is there a solution of size <= M?

        Returns ``(verdict, witness, stats)`` with verdict in
        ``{"feasible", "infeasible", "unknown"}``."""
        M = int(M)
        self.stats = {"nodes": 0, "leaves": 0, "pruned_bound": 0,
                      "pruned_strong": 0, "pruned_iso": 0, "M": M,
                      "secs": 0.0}
        self._deadline = (time.time() + time_limit) if time_limit else None
        self._node_limit = node_limit
        self._log_every = int(log_every)
        self._on_log = on_log or (lambda s: None)
        self._witness = None
        self._stopped = False
        t0 = time.time()
        stab0 = np.arange(len(self.elements), dtype=np.int64)
        if self.mu == 1:
            found = self._rec(0, self.full, [], M, stab0)
        else:
            found = self._rec_mu(0, (0,) * self.mu, self.mu * self.size,
                                 [], M, stab0)
        self.stats["secs"] = round(time.time() - t0, 3)
        self.stats["nodes_per_s"] = (
            round(self.stats["nodes"] / max(self.stats["secs"], 1e-3)))
        if found:
            return "feasible", np.array(self._witness, dtype=np.int64), \
                dict(self.stats)
        if self._stopped:
            return "unknown", None, dict(self.stats)
        return "infeasible", None, dict(self.stats)

    # -------- mu = 1: bitset path
    def _rec(self, depth, U, P, M, stab):
        self.stats["nodes"] += 1
        if self._budget_out():
            return False
        if U == 0:
            self._witness = list(P)
            return True
        r = M - len(P)
        if r <= 0:
            self.stats["leaves"] += 1
            return False
        need = U.bit_count()
        if r * self.maxball < need:
            self.stats["pruned_bound"] += 1
            return False
        if r * self.maxball < need + self.strong_bound * self.maxball:
            if r * self._best_drop(U) < need:
                self.stats["pruned_strong"] += 1
                return False
        u = (U & -U).bit_length() - 1          # lowest still-uncovered word
        if r == 1:                              # exact one-codeword test
            for c in self.coverers[u]:
                if U & ~self.balls[int(c)] == 0:
                    self._witness = list(P) + [int(c)]
                    return True
            self.stats["pruned_bound"] += 1
            return False
        cands, sub = self._branch_set(depth, u, P, stab)
        for k, c in enumerate(cands):
            c = int(c)
            if self._rec(depth + 1, U & ~self.balls[c], P + [c], M, sub[k]):
                return True
        return False

    # -------- the exact greedy bound (opt-in: measured a wash)
    #
    # Adding a codeword increments cnt by one on every word of its ball,
    # so it reduces the deficit by exactly |B ∩ short|, where ``short``
    # is the set of words still below their multiplicity requirement.
    # That is <= |B|, which is where the free bound ``r * maxball >= D``
    # comes from; the exact version replaces ``maxball`` with the largest
    # |B ∩ short| any single ball still achieves. Valid as a bound
    # because ``short`` only shrinks as codewords are added, so no future
    # codeword can beat today's maximum, and it is run only when the free
    # bound is within ``strong_bound`` codewords of failing.
    #
    # DEFAULT OFF, because it does not pay on either cell it was built
    # for. It is a whole-space scan, so it costs O(|X|) big-int ANDs per
    # node:
    #
    #   K(8,2,2), |X|=256   M=14: nodes 7.80e6 -> 6.70e6 (-14%), wall
    #                       13.1s -> 13.8s. A wash.
    #   K_8(4,2), |X|=4096  M=22: estimated tree 4.97e47 -> 3.23e44
    #                       (~1000x), throughput 1.15e6 -> 1.4e3 nodes/s
    #                       (~820x). Also a wash, and the wall-clock
    #                       verdict is unchanged (1e34 vs 7e33 years).
    #
    # Worth recording: the cheapest step toward "better node bounds" buys
    # a lot of tree and gives it all back in per-node cost. The bounds
    # that would actually move K_8(4,2) have to be much cheaper per node
    # than a whole-space rescan, or much stronger than greedy.
    def _best_drop(self, short):
        """max over w of the exact deficit drop of adding ball w."""
        best = 0
        for B in self.balls:
            d = (B & short).bit_count()
            if d > best:
                best = d
        return best

    # -------- mu > 1: layered bitset path
    #
    # State is mu bitmasks, ``lay[k]`` = the words already covered at
    # least k+1 times, so the deficit is ``mu*|X| - sum popcount(lay)``
    # and the still-deficient words are the complement of ``lay[mu-1]``.
    # Adding a ball ``B`` is
    #
    #     lay'[k] = lay[k] | (lay[k-1] & B),   lay[-1] = everything
    #
    # (a word reaches k+1 iff it was already there, or it was at k and B
    # covers it) -- mu big-int operations, using the OLD lower layer. At
    # mu = 1 this is exactly the single-mask rule of _rec, so the two
    # paths agree by construction; the numpy per-node count array this
    # replaces cost a whole-space copy per node.
    def _new_layers(self, lay, ball):
        out = []
        lower = self.full
        for k in range(self.mu):
            out.append(lay[k] | (lower & ball))
            lower = lay[k]
        return tuple(out)

    def _rec_mu(self, depth, lay, deficit, P, M, stab):
        self.stats["nodes"] += 1
        if self._budget_out():
            return False
        if deficit == 0:
            self._witness = list(P)
            return True
        r = M - len(P)
        if r <= 0:
            self.stats["leaves"] += 1
            return False
        if r * self.maxball < deficit:
            self.stats["pruned_bound"] += 1
            return False
        short = self.full & ~lay[self.mu - 1]     # not yet covered mu times
        if r * self.maxball < deficit + self.strong_bound * self.maxball:
            if r * self._best_drop(short) < deficit:
                self.stats["pruned_strong"] += 1
                return False
        u = (short & -short).bit_length() - 1
        cands, sub = self._branch_set(depth, u, P, stab)
        for k, c in enumerate(cands):
            c = int(c)
            nxt = self._new_layers(lay, self.balls[c])
            drop = sum(a.bit_count() - b.bit_count() for a, b in zip(nxt, lay))
            if self._rec_mu(depth + 1, nxt, deficit - drop, P + [c], M,
                            sub[k]):
                return True
        return False

    # -------- branching with isomorph rejection
    def _branch_set(self, depth, u, P, stab):
        """The children of a node, and each child's stabiliser.

        Two different rules, both exact:

        * **root (P empty)** — every non-empty solution contains *some*
          codeword, and G may be applied freely, so the first codeword
          may be taken to be one G-orbit representative per orbit of the
          whole space. For a transitive G (any translation group) that
          is a single branch instead of ``|coverers(u*)|``.
        * **inner nodes** — branch over ``coverers(u*)`` reduced to
          ``Stab_H(u*)``-orbit representatives, ``H = Stab_G(P)``. Only
          group elements that fix both P (setwise) and ``u*`` may be used
          here, because the branch element has to stay a coverer of
          ``u*``.
        """
        if depth == 0 and len(self.elements) > 1:
            reps = (self._root_reps if self._root_reps is not None
                    else self.group.orbits().reps)
            self.stats["pruned_iso"] += self.size - len(reps)
            return reps, [self._stab_of(stab, np.array([c])) for c in reps]
        cands = self.coverers[u]
        if self.mu > 1 and P:
            # a solution is a SET. At mu = 1 an incumbent can never cover
            # a still-uncovered word, so this cannot happen; at mu > 1 a
            # word of P may well reach a word that is merely *short* of
            # its multiplicity, and picking it again would build a
            # multiset. (Dropping members of P keeps the branch set
            # H-invariant: H fixes P setwise.)
            cands = cands[~np.isin(cands, np.asarray(P, dtype=np.int64))]
        if depth >= self.iso_depth or len(stab) <= 1 or not len(self.gperm):
            return cands, [stab] * len(cands)
        hstar = stab[self.gperm[stab, u] == u]      # H* = Stab_H(u*)
        if len(hstar) <= 1:
            return cands, [hstar] * len(cands)
        img = self.gperm[np.ix_(hstar, cands)]      # (|H*|, |cands|)
        reps = cands[img.min(axis=0) == cands]
        self.stats["pruned_iso"] += len(cands) - len(reps)
        pa = np.asarray(P, dtype=np.int64)
        subs = [self._stab_of(hstar, np.append(pa, c)) for c in reps]
        return reps, subs

    def _stab_of(self, stab, subset):
        """The elements of ``stab`` fixing ``subset`` setwise."""
        if len(stab) <= 1:
            return stab
        want = np.sort(np.asarray(subset, dtype=np.int64))
        im = np.sort(self.gperm[np.ix_(stab, want)], axis=1)
        return stab[(im == want[None, :]).all(axis=1)]

    def _budget_out(self):
        if self._node_limit and self.stats["nodes"] >= self._node_limit:
            self._stopped = True
            return True
        if self._deadline and not self.stats["nodes"] % 4096:
            if time.time() > self._deadline:
                self._stopped = True
                return True
        if self._log_every and not self.stats["nodes"] % self._log_every:
            self._on_log(f"nodes={self.stats['nodes']} "
                         f"pruned={self.stats['pruned_bound']} "
                         f"iso={self.stats['pruned_iso']}")
        return self._stopped

    # -------------------------------------------------- tree-size estimate
    def estimate_nodes(self, M, samples=200, seed=1, max_depth=None):
        """Knuth's random-dive estimate of the backtrack tree size.

        One dive walks from the root, and at a node with ``b`` surviving
        children multiplies a running weight by ``b`` and descends into a
        uniformly random one; the sum of the weights along the dive is an
        unbiased estimator of the total node count of the tree the *real*
        search would traverse (Knuth 1975, "Estimating the efficiency of
        backtrack programs"). Averaging dives gives the expected size —
        the honest way to answer "would this ever finish?" without
        running it.

        Returns ``{"mean", "median", "p90", "samples", "nodes_per_s"}``
        in nodes; the estimator is high-variance (log-normal-ish), so the
        median is the more robust readout and the mean the more
        conservative one."""
        rng = np.random.default_rng(seed)
        M = int(M)
        totals = []
        self.stats.setdefault("pruned_iso", 0)
        maxd = max_depth or M
        for _ in range(int(samples)):
            U, P, stab, weight, total = self.full, [], \
                np.arange(len(self.elements), dtype=np.int64), 1.0, 0.0
            lay = (0,) * self.mu
            deficit = self.mu * self.size
            for depth in range(maxd):
                total += weight
                done = (U == 0) if self.mu == 1 else (deficit == 0)
                if done or len(P) >= M:
                    break
                r = M - len(P)
                need = U.bit_count() if self.mu == 1 else deficit
                if r * self.maxball < need:
                    break
                short = (U if self.mu == 1
                         else self.full & ~lay[self.mu - 1])
                if (r * self.maxball < need + self.strong_bound * self.maxball
                        and r * self._best_drop(short) < need):
                    break
                u = (short & -short).bit_length() - 1
                cands, subs = self._branch_set(depth, u, P, stab)
                if not len(cands):
                    break
                weight *= len(cands)
                k = int(rng.integers(len(cands)))
                c = int(cands[k])
                stab = subs[k]
                P = P + [c]
                if self.mu == 1:
                    U &= ~self.balls[c]
                else:
                    nxt = self._new_layers(lay, self.balls[c])
                    deficit -= sum(a.bit_count() - b.bit_count()
                                   for a, b in zip(nxt, lay))
                    lay = nxt
            totals.append(total)
        a = np.array(totals, dtype=np.float64)
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "samples": len(a),
                "M": M}

    # ----------------------------------------------------------- drivers
    def minimum(self, hi, lo=1, **kw):
        """Smallest provably-feasible size, walking down from ``hi``.

        Returns ``(M, witness, trace)``; ``trace`` records the verdict of
        every size tried, so a run that ran out of budget says exactly
        where."""
        trace, best, wit = [], None, None
        M = int(hi)
        while M >= lo:
            v, w, st = self.decide(M, **kw)
            trace.append((M, v, st))
            if v == "feasible":
                best, wit = M, w
                M -= 1
            else:
                break
        return best, wit, trace


def _mask(idx, size):
    """Index array -> Python big-int bitmask (bit i set iff i in idx)."""
    a = np.zeros(size, dtype=bool)
    a[np.asarray(idx, dtype=np.int64)] = True
    return int.from_bytes(np.packbits(a, bitorder="little").tobytes(),
                          "little")


def settle(backend, hi, group=None, **kw):
    """One-call exhaustive settlement: ``(M_opt, witness, trace)``."""
    return IsomorphFreeSearch(backend, group=group).minimum(hi, **kw)

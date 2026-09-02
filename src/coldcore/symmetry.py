"""Symmetry layer: group-invariant construction and orbit-restricted search.

Every space coldcore searches carries a large automorphism group, and the
search core ignores it: it treats the |X| words as |X| unrelated
candidates. This module makes symmetry a first-class IR concept.

The group
---------
A group element is a pair *(coordinate permutation, per-coordinate value
permutations)* — the general element of the wreath product
``(S_{a_0} x ... x S_{a_{n-1}}) |x| S_n`` that is the full automorphism
group of the Hamming space, and contains the isometry groups of the Lee
and torus-Chebyshev spaces as subgroups (per-axis rotations/reflections
plus axis permutations). Acting on a word ``x``:

    (g.x)_j = v_j[ x_{p_j} ]

``p_j`` names the *source* coordinate of output position ``j``, so the
action is a numpy gather; ``v_j`` permutes the alphabet of axis ``j``.
Mixed radix is supported: ``p`` may only move coordinates of equal axis
size. Composition, inversion and the induced permutation of *word
indices* (computed straight from mixed-radix digits, fully vectorised)
are all here.

Orbit-restricted search
-----------------------
:class:`SymmetricSearcher` searches over ORBITS, not words: a solution is
a union of chosen G-orbits. This is the 1990s group-invariant-code
construction, mechanised — and it is what beats free search whenever the
optimum has structure, because it collapses the candidate set from |X|
words to |X|/|G| orbits.

Two facts make it cheap *and* exact:

* **Correctness never depends on G.** Any union of orbits is an ordinary
  solution, scored by the backend's exact fields. Choosing a G that is
  *not* an automorphism group merely restricts the search to a bad
  subfamily; it can never report a wrong answer.
* **Orbit gain is an exact marginal, and the gain map bounds it.** The
  exact gain of adding orbit ``O`` is measured by applying it
  (``ball_update(O, +1)``) and reading the exact deficit back. The sum of
  the gain map over ``O`` — one whole-space transform gives it for *all*
  orbits at once, by a segment sum over the orbit CSR — is an upper
  bound on that exact gain, and is non-increasing as the solution grows.
  So the lazy-greedy bound heap over orbits is exact greedy (Minoux),
  with one transform amortised over the whole build.

Never an approximation: every accepted move is scored by a real deficit
readback, exactly as the free searcher scores a placement.
"""
import heapq
import time

import numpy as np

from .space import ProductSpace

#: Refuse to materialise a whole-space orbit partition above this size.
MAX_PARTITION = 1 << 24

#: Cap on explicit group-element enumeration (:meth:`Group.elements`).
MAX_ELEMENTS = 1 << 18


# ====================================================================
# group elements
# ====================================================================
class GroupElement:
    """One automorphism ``(perm, vperms)`` of a mixed-radix product space.

    ``perm[j]`` is the source coordinate feeding output position ``j``
    (so ``perm`` is a permutation of ``range(n)`` with
    ``axes[perm[j]] == axes[j]``), and ``vperms[j]`` is a permutation of
    ``range(axes[j])`` applied after the gather.
    """

    __slots__ = ("axes", "n", "perm", "vperms")

    def __init__(self, axes, perm=None, vperms=None):
        self.axes = tuple(int(a) for a in axes)
        self.n = len(self.axes)
        if perm is None:
            perm = np.arange(self.n, dtype=np.int64)
        self.perm = np.asarray(perm, dtype=np.int64)
        if self.perm.shape != (self.n,):
            raise ValueError("perm must have one entry per coordinate")
        if sorted(self.perm.tolist()) != list(range(self.n)):
            raise ValueError("perm is not a permutation")
        for j in range(self.n):
            if self.axes[self.perm[j]] != self.axes[j]:
                raise ValueError(
                    f"coordinate permutation moves axis of size "
                    f"{self.axes[self.perm[j]]} into a position of size "
                    f"{self.axes[j]}")
        if vperms is None:
            vperms = [np.arange(a, dtype=np.int64) for a in self.axes]
        self.vperms = []
        for j, v in enumerate(vperms):
            v = np.asarray(v, dtype=np.int64)
            if v.shape != (self.axes[j],):
                raise ValueError(f"vperms[{j}] must permute {self.axes[j]} values")
            if sorted(v.tolist()) != list(range(self.axes[j])):
                raise ValueError(f"vperms[{j}] is not a permutation")
            self.vperms.append(v)

    # ---------------------------------------------------------- action
    def apply_words(self, words):
        """(k, n) digit array -> (k, n) digit array."""
        w = np.asarray(words, dtype=np.int64)
        out = np.empty_like(w)
        for j in range(self.n):
            out[..., j] = self.vperms[j][w[..., self.perm[j]]]
        return out

    def apply_index(self, space, idx):
        """int64 word indices -> int64 word indices (no words materialised).

        Digit ``s`` of ``idx`` is ``(idx // pow_q[s]) % axes[s]``; the
        image index is the mixed-radix recomposition of the permuted,
        value-mapped digits. O(n) vectorised passes, any array shape."""
        idx = np.asarray(idx, dtype=np.int64)
        pw = space.pow_q
        ax = space.axes
        out = np.zeros(idx.shape, dtype=np.int64)
        for j in range(self.n):
            s = int(self.perm[j])
            d = (idx // int(pw[s])) % int(ax[s])
            out += self.vperms[j][d] * int(pw[j])
        return out

    def permutation(self, space):
        """The induced permutation of the whole index set (size,)."""
        if space.size > MAX_PARTITION:
            raise ValueError(f"|X|={space.size} too large to materialise")
        return self.apply_index(space, np.arange(space.size, dtype=np.int64))

    # ------------------------------------------------------- structure
    def compose(self, other):
        """``self . other`` — apply ``other`` first, then ``self``."""
        if self.axes != other.axes:
            raise ValueError("group elements live on different spaces")
        perm = other.perm[self.perm]
        vperms = [self.vperms[j][other.vperms[int(self.perm[j])]]
                  for j in range(self.n)]
        return GroupElement(self.axes, perm, vperms)

    def inverse(self):
        inv = np.empty(self.n, dtype=np.int64)
        inv[self.perm] = np.arange(self.n, dtype=np.int64)
        vperms = []
        for k in range(self.n):
            j = int(inv[k])
            v = self.vperms[j]
            iv = np.empty_like(v)
            iv[v] = np.arange(len(v), dtype=np.int64)
            vperms.append(iv)
        return GroupElement(self.axes, inv, vperms)

    def is_identity(self):
        return (bool((self.perm == np.arange(self.n)).all())
                and all(bool((v == np.arange(len(v))).all())
                        for v in self.vperms))

    def key(self):
        """Hashable canonical key (used for element-set closure)."""
        return (self.perm.tobytes(),
                tuple(v.tobytes() for v in self.vperms))

    def __eq__(self, other):
        return isinstance(other, GroupElement) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return (f"GroupElement(perm={self.perm.tolist()}, "
                f"vperms={[v.tolist() for v in self.vperms]})")


# ====================================================================
# orbit partition
# ====================================================================
class OrbitPartition:
    """The G-orbits of the whole space, in CSR form.

    Attributes
    ----------
    labels : (size,) int64 — orbit id of every word, in ``0..T-1``
    members : (size,) int64 — word indices sorted by orbit id
    starts, sizes : (T,) int64 — CSR offsets and orbit sizes
    reps : (T,) int64 — the smallest word index of each orbit
    """

    def __init__(self, size, labels):
        self.size = int(size)
        reps, inv = np.unique(labels, return_inverse=True)
        self.labels = inv.astype(np.int64).ravel()
        self.reps = reps.astype(np.int64)
        self.n_orbits = len(reps)
        self.sizes = np.bincount(self.labels,
                                 minlength=self.n_orbits).astype(np.int64)
        self.starts = np.concatenate(
            ([0], np.cumsum(self.sizes)[:-1])).astype(np.int64)
        self.members = np.argsort(self.labels, kind="stable").astype(np.int64)

    def orbit(self, o):
        """Word indices of orbit ``o`` (sorted)."""
        s = int(self.starts[o])
        return self.members[s:s + int(self.sizes[o])]

    def segment_sum(self, values):
        """Sum ``values[w]`` over each orbit -> (T,) int64.

        ``values`` is indexed by word; one call turns a whole-space field
        (gain map, loss map) into a per-orbit total. The accumulator is
        int64 regardless of the field's dtype, so no widening copy of the
        whole space is made on the way in."""
        v = np.asarray(values)[self.members]
        if not self.n_orbits:
            return np.zeros(0, dtype=np.int64)
        return np.add.reduceat(v, self.starts, dtype=np.int64)

    def __len__(self):
        return self.n_orbits

    def __repr__(self):
        return (f"OrbitPartition(|X|={self.size}, orbits={self.n_orbits}, "
                f"sizes {int(self.sizes.min())}..{int(self.sizes.max())})")


# ====================================================================
# groups
# ====================================================================
class Group:
    """A permutation group acting on a :class:`~coldcore.space.ProductSpace`.

    Stored by generators; orbits and (when small) the element list are
    computed on demand and cached.
    """

    def __init__(self, space, gens=(), name=None):
        self.space = space if isinstance(space, ProductSpace) else \
            ProductSpace(space)
        axes = tuple(self.space.axes)
        self.gens = []
        for g in gens:
            if not isinstance(g, GroupElement):
                raise TypeError("generators must be GroupElement")
            if tuple(g.axes) != axes:
                raise ValueError("generator lives on a different space")
            if not g.is_identity():
                self.gens.append(g)
        self.name = name or "group"
        self._perms = None
        self._part = None
        self._elems = None

    # ------------------------------------------------------ combination
    def join(self, other, name=None):
        """The group generated by both generator sets."""
        return Group(self.space, list(self.gens) + list(other.gens),
                     name=name or f"{self.name}+{other.name}")

    __add__ = join

    # ----------------------------------------------------------- orbits
    def _gen_perms(self):
        """Index permutations of every generator and its inverse."""
        if self._perms is None:
            if self.space.size > MAX_PARTITION:
                raise ValueError(
                    f"|X|={self.space.size} exceeds MAX_PARTITION; use "
                    f"orbit_of() for individual orbits")
            ps = []
            for g in self.gens:
                ps.append(g.permutation(self.space))
                ps.append(g.inverse().permutation(self.space))
            self._perms = ps
        return self._perms

    def orbit_of(self, idx):
        """Closure of ``idx`` under the generators — sorted index array.

        Works at any space size: it only ever touches the orbit itself.
        """
        seen = np.unique(np.atleast_1d(np.asarray(idx, dtype=np.int64)))
        if not self.gens:
            return seen
        frontier = seen
        while len(frontier):
            imgs = [g.apply_index(self.space, frontier) for g in self.gens]
            imgs += [g.inverse().apply_index(self.space, frontier)
                     for g in self.gens]
            cand = np.unique(np.concatenate(imgs))
            frontier = cand[~np.isin(cand, seen)]
            if len(frontier):
                seen = np.union1d(seen, frontier)
        return seen

    def orbits(self):
        """Whole-space orbit partition (cached).

        Label propagation on the generator permutations: ``lab`` starts as
        the identity and is repeatedly lowered by ``min(lab, lab[sigma])``
        with pointer jumping. At the fixpoint ``lab`` is constant on each
        orbit and equals its smallest word index (generators *and*
        inverses are used, so the relation is symmetric)."""
        if self._part is None:
            size = self.space.size
            lab = np.arange(size, dtype=np.int64)
            perms = self._gen_perms()
            if perms:
                while True:
                    prev = lab.copy()
                    for p in perms:
                        np.minimum(lab, lab[p], out=lab)
                    while True:                       # pointer jumping
                        nxt = lab[lab]
                        if np.array_equal(nxt, lab):
                            break
                        lab = nxt
                    if np.array_equal(lab, prev):
                        break
            self._part = OrbitPartition(size, lab)
        return self._part

    # --------------------------------------------------------- elements
    def elements(self, cap=MAX_ELEMENTS):
        """Every element, by closure over the generators (cached).

        Raises when the group is bigger than ``cap`` — the point of the
        orbit machinery is that it never needs this; the exhaustive
        isomorph-free module does, for small groups."""
        if self._elems is None:
            ident = GroupElement(self.space.axes)
            found = {ident.key(): ident}
            frontier = [ident]
            gens = list(self.gens) + [g.inverse() for g in self.gens]
            while frontier:
                nxt = []
                for a in frontier:
                    for g in gens:
                        h = g.compose(a)
                        k = h.key()
                        if k not in found:
                            if len(found) >= cap:
                                raise ValueError(
                                    f"group larger than cap={cap}")
                            found[k] = h
                            nxt.append(h)
                frontier = nxt
            self._elems = list(found.values())
        return self._elems

    def order(self, cap=MAX_ELEMENTS):
        return len(self.elements(cap))

    # ------------------------------------------------------ diagnostics
    def is_automorphism_of(self, backend):
        """True when every generator preserves the coverage relation.

        Exact, brute force: checks ``ball(g.w) == g.ball(w)`` for every
        word. Needs ``backend.ball_indices`` (the reference backends have
        it) and a tiny space. Orbit search does not *need* this to be
        true — it only needs it to be *useful*."""
        if not hasattr(backend, "ball_indices"):
            raise TypeError("backend cannot enumerate balls")
        sp = self.space
        for g in self.gens:
            p = g.permutation(sp)
            for w in range(sp.size):
                a = np.sort(p[backend.ball_indices(w)])
                b = np.sort(backend.ball_indices(int(p[w])))
                if len(a) != len(b) or not np.array_equal(a, b):
                    return False
        return True

    def __len__(self):
        return len(self.gens)

    def __repr__(self):
        return (f"Group({self.name!r}, axes={list(self.space.axes)}, "
                f"{len(self.gens)} generators)")


# ====================================================================
# standard subgroups
# ====================================================================
def trivial_group(space):
    """The one-element group: orbits are singletons, orbit search
    degenerates *exactly* to free search over words."""
    return Group(space, [], name="trivial")


def translation(space, vec):
    """The single automorphism ``x -> x + vec`` (per-axis, mod a_j)."""
    sp = space if isinstance(space, ProductSpace) else ProductSpace(space)
    v = np.asarray(vec, dtype=np.int64).ravel()
    if len(v) != sp.n:
        raise ValueError("translation vector must have one entry per axis")
    vperms = [(np.arange(a, dtype=np.int64) + int(v[j])) % a
              for j, a in enumerate(sp.axes)]
    return GroupElement(sp.axes, None, vperms)


def translation_group(space, vectors, name=None):
    """Translations by the given vectors.

    Over a prime alphabet this is exactly "cosets of the linear code
    spanned by ``vectors``": the orbits are the cosets, so an
    orbit-restricted search *is* a search for a union of code cosets —
    the classical group-invariant construction."""
    sp = space if isinstance(space, ProductSpace) else ProductSpace(space)
    gens = [translation(sp, v) for v in np.atleast_2d(vectors)]
    return Group(sp, gens, name=name or "translate")


#: The construction-oriented alias: over Z_q the translation group
#: generated by the rows of a generator matrix has the code's cosets as
#: its orbits.
linear_code_translations = translation_group


def diagonal_translations(space, step=1, name=None):
    """``x -> x + c*(1,1,...,1)`` — the diagonal translation subgroup."""
    sp = space if isinstance(space, ProductSpace) else ProductSpace(space)
    return translation_group(sp, [[step] * sp.n], name=name or "diag")


def value_shift(space, axis, step=1, name=None):
    """Cyclic value shift on one axis: ``x_axis -> x_axis + step``.

    The q-ary translation subgroup of a single coordinate; the product
    over all axes is the full translation group of the space."""
    sp = space if isinstance(space, ProductSpace) else ProductSpace(space)
    v = np.zeros(sp.n, dtype=np.int64)
    v[int(axis)] = int(step)
    return Group(sp, [translation(sp, v)], name=name or f"shift{axis}")


def value_shifts(space, name=None):
    """All per-axis cyclic value shifts — the full translation group."""
    sp = space if isinstance(space, ProductSpace) else ProductSpace(space)
    gens = []
    for j in range(sp.n):
        v = np.zeros(sp.n, dtype=np.int64)
        v[j] = 1
        gens.append(translation(sp, v))
    return Group(sp, gens, name=name or "shifts")


def coordinate_permutation(space, perm):
    """The automorphism that gathers coordinate ``perm[j]`` into slot j."""
    sp = space if isinstance(space, ProductSpace) else ProductSpace(space)
    return GroupElement(sp.axes, perm, None)


def coordinate_rotation(space, name=None):
    """The cyclic rotation of coordinates (needs all axes equal)."""
    sp = space if isinstance(space, ProductSpace) else ProductSpace(space)
    perm = (np.arange(sp.n, dtype=np.int64) + 1) % sp.n
    return Group(sp, [coordinate_permutation(sp, perm)], name=name or "rot")


def coordinate_symmetric(space, name=None):
    """The full symmetric group on coordinates (needs all axes equal)."""
    sp = space if isinstance(space, ProductSpace) else ProductSpace(space)
    gens = []
    if sp.n >= 2:
        swap = np.arange(sp.n, dtype=np.int64)
        swap[0], swap[1] = 1, 0
        gens.append(coordinate_permutation(sp, swap))
    if sp.n >= 3:
        gens.append(coordinate_permutation(
            sp, (np.arange(sp.n, dtype=np.int64) + 1) % sp.n))
    return Group(sp, gens, name=name or "sym")


def value_permutation(space, axis, vperm):
    """Relabel the alphabet of one axis by ``vperm``."""
    sp = space if isinstance(space, ProductSpace) else ProductSpace(space)
    vs = [np.arange(a, dtype=np.int64) for a in sp.axes]
    vs[int(axis)] = np.asarray(vperm, dtype=np.int64)
    return GroupElement(sp.axes, None, vs)


def monomial_group(space, name=None):
    """The full Hamming automorphism group ``(prod_j S_{a_j}) |x| S_n``.

    Generators: a transposition and a cycle of each axis alphabet, plus a
    transposition and a cycle of the coordinates. This is the group the
    isomorph-free exhaustive module classifies codes under."""
    sp = space if isinstance(space, ProductSpace) else ProductSpace(space)
    gens = []
    for j, a in enumerate(sp.axes):
        if a >= 2:
            sw = np.arange(a, dtype=np.int64)
            sw[0], sw[1] = 1, 0
            gens.append(value_permutation(sp, j, sw))
        if a >= 3:
            gens.append(value_permutation(
                sp, j, (np.arange(a, dtype=np.int64) + 1) % a))
    gens += coordinate_symmetric(sp).gens
    return Group(sp, gens, name=name or "mono")


def isometry_group(space, problem="hamming", name=None):
    """Generators of the isometry group of a shipped problem family.

    ``hamming``  full monomial group (any relabelling of any axis)
    ``lee`` / ``torus_linf``  per-axis rotations and reflections
    (the dihedral group of each circle), plus coordinate permutations
    between equal axes.  ``grid_linf`` has no translations: only
    per-axis reflections and coordinate permutations survive the open
    boundary."""
    sp = space if isinstance(space, ProductSpace) else ProductSpace(space)
    if problem == "hamming":
        return monomial_group(sp, name=name)
    gens = []
    if problem in ("lee", "torus_linf"):
        gens += value_shifts(sp).gens
    if problem in ("lee", "torus_linf", "grid_linf"):
        for j, a in enumerate(sp.axes):
            gens.append(value_permutation(
                sp, j, (-np.arange(a, dtype=np.int64)) % a
                if problem != "grid_linf"
                else np.arange(a - 1, -1, -1, dtype=np.int64)))
    if len(set(sp.axes)) == 1:
        gens += coordinate_symmetric(sp).gens
    return Group(sp, gens, name=name or f"iso-{problem}")


# ------------------------------------------------------------ spec parser
def _parse_vector(sp, s):
    if "x" in s:
        v = [int(t) for t in s.split("x")]
    elif len(s) == sp.n and s.isdigit():
        v = [int(c) for c in s]
    else:
        raise ValueError(f"cannot read translation vector {s!r}")
    if len(v) != sp.n:
        raise ValueError(f"vector {s!r} has {len(v)} entries, need {sp.n}")
    return v


def parse_group(space, spec):
    """Build a group from a CLI/plugin spec string.

    Terms are joined with ``+``; each term is one of

      ``trivial``                     the one-element group
      ``diag`` / ``diag:S``           x -> x + S*(1,...,1)
      ``shift:J`` / ``shift:J:S``     cyclic value shift on axis J
      ``shifts``                      all per-axis shifts
      ``translate:VEC``               x -> x + VEC  (``1011`` or ``1x0x1x1``)
      ``rot``                         cyclic coordinate rotation
      ``sym``                         full coordinate symmetric group
      ``mono``                        full monomial (Hamming) group
      ``iso:FAMILY``                  isometry group of a problem family

    e.g. ``translate:1011+translate:0112`` is the ternary Hamming code's
    own translation group, whose orbits are its 9 cosets.
    """
    sp = space if isinstance(space, ProductSpace) else ProductSpace(space)
    g = Group(sp, [], name=spec)
    for term in str(spec).split("+"):
        term = term.strip()
        if not term:
            continue
        head, _, rest = term.partition(":")
        head = head.strip().lower()
        if head == "trivial":
            part = trivial_group(sp)
        elif head == "diag":
            part = diagonal_translations(sp, int(rest) if rest else 1)
        elif head == "shift":
            bits = rest.split(":")
            part = value_shift(sp, int(bits[0]),
                               int(bits[1]) if len(bits) > 1 else 1)
        elif head == "shifts":
            part = value_shifts(sp)
        elif head == "translate":
            part = translation_group(sp, [_parse_vector(sp, rest)])
        elif head == "rot":
            part = coordinate_rotation(sp)
        elif head == "sym":
            part = coordinate_symmetric(sp)
        elif head == "mono":
            part = monomial_group(sp)
        elif head == "iso":
            part = isometry_group(sp, rest or "hamming")
        else:
            raise ValueError(f"unknown group term {term!r}")
        g = g.join(part, name=spec)
    return g


# ====================================================================
# orbit-restricted search
# ====================================================================
#: Ruin operators of the orbit searcher, and the draw list used by the
#: fixed selector.
ORBIT_RUIN_MODES = ("low", "random", "deficit")
_ORBIT_FIXED = ["low", "low", "random", "deficit"]


class SymmetricSearcher:
    """Ruin-and-recreate search whose atoms are G-orbits, not words.

    The solution is always ``union of self.selected orbits``; every score
    is read back from the backend's exact fields, so the result is an
    ordinary solution that any verifier accepts.

    Parameters
    ----------
    backend : coldcore.protocol.Backend
    group   : :class:`Group` on the backend's space
    max_orbit : drop orbits larger than this from the candidate set
                (they can only overshoot a small target M)
    """

    def __init__(self, backend, group, seed=1, deadline=None, on_log=None,
                 on_improve=None, on_solved=None, max_orbit=None):
        self.b = backend
        if list(group.space.axes) != list(backend.space.axes):
            raise ValueError("group and backend live on different spaces")
        self.group = group
        self.part = group.orbits()
        self.rng = np.random.default_rng(seed * 1000003 + 12345)
        self.deadline = deadline
        self.on_log = on_log or (lambda s: None)
        self.on_improve = on_improve or (lambda s: None)
        self.on_solved = on_solved or (lambda m, idx: None)
        allowed = np.ones(len(self.part), dtype=bool)
        if max_orbit is not None:
            allowed &= self.part.sizes <= int(max_orbit)
        self.allowed = allowed
        self.selected = np.zeros(len(self.part), dtype=bool)
        self._code = np.zeros(0, dtype=np.int64)   # None = needs rebuild
        self.n_points = 0                          # |code|, always current
        self.uncov = None
        self.best_sel = None
        self.best_uncov = None
        self.stats = {"transforms": 0, "orbit_adds": 0, "orbit_drops": 0,
                      "probes": 0, "rounds": 0, "recounts": 0}
        self.b.recount(self.code)
        self.uncov = self.b.uncovered()

    # ------------------------------------------------------------ time
    def time_left(self):
        return np.inf if self.deadline is None else self.deadline - time.time()

    # ------------------------------------------------------ state moves
    @property
    def code(self):
        """The solution as a sorted index array, materialised on demand.

        A greedy *probe* adds an orbit, reads its exact marginal back and
        usually drops it again; splicing the point list twice per probe
        was pure overhead, since nothing between the two reads the list —
        only its length, which :attr:`n_points` tracks exactly. So the
        array is rebuilt from ``selected`` the first time anyone asks."""
        if self._code is None:
            sel = np.flatnonzero(self.selected)
            self._code = (np.sort(np.concatenate(
                [self.part.orbit(o) for o in sel])) if len(sel)
                else np.zeros(0, dtype=np.int64))
        return self._code

    @code.setter
    def code(self, value):
        self._code = np.asarray(value, dtype=np.int64)
        self.n_points = len(self._code)

    def _rebuild_code(self):
        self._code = None
        self.n_points = int(self.part.sizes[self.selected].sum())
        return self.code

    def add_orbit(self, o):
        """Add orbit ``o``; returns its EXACT marginal gain (deficit drop)."""
        if self.selected[o]:
            return 0
        pts = self.part.orbit(o)
        self.b.ball_update(pts, +1)
        new = self.b.uncovered()
        gain = self.uncov - new
        self.uncov = new
        self.selected[o] = True
        self._code = None
        self.n_points += len(pts)
        self.stats["orbit_adds"] += 1
        return int(gain)

    def drop_orbit(self, o):
        """Drop orbit ``o``; returns its EXACT marginal loss (deficit rise)."""
        if not self.selected[o]:
            return 0
        pts = self.part.orbit(o)
        self.b.ball_update(pts, -1)
        new = self.b.uncovered()
        loss = new - self.uncov
        self.uncov = new
        self.selected[o] = False
        self._code = None
        self.n_points -= len(pts)
        self.stats["orbit_drops"] += 1
        return int(loss)

    def _undo_add(self, o, gain):
        """Exact inverse of the :meth:`add_orbit` that returned ``gain``.

        ``ball_update`` is its own exact inverse, so the deficit after
        the undo is *known*: it is the value from before the add, and
        ``gain`` is the difference the add itself measured. Reading it
        back off the field would be a whole-space scan to learn a number
        already in hand — and on a greedy build most probes are undone,
        so that scan was half the cost of the pass. Bookkeeping is
        identical to :meth:`drop_orbit`'s, stats included."""
        if not self.selected[o]:              # same guard as drop_orbit
            return
        pts = self.part.orbit(o)
        self.b.ball_update(pts, -1)
        self.uncov += int(gain)
        self.selected[o] = False
        self._code = None
        self.n_points -= len(pts)
        self.stats["orbit_drops"] += 1

    def _undo_drop(self, o, loss):
        """Exact inverse of the :meth:`drop_orbit` that returned ``loss``."""
        if self.selected[o]:                  # same guard as add_orbit
            return
        pts = self.part.orbit(o)
        self.b.ball_update(pts, +1)
        self.uncov -= int(loss)
        self.selected[o] = True
        self._code = None
        self.n_points += len(pts)
        self.stats["orbit_adds"] += 1

    def load_orbits(self, orbit_ids):
        """Set the solution to exactly these orbits (full recount)."""
        self.selected[:] = False
        self.selected[np.asarray(orbit_ids, dtype=np.int64)] = True
        self._rebuild_code()
        self.b.recount(self.code)
        self.stats["transforms"] += 1
        self.uncov = self.b.uncovered()
        self._snapshot()
        return self.uncov

    def verify_full(self):
        self.b.recount(self.code)
        self.stats["transforms"] += 1
        self.stats["recounts"] += 1
        self.uncov = self.b.uncovered()
        return self.uncov

    def _snapshot(self):
        if self.best_uncov is None or self.uncov < self.best_uncov or (
                self.uncov == self.best_uncov and
                self.n_points < int(self.best_sel[1])):
            self.best_uncov = self.uncov
            self.best_sel = (self.selected.copy(), self.n_points)
            self.on_improve(self)

    def _restore_best(self):
        if self.best_sel is None:
            return
        self.selected = self.best_sel[0].copy()
        self._rebuild_code()
        self.b.recount(self.code)
        self.stats["transforms"] += 1
        self.uncov = self.b.uncovered()

    # ------------------------------------------------------- orbit fields
    def gain_bounds(self):
        """Upper bound on every orbit's exact gain — one transform.

        ``gain_map()[w]`` is the exact per-word gain, so summing it over
        an orbit over-counts exactly the words two members both newly
        cover: the sum bounds the true joint gain from above, and (like
        the per-word gains) it never increases as the solution grows.
        That is all a lazy-greedy bound heap needs to stay exact."""
        self.b.gain_map()
        self.stats["transforms"] += 1
        return self.part.segment_sum(self.b.map_all())

    def loss_bounds(self):
        """Per-orbit sums of the exact loss map.

        Zero is a *necessary* condition for an orbit to be removable
        without losing feasibility (any critical word in a member's ball
        makes the sum positive), so it is an exact filter for the peel —
        but it is not sufficient (a word covered twice from inside the
        same orbit is not critical), so the peel always re-checks by
        actually dropping the orbit."""
        self.b.loss_map()
        self.stats["transforms"] += 1
        return self.part.segment_sum(self.b.map_all())

    # -------------------------------------------------------- recreate
    def greedy_fill(self, target_m, refresh_every=64):
        """Exact lazy greedy over orbits, up to ``target_m`` POINTS.

        Bounds come from one gain transform and only ever decrease, so
        popping the best bound, scoring it exactly and re-inserting it
        when it no longer leads is the textbook lazy greedy — exact, with
        one whole-space transform amortised over the whole build."""
        heap = []
        placed = refresh_every
        while self.uncov > 0 and self.n_points < target_m:
            if self.time_left() < 5:
                break
            if placed >= refresh_every or not heap:
                bnd = self.gain_bounds()
                room = target_m - self.n_points
                # one vectorised selection instead of a Python scan with
                # numpy scalar indexing over every orbit: same orbits,
                # ascending, same tuples, same heap
                pick = np.flatnonzero(self.allowed & ~self.selected
                                      & (bnd > 0)
                                      & (self.part.sizes <= room))
                heap = list(zip((-bnd[pick]).tolist(), pick.tolist()))
                heapq.heapify(heap)
                placed = 0
                if not heap:
                    break
            v, o = heapq.heappop(heap)
            if self.part.sizes[o] > target_m - self.n_points:
                continue
            gain = self.add_orbit(o)              # exact, leaves it added
            self.stats["probes"] += 1
            nxt = -heap[0][0] if heap else 0
            if gain <= 0:
                self._undo_add(o, gain)
                continue
            if gain >= nxt:
                placed += 1                       # keep it: exact greedy
            else:
                self._undo_add(o, gain)
                heapq.heappush(heap, (-gain, o))
        self._snapshot()
        return self.uncov

    # ------------------------------------------------------------- peel
    def peel(self):
        """Drop orbits whose removal keeps the solution feasible.

        Cheap necessary filter (zero loss-map sum) then an exact check:
        actually drop the orbit and read the deficit back."""
        removed = 0
        while True:
            if self.time_left() < 5:
                break
            lb = self.loss_bounds()
            cand = np.flatnonzero(self.selected & (lb == 0))
            if not len(cand):
                break
            self.rng.shuffle(cand)
            hit = False
            for o in cand:
                loss = self.drop_orbit(int(o))
                if loss == 0:
                    removed += 1
                    hit = True
                    break
                self._undo_drop(int(o), loss)     # not removable: put back
            if not hit:
                break
        if removed:
            self._snapshot()
        return removed

    # ------------------------------------------------------------- ruin
    def ruin(self, k, mode="low"):
        """Drop k selected orbits; returns the orbit ids removed."""
        sel = np.flatnonzero(self.selected)
        k = min(int(k), max(0, len(sel) - 1))
        if k <= 0:
            return np.zeros(0, dtype=np.int64)
        if mode == "random":
            pick = self.rng.choice(sel, size=k, replace=False)
        elif mode == "deficit":
            tgt = self._deficit_words()
            if not len(tgt):
                return self.ruin(k, "low")
            reps = self.b.index_word(self.part.reps[sel]).astype(np.int16)
            tw = self.b.index_word(tgt).astype(np.int16)
            d = np.full(len(sel), np.iinfo(np.int32).max, dtype=np.int64)
            for t in tw:
                np.minimum(d, (reps != t[None, :]).sum(axis=1), out=d)
            order = np.argsort(d + self.rng.random(len(sel)))
            near = sel[order[:min(len(sel), max(3 * k, k + 4))]]
            pick = self.rng.choice(near, size=k, replace=False)
        else:                                     # "low": cheapest orbits
            lb = self.loss_bounds()[sel].astype(np.float64)
            jit = self.rng.exponential(
                scale=max(1.0, lb.mean() * 0.15), size=len(sel))
            pick = sel[np.argsort(lb + jit)[:k]]
        for o in pick:
            self.drop_orbit(int(o))
        return np.asarray(pick, dtype=np.int64)

    def _deficit_words(self, limit=4):
        """A few currently-deficient word indices (exact), or empty."""
        if not hasattr(self.b, "read_cnt"):
            return np.zeros(0, dtype=np.int64)
        cnt = np.asarray(self.b.read_cnt())
        mu = int(getattr(self.b, "mu", 1))
        bad = np.flatnonzero(cnt < mu)
        return bad[:limit].astype(np.int64)

    # -------------------------------------------------------------- LNS
    def lns(self, M, round_budget_s, kmin=1, kmax=None, log_every=0,
            seed=None):
        """Ruin/recreate rounds at fixed size ``M`` (points).

        Identical bookkeeping to the free searcher's LNS — best snapshot,
        ~10% random acceptance of regressions — with orbits as atoms."""
        rng_saved = None
        if seed is not None:
            rng_saved, self.rng = self.rng, np.random.default_rng(
                seed * 1000003 + 12345)
        try:
            return self._lns(M, round_budget_s, kmin, kmax, log_every)
        finally:
            if rng_saved is not None:
                self.rng = rng_saved

    def _lns(self, M, round_budget_s, kmin, kmax, log_every):
        t0 = time.time()
        n_sel = max(1, int(self.selected.sum()))
        kmax = kmax or max(kmin + 1, n_sel // 3)
        self.best_uncov = None
        self.best_sel = None
        # bring the state to M first: a notch of the descent arrives here
        # ABOVE M and feasible, and without this the loop below never
        # runs (uncov == 0), so the notch was silently reported failed
        while self.n_points > M and int(self.selected.sum()) > 1:
            self.ruin(1, "low")
        if self.n_points < M:
            self.greedy_fill(M)
        self._snapshot()
        r = 0
        while self.uncov > 0:
            if time.time() - t0 > round_budget_s or self.time_left() < 10:
                break
            pre = self.uncov
            k = int(self.rng.integers(kmin, kmax + 1))
            mode = _ORBIT_FIXED[int(self.rng.integers(len(_ORBIT_FIXED)))]
            self.ruin(k, mode)
            self.greedy_fill(M)
            self.stats["rounds"] += 1
            r += 1
            if self.uncov > pre and self.rng.random() > 0.10:
                self._restore_best()
            else:
                self._snapshot()
            if log_every and r % log_every == 0:
                self.on_log(f"orbit-lns M={M} round={r} uncov={self.uncov} "
                            f"best={self.best_uncov} k={k} "
                            f"({time.time() - t0:.0f}s)")
        if self.uncov == 0:
            self._snapshot()
            self.on_solved(len(self.code), self.code.copy())
        elif self.best_uncov is not None and self.best_uncov < self.uncov:
            self._restore_best()
        return self.uncov

    # ---------------------------------------------------------- descent
    def descend(self, floor=1, notch_budget_s=30.0, attempts=2, log=True):
        """Notch ladder over orbit sizes, from the current feasible size.

        One notch = "drop the smallest orbit's worth of points and try to
        repair", which is the natural step of a symmetric search: a
        group-invariant solution can only change size in orbit-sized
        chunks. A failed notch is retried ``attempts`` times from the
        last feasible solution (different random stream each time)
        before the ladder stops."""
        if self.uncov != 0:
            raise ValueError("descend needs a feasible starting solution")
        step = int(self.part.sizes[self.allowed].min())
        best = self.code.copy()
        best_sel = self.selected.copy()
        self.on_solved(len(best), best.copy())
        while len(self.code) - step >= floor:
            if self.time_left() < 2:
                break
            target = len(self.code) - step
            won = False
            for _ in range(max(1, int(attempts))):
                if self.time_left() < 2:
                    break
                unc = self.lns(target, notch_budget_s)
                if unc == 0 and len(self.code) <= target:
                    best = self.code.copy()
                    best_sel = self.selected.copy()
                    won = True
                    if log:
                        self.on_log(
                            f"orbit-descend: feasible at M={len(best)}")
                    self.on_solved(len(best), best.copy())
                    break
                self.selected = best_sel.copy()   # restart from last feasible
                self._rebuild_code()
                self.b.recount(self.code)
                self.stats["transforms"] += 1
                self.uncov = self.b.uncovered()
            if not won:
                break
        self.selected = best_sel.copy()
        self._rebuild_code()
        self.b.recount(self.code)
        self.stats["transforms"] += 1
        self.uncov = self.b.uncovered()
        return best

    # ------------------------------------------------------ handoff
    def free_searcher(self, seed=1, **kw):
        """A plain :class:`~coldcore.search.Searcher` seeded with this
        solution — the structured-start / free-polish hybrid: take the
        group-invariant code, then let word-level LNS break symmetry."""
        from .search import Searcher
        s = Searcher(self.b, seed=seed, deadline=self.deadline, **kw)
        s.load(self.code)
        return s

    def summary(self):
        return {"orbits": len(self.part),
                "orbit_sizes": (int(self.part.sizes.min()),
                                int(self.part.sizes.max())),
                "selected": int(self.selected.sum()),
                "points": int(self.n_points),
                "uncovered": int(self.uncov)}


def orbit_solve(backend, group, budget_s=30.0, seed=1, floor=1, **kw):
    """Convenience driver: greedy over orbits -> peel -> orbit descent.

    Returns ``(code, searcher)``; ``code`` is a plain index array that
    the ordinary verifier accepts."""
    deadline = time.time() + budget_s
    s = SymmetricSearcher(backend, group, seed=seed, deadline=deadline, **kw)
    s.greedy_fill(10 ** 9)
    if s.uncov != 0:
        return s.code.copy(), s
    s.peel()
    code = s.descend(floor=floor, notch_budget_s=max(1.0, budget_s / 6))
    return code, s

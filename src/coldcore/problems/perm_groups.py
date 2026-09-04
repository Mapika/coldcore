"""Permutation groups on the point set ``{0..v-1}`` and their orbits on
r-subsets — the symmetry layer of the covering designs.

A group is given by generators (permutation arrays ``p`` with ``p[x]``
the image of ``x``); the element list is enumerated by closure, capped
at :data:`MAX_ELEMENTS`. Orbits of r-subsets are computed the way
``coldcore.symmetry.Group.orbits`` computes orbits of words: each
generator induces a permutation of the colex ranks (rank of the sorted
image), and label propagation ``lab = min(lab, lab[sigma])`` with
pointer jumping over generators and inverses reaches the fixpoint where
``lab`` is the smallest rank in each orbit. Orbit partitions are
:class:`coldcore.symmetry.OrbitPartition` instances (CSR: ``labels``,
``members``, ``starts``, ``sizes``, ``reps``).

Built-in groups (``parse_design_group(v, spec)``; terms joined with
``+`` generate the join):

    cyclic          Z_v          x -> x+1
    cyclic-1        Z_{v-1}      on 0..v-2, the point v-1 fixed
    cyclic-2        Z_{v-2}      on 0..v-3, two fixed points
    dihedral        D_v          x -> x+1, x -> -x
    prod:AxB        Z_A x Z_B    v = A*B, x = (i, j) with x = i*B + j
    affine          AGL(1,p)     x -> a x + b, prime v = p
    affine:A        <x -> Ax, x -> x+1>, a subgroup of AGL(1,p)
    mult:A          <x -> Ax>            (v prime)
    gens=(0 1 2)(3 4);(5 6)   explicit cycles, generators separated by ;
    trivial
"""
import re
from math import comb

import numpy as np

from ..symmetry import OrbitPartition
from .covering_design import binom_table, rank_subsets, unrank_subsets

#: Closure cap: the element list is a Python-level enumeration.
MAX_ELEMENTS = 200_000


class PermGroup:
    """A permutation group on ``v`` points, by generators."""

    def __init__(self, v, gens=(), name=None):
        self.v = int(v)
        seen = {}
        for g in gens:
            g = np.asarray(g, dtype=np.int64).ravel()
            if len(g) != self.v or sorted(g.tolist()) != list(range(self.v)):
                raise ValueError("generator is not a permutation of 0..v-1")
            if (g == np.arange(self.v)).all():
                continue
            seen.setdefault(g.tobytes(), g)
        self.gens = list(seen.values())
        self.name = name or "custom"
        self._elements = None

    # -- algebra --
    @staticmethod
    def compose(g, h):
        """``g o h``: apply ``h`` first."""
        return g[h]

    @staticmethod
    def inverse(g):
        inv = np.empty_like(g)
        inv[g] = np.arange(len(g))
        return inv

    def join(self, other, name=None):
        if other.v != self.v:
            raise ValueError("groups on different point sets")
        return PermGroup(self.v, self.gens + other.gens,
                         name=name or f"{self.name}+{other.name}")

    def elements(self, cap=MAX_ELEMENTS):
        """All group elements, by closure (identity first)."""
        if self._elements is None:
            ident = np.arange(self.v, dtype=np.int64)
            seen = {ident.tobytes(): ident}
            frontier = [ident]
            while frontier:
                nxt = []
                for e in frontier:
                    for g in self.gens:
                        h = g[e]
                        key = h.tobytes()
                        if key not in seen:
                            seen[key] = h
                            nxt.append(h)
                            if len(seen) > cap:
                                raise ValueError(
                                    f"group has more than {cap} elements")
                frontier = nxt
            self._elements = list(seen.values())
        return self._elements

    def order(self, cap=MAX_ELEMENTS):
        return len(self.elements(cap))

    # -- action on r-subsets --
    def induced(self, blocks, C=None):
        """For each generator, the permutation of the rows of ``blocks``
        (``(n, r)`` sorted subsets in colex order) it induces."""
        blocks = np.asarray(blocks)
        n, r = blocks.shape
        if C is None:
            C = binom_table(self.v, r)
        out = []
        for g in self.gens:
            sig = np.empty(n, dtype=np.int64)
            step = 1 << 20
            for lo in range(0, n, step):
                img = np.sort(g[blocks[lo:lo + step].astype(np.int64)], axis=1)
                sig[lo:lo + step] = rank_subsets(img, C=C)
            out.append(sig)
        return out

    def subset_orbits(self, r, blocks=None):
        """Orbit partition of the r-subsets of ``{0..v-1}`` (colex ranks).

        ``blocks`` may be passed when the caller already holds the
        ``(C(v,r), r)`` table."""
        n = comb(self.v, r)
        if blocks is None:
            blocks = unrank_subsets(np.arange(n, dtype=np.int64), self.v, r)
        lab = np.arange(n, dtype=np.int64)
        perms = self.induced(blocks)
        perms += [self.inverse(p) for p in perms]
        while True:
            before = lab.copy()
            for p in perms:
                np.minimum(lab, lab[p], out=lab)
            lab = lab[lab]                        # pointer jumping
            if (lab == before).all():
                break
        return OrbitPartition(n, lab)

    def __len__(self):
        return self.order()

    def __repr__(self):
        return f"PermGroup(v={self.v}, name={self.name!r}, gens={len(self.gens)})"


# ------------------------------------------------------------ built-ins
def _cycle_shift(v, n_move):
    """``x -> x+1`` on ``0..n_move-1``, the rest fixed."""
    p = np.arange(v, dtype=np.int64)
    p[:n_move] = (np.arange(n_move) + 1) % n_move
    return p


def trivial_group(v):
    return PermGroup(v, [], name="trivial")


def cyclic_group(v, fixed=0):
    """``Z_{v-fixed}`` on the first ``v - fixed`` points."""
    n = int(v) - int(fixed)
    if n < 1:
        raise ValueError("no points left to move")
    return PermGroup(v, [_cycle_shift(v, n)],
                     name="cyclic" if not fixed else f"cyclic-{fixed}")


def dihedral_group(v):
    """``D_v``: rotation and reflection of the v-cycle."""
    refl = (-np.arange(v, dtype=np.int64)) % int(v)
    return PermGroup(v, [_cycle_shift(v, v), refl], name="dihedral")


def product_cyclic_group(a, b):
    """``Z_a x Z_b`` on ``v = a b`` points, ``x = i*b + j``."""
    a, b = int(a), int(b)
    v = a * b
    i, j = np.divmod(np.arange(v, dtype=np.int64), b)
    return PermGroup(v, [((i + 1) % a) * b + j, i * b + (j + 1) % b],
                     name=f"prod:{a}x{b}")


def _is_prime(p):
    return p > 1 and all(p % d for d in range(2, int(p ** 0.5) + 1))


def _primitive_root(p):
    for g in range(2, p):
        x, seen = 1, set()
        for _ in range(p - 1):
            x = x * g % p
            seen.add(x)
        if len(seen) == p - 1:
            return g
    return 1


def affine_group(p, mult=None):
    """``x -> a x + b`` mod prime ``p``: all of ``AGL(1,p)`` when ``mult``
    is None, else the subgroup generated by ``x -> mult*x`` and the
    translation."""
    p = int(p)
    if not _is_prime(p):
        raise ValueError(f"affine groups need a prime v, got {p}")
    a = _primitive_root(p) if mult is None else int(mult) % p
    if a == 0:
        raise ValueError("multiplier must be nonzero mod p")
    x = np.arange(p, dtype=np.int64)
    return PermGroup(p, [(x + 1) % p, (a * x) % p],
                     name="affine" if mult is None else f"affine:{a}")


def multiplier_group(p, mult):
    p = int(p)
    if not _is_prime(p):
        raise ValueError(f"multiplier groups need a prime v, got {p}")
    x = np.arange(p, dtype=np.int64)
    return PermGroup(p, [(int(mult) * x) % p], name=f"mult:{mult}")


_CYC = re.compile(r"\(([^()]*)\)")


def group_from_cycles(v, text, name=None):
    """``(0 1 2)(3 4);(5 6)``: generators separated by ``;``, each a
    product of disjoint cycles in 0-based point labels."""
    gens = []
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        p = np.arange(v, dtype=np.int64)
        for cyc in _CYC.findall(part):
            pts = [int(x) for x in cyc.replace(",", " ").split()]
            if len(pts) != len(set(pts)) or any(not 0 <= x < v for x in pts):
                raise ValueError(f"bad cycle ({cyc})")
            for i, x in enumerate(pts):
                p[x] = pts[(i + 1) % len(pts)]
        gens.append(p)
    return PermGroup(v, gens, name=name or "gens")


def parse_design_group(v, spec):
    """Spec string (terms joined with ``+``) -> :class:`PermGroup`."""
    v = int(v)
    G = None
    for term in str(spec).split("+"):
        term = term.strip()
        if not term:
            continue
        if term == "trivial":
            g = trivial_group(v)
        elif term == "cyclic":
            g = cyclic_group(v)
        elif term.startswith("cyclic-"):
            g = cyclic_group(v, fixed=int(term[7:]))
        elif term == "dihedral":
            g = dihedral_group(v)
        elif term.startswith("prod:"):
            a, b = (int(x) for x in term[5:].split("x"))
            if a * b != v:
                raise ValueError(f"prod:{a}x{b} needs v = {a * b}, got {v}")
            g = product_cyclic_group(a, b)
        elif term == "affine":
            g = affine_group(v)
        elif term.startswith("affine:"):
            g = affine_group(v, mult=int(term[7:]))
        elif term.startswith("mult:"):
            g = multiplier_group(v, int(term[5:]))
        elif term.startswith("gens="):
            g = group_from_cycles(v, term[5:])
        else:
            raise ValueError(f"unknown group term {term!r}")
        G = g if G is None else G.join(g, name=f"{G.name}+{g.name}")
    if G is None:
        raise ValueError("empty group spec")
    return G


def builtin_specs(v):
    """The built-in group specs that apply to ``v``, for sweeps."""
    out = ["cyclic", "cyclic-1", "cyclic-2", "dihedral"]
    for a in range(2, int(v ** 0.5) + 1):
        if v % a == 0:
            out.append(f"prod:{a}x{v // a}")
    if _is_prime(v):
        out.append("affine")
    return out

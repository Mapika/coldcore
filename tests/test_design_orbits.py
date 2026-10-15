"""Group-invariant covering designs: permutation groups, subset orbits,
the quotient backend (weights by brute force, quotient cnt = full cnt),
known group-invariant optima, GPU parity of the weighted cell, and the
free greedy on C(24,8,5)."""
import itertools
import os
from math import comb

import numpy as np
import pytest

from coldcore.incidence import WeightedIncidenceBackend
from coldcore.problems.covering_design import CoveringDesignBackend
from coldcore.problems.design_orbits import (DesignOrbits,
                                             QuotientDesignBackend,
                                             orbit_search,
                                             orbit_search_points, release)
from coldcore.problems.perm_groups import (PermGroup, builtin_specs,
                                           parse_design_group)
from coldcore.search import Searcher


# ------------------------------------------------------------ groups
@pytest.mark.parametrize("v,spec,order", [
    (7, "cyclic", 7), (8, "cyclic-1", 7), (9, "cyclic-2", 7),
    (7, "dihedral", 14), (9, "prod:3x3", 9), (12, "prod:3x4", 12),
    (7, "affine", 42), (13, "affine:3", 39), (13, "mult:3", 3),
    (6, "gens=(0 1 2)(3 4);(5 0)", 48), (7, "cyclic+dihedral", 14),
    (5, "trivial", 1),
])
def test_group_orders(v, spec, order):
    G = parse_design_group(v, spec)
    assert G.order() == order
    els = G.elements()
    keys = {e.tobytes() for e in els}
    assert len(keys) == order                       # distinct elements
    for e in els[:20]:                              # closed under generators
        for g in G.gens:
            assert g[e].tobytes() in keys


def test_group_validation():
    with pytest.raises(ValueError):
        parse_design_group(8, "affine")             # not prime
    with pytest.raises(ValueError):
        parse_design_group(9, "prod:2x4")
    with pytest.raises(ValueError):
        parse_design_group(7, "nonsense")
    with pytest.raises(ValueError):
        PermGroup(5, [np.array([0, 0, 1, 2, 3])])
    big = PermGroup(9, [np.roll(np.arange(9), 1),
                        np.array([1, 0, 2, 3, 4, 5, 6, 7, 8])])   # S_9
    with pytest.raises(ValueError):
        big.order(cap=5000)
    assert "cyclic" in builtin_specs(28) and "affine" in builtin_specs(31)


@pytest.mark.parametrize("v,r,spec", [(7, 3, "cyclic"), (9, 3, "prod:3x3"),
                                      (8, 4, "dihedral"), (7, 2, "affine")])
def test_subset_orbits(v, r, spec):
    G = parse_design_group(v, spec)
    P = G.subset_orbits(r)
    assert int(P.sizes.sum()) == comb(v, r)
    # orbit sizes divide |G|, and each orbit is closed under G (brute force)
    from coldcore.problems.covering_design import all_blocks, rank_subsets
    blocks = all_blocks(v, r)
    for o in range(len(P)):
        assert G.order() % int(P.sizes[o]) == 0
        members = set(P.orbit(o).tolist())
        for e in G.elements():
            img = np.sort(e[blocks[P.orbit(o)].astype(np.int64)], axis=1)
            assert set(rank_subsets(img, v=v).tolist()) == members


# ------------------------------------------------------------ quotient
def _brute_weights(v, k, t, G):
    """w(O,U) by brute force: blocks of O containing the rep of U."""
    from coldcore.problems.covering_design import all_blocks
    orb = DesignOrbits(v, k, t, G)
    blocks = all_blocks(v, k)
    tsub = all_blocks(v, t)
    W = np.zeros((len(orb.block_part), len(orb.univ_part)), dtype=np.int64)
    for o in range(len(orb.block_part)):
        for u in range(len(orb.univ_part)):
            T = set(tsub[orb.univ_part.reps[u]].tolist())
            W[o, u] = sum(T <= set(blocks[b].tolist())
                          for b in orb.block_part.orbit(o))
    return orb, W


@pytest.mark.parametrize("v,k,t,spec", [(7, 3, 2, "cyclic"),
                                        (9, 3, 2, "prod:3x3"),
                                        (8, 4, 2, "dihedral"),
                                        (9, 4, 3, "cyclic-1")])
def test_quotient_weights_and_counts(v, k, t, spec):
    orb, W = _brute_weights(v, k, t, spec)
    cand, indptr, indices, w, uw = orb.quotient_csr(short_orbits=True)
    dense = np.zeros_like(W)
    for i, o in enumerate(cand):
        dense[o, indices[indptr[i]:indptr[i + 1]]] = w[indptr[i]:indptr[i + 1]]
    assert (dense == W).all()
    assert (uw == orb.univ_part.sizes).all()
    # quotient cnt == full cnt at the representatives, deficiency equal,
    # orbit gain == exact deficit drop on the full backend
    q = QuotientDesignBackend(v, k, t, spec, short_orbits=True)
    f = CoveringDesignBackend(v, k, t)
    rng = np.random.default_rng(1)
    for _ in range(4):
        sel = rng.choice(q.n_cand, min(q.n_cand, 3), replace=False)
        q.recount(sel)
        f.recount(q.expand(sel))
        assert (q.cnt == f.cnt[orb.univ_part.reps]).all()
        assert q.uncovered() == f.uncovered()
        assert q.block_count(sel) == len(q.expand(sel))
        q.gain_map()
        for c in range(q.n_cand):
            before = f.uncovered()
            f.ball_update(q.expand([c]), +1)
            assert q.map[c] == before - f.uncovered() == q.ball_gather([c], 0)[0]
            f.ball_update(q.expand([c]), -1)
        q.loss_map()
        for c in sel:
            before = f.uncovered()
            f.ball_update(q.expand([c]), -1)
            assert q.map[c] == f.uncovered() - before
            f.ball_update(q.expand([c]), +1)
    assert q.verify(sel) == f.verify(q.expand(sel))


def test_weighted_backend_reduces_to_unit_case():
    """With all weights 1 the weighted cell is the plain incidence cell."""
    from coldcore.incidence import SparseIncidenceBackend
    rng = np.random.default_rng(4)
    inc = np.stack([np.sort(rng.choice(40, 5, replace=False))
                    for _ in range(50)]).astype(np.int32)
    a = SparseIncidenceBackend(inc, 40, mu=2)
    b = WeightedIncidenceBackend(np.arange(0, 251, 5), inc.ravel(),
                                 np.ones(250), 40, mu=2)
    sol = rng.choice(50, 12, replace=False)
    a.recount(sol)
    b.recount(sol)
    assert (a.cnt == b.cnt).all() and a.uncovered() == b.uncovered()
    for fld in ("gain_map", "loss_map"):
        getattr(a, fld)()
        getattr(b, fld)()
        assert (a.map == b.map).all()
    probe = rng.choice(50, 9)
    for t in (0, 1, 2):
        assert (a.ball_gather(probe, t) == b.ball_gather(probe, t)).all()
    with pytest.raises(ValueError):
        WeightedIncidenceBackend([0, 2], [0, 1], [1, 0], 3)      # weight 0


# ------------------------------------------------------------ optima
QUOTIENT = [(7, 3, 2, "cyclic", 7), (13, 4, 2, "cyclic", 13),
            (21, 5, 2, "cyclic", 21), (31, 6, 2, "cyclic", 31),
            (19, 3, 2, "cyclic", 57)]


@pytest.mark.parametrize("v,k,t,spec,opt", QUOTIENT)
def test_orbit_search_reaches_optimum(v, k, t, spec, opt):
    blocks, s, q = orbit_search(v, k, t, spec, budget_s=40, seed=1,
                                notch_budget_s=5)
    assert len(blocks) == opt
    assert CoveringDesignBackend(v, k, t).verify(blocks)


def test_points_mode_short_orbit_optimum():
    """C(9,3,2) = 12 = 9 + 3 under Z_3 x Z_3: a short orbit with its
    true cost, which only the points-counting driver can prefer."""
    blocks, s, b = orbit_search_points(9, 3, 2, "prod:3x3", budget_s=30,
                                       seed=1, notch_budget_s=5)
    assert len(blocks) == 12
    assert CoveringDesignBackend(9, 3, 2).verify(blocks)


def test_release_keeps_feasibility_and_size():
    blocks, s, q = orbit_search(13, 4, 2, "cyclic", budget_s=5, seed=1,
                                notch_budget_s=2)
    rel, s2 = release(13, 4, 2, blocks, 5, seed=1, notch_budget_s=2, step0=1)
    assert len(rel) <= len(blocks) == 13
    assert CoveringDesignBackend(13, 4, 2).verify(rel)


def test_free_greedy_c24_8_5_is_759():
    b = CoveringDesignBackend(24, 8, 5)
    s = Searcher(b, seed=1)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 9) == 0
    assert len(s.code) == 759 and b.verify(s.code)


# ------------------------------------------------------------ GPU
def _sparse_plugin():
    from coldcore.incidence_gpu import default_sparse_plugin_path
    p = default_sparse_plugin_path()
    if not os.path.exists(p):
        pytest.skip(f"sparse plugin not built: {p}")


@pytest.mark.gpu
@pytest.mark.parametrize("inc_mode", [0, 1])
def test_gpu_weighted_matches_host(gpu_backend_factory, inc_mode):
    from coldcore.incidence_gpu import GpuWeightedIncidenceBackend
    _sparse_plugin()
    rng = np.random.default_rng(7 + inc_mode)
    n_cand, n_univ, mu = 400, 150, 2
    lens = rng.integers(0, 10, n_cand)
    indptr = np.concatenate(([0], np.cumsum(lens)))
    indices = np.concatenate([rng.choice(n_univ, n, replace=False)
                              for n in lens]).astype(np.int32)
    w = rng.integers(1, 4, len(indices)).astype(np.int32)
    uw = rng.integers(1, 5, n_univ).astype(np.int32)
    c = WeightedIncidenceBackend(indptr, indices, w, n_univ, uw=uw, mu=mu,
                                 extract_cap=16)
    with GpuWeightedIncidenceBackend(indptr, indices, w, n_univ, uw=uw, mu=mu,
                                     extract_cap=16, inc_mode=inc_mode) as g:
        for _ in range(3):
            sol = rng.choice(n_cand, int(rng.integers(0, 50)), replace=False)
            c.recount(sol)
            g.recount(sol)
            assert (c.read_cnt() == g.read_cnt()).all()
            assert c.uncovered() == g.uncovered()
            every = np.arange(n_cand)
            for fld in ("gain_map", "loss_map"):
                getattr(c, fld)()
                getattr(g, fld)()
                assert (c.map_read_at(every) == g.map_read_at(every)).all()
                mx = max(1, c.map_max())
                assert (c.map_hist(9, mx) == g.map_hist(9, mx)).all()
                ci, cv, cf = c.map_extract(1)
                gi, gv, gf = g.map_extract(1)
                assert cf == gf and (c.map[gi] == gv).all()
            probe = rng.choice(n_cand, 17)
            for t in (0, 1, 2, 3):
                assert (c.ball_gather(probe, t) == g.ball_gather(probe, t)).all()
            add = rng.choice(n_cand, 5)
            c.ball_update(add, +1)
            g.ball_update(add, +1)
            assert (c.read_cnt() == g.read_cnt()).all()


@pytest.mark.gpu
def test_gpu_orbit_search_c21_5_2(gpu_backend_factory):
    _sparse_plugin()
    blocks, s, q = orbit_search(21, 5, 2, "cyclic", budget_s=20, seed=1,
                                device="gpu", notch_budget_s=5)
    assert len(blocks) == 21
    assert CoveringDesignBackend(21, 5, 2).verify(blocks)


def test_quotient_backend_repr_and_label():
    q = QuotientDesignBackend(7, 3, 2, "cyclic")
    assert q.label == "C(7,3,2)/cyclic" and "nnz" in repr(q)
    assert list(itertools.chain(q.cand)) == list(range(5))

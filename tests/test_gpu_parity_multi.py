"""GPU parity for mu-fold multiple covering (dct_set_mu extension).

mu lives at the coverage-predicate seam, so it must hold for EVERY
problem family: the cases below run mu = 2 and 3 across hamming (mixed
radix included), lee and torus_linf cells against the independent
brute-force references. Function-scoped init/free per the v1 singleton
ABI; the last test pins that mu does not leak into the next cell.
"""
from contextlib import contextmanager

import numpy as np
import pytest

from coldcore.reference import (RefHammingBackend, RefLeeBackend,
                                RefTorusBackend)

pytestmark = pytest.mark.gpu

REF = {"hamming": RefHammingBackend, "lee": RefLeeBackend,
       "torus_linf": RefTorusBackend}


@contextmanager
def gpu_cell(axes, R, problem, mu):
    from coldcore.gpu import GpuBackend
    g = GpuBackend(axes, R, problem=problem, mu=mu)
    try:
        yield g
    finally:
        g.close()


CASES = [
    ("hamming", [3, 4, 5], 1),      # mixed radix
    ("hamming", [2, 2, 2, 2], 1),
    ("hamming", [3, 3, 3], 2),
    ("lee", [5, 5], 1),
    ("lee", [4, 6], 2),             # even axes: antipodal edge case
    ("torus_linf", [6, 6], 1),
    ("torus_linf", [5, 5, 5], 1),
]


@pytest.mark.parametrize("problem,axes,R", CASES)
@pytest.mark.parametrize("mu", [2, 3])
def test_fields_parity_mu(gpu_backend_factory, problem, axes, R, mu):
    r = REF[problem](axes, R, mu=mu)
    rng = np.random.default_rng(20)
    code = np.sort(rng.choice(r.space.size, size=8, replace=False))
    every = np.arange(r.space.size, dtype=np.int64)
    with gpu_cell(axes, R, problem, mu) as g:
        assert g.mu == mu
        g.recount(code)
        r.recount(code)
        assert (g.read_cnt() == r.read_cnt()).all()
        # objective: total deficit sum_x max(0, mu - cnt)
        assert g.uncovered() == r.uncovered()
        assert g.uncovered() == int(np.maximum(mu - r.cnt, 0).sum())
        for t in range(mu + 2):
            assert g.count_eq(t) == r.count_eq(t)      # count_eq stays literal
        g.gain_map()
        r.gain_map()
        assert (g.map_read_at(every) == r.map_read_at(every)).all()
        g.loss_map()
        r.loss_map()
        assert (g.map_read_at(every) == r.map_read_at(every)).all()
        for t in (0, 1):
            assert (g.ball_gather(every, t) == r.ball_gather(every, t)).all()
        assert g.map_max() == r.map_max()


@pytest.mark.parametrize("problem,axes,R", [("hamming", [3, 4, 5], 1),
                                            ("lee", [4, 6], 2),
                                            ("torus_linf", [6, 6], 1)])
def test_incremental_parity_mu(gpu_backend_factory, problem, axes, R):
    """Support-walk updates are mu-blind (they touch cnt only), but the
    derived deficit must track a from-scratch recount."""
    mu = 3
    r = REF[problem](axes, R, mu=mu)
    rng = np.random.default_rng(4)
    code = np.sort(rng.choice(r.space.size, size=10, replace=False))
    with gpu_cell(axes, R, problem, mu) as g:
        g.recount(code[:5])
        g.ball_update(code[5:], +1)
        inc, unc = g.read_cnt().copy(), g.uncovered()
        g.recount(code)
        assert (inc == g.read_cnt()).all() and unc == g.uncovered()
        r.recount(code)
        assert (inc == r.read_cnt()).all() and unc == r.uncovered()


def test_gpu_binary_double_cover_optimum(gpu_backend_factory):
    """End-to-end on GPU through the untouched search core:
    K_2^2(3,1) = 4 (exhaustively verified in tests/test_multi.py)."""
    import time

    from coldcore.search import Searcher
    with gpu_cell([2, 2, 2], 1, "hamming", 2) as g:
        s = Searcher(g, seed=3, deadline=time.time() + 120)
        s.load(np.zeros(0, dtype=np.int64))
        assert s.greedy_fill(10 ** 6) == 0
        s.peel()
        if len(s.code) > 4:
            s.ruin(len(s.code) - 4)
        assert s.lns(4, round_budget_s=30) == 0
        assert len(s.code) <= 4 and s.verify_full() == 0
        assert (g.read_cnt() >= 2).all()


def test_gpu_torus_double_domination_6x6(gpu_backend_factory):
    """K^2 of the 6x6 king torus = 8 (sphere bound 2*36/9, attained by a
    pair of shifted perfect tilings)."""
    import time

    from coldcore.search import Searcher
    with gpu_cell([6, 6], 1, "torus_linf", 2) as g:
        s = Searcher(g, seed=5, deadline=time.time() + 180)
        s.load(np.zeros(0, dtype=np.int64))
        assert s.greedy_fill(10 ** 6) == 0
        s.peel()
        if len(s.code) > 8:
            s.ruin(len(s.code) - 8)
        assert s.lns(8, round_budget_s=60) == 0
        assert len(s.code) == 8 and s.verify_full() == 0
        assert (g.read_cnt() == 2).all()


def test_mu_does_not_leak_across_cells(gpu_backend_factory):
    """v1 singleton ABI: dct_free restores mu = 1, and a plain cell built
    afterwards must compute the legacy [cnt==0]/[cnt==1] fields."""
    axes, R = [3, 3, 3], 1
    code = np.arange(0, 27, 5, dtype=np.int64)
    with gpu_cell(axes, R, "hamming", 3) as g:
        assert g._lib.dct_get_mu() == 3
    with gpu_cell(axes, R, "hamming", 1) as g:
        assert g._lib.dct_get_mu() == 1
        r = RefHammingBackend(axes, R)
        g.recount(code)
        r.recount(code)
        every = np.arange(r.space.size, dtype=np.int64)
        assert g.uncovered() == r.uncovered() == r.count_eq(0)
        g.gain_map()
        r.gain_map()
        assert (g.map_read_at(every) == r.map_read_at(every)).all()
        g.loss_map()
        r.loss_map()
        assert (g.map_read_at(every) == r.map_read_at(every)).all()

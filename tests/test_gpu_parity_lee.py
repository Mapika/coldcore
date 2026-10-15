"""GPU parity for the Lee-metric problem family (PROB_LEE).

Same parametrized pattern as the torus parity tests: tiny cells,
function-scoped init/free (v1 singleton ABI), reference = the
independent brute-force RefLeeBackend. Even axes are included to
exercise the antipodal multiplicity-1 edge case, and R=2/3 cells the
grade-truncated DP beyond one layer.
"""
from contextlib import contextmanager

import numpy as np
import pytest

from coldcore.reference import RefLeeBackend

pytestmark = pytest.mark.gpu


@contextmanager
def gpu_cell(axes, R):
    from coldcore.gpu import GpuBackend
    g = GpuBackend(axes, R, problem="lee")
    try:
        yield g
    finally:
        g.close()


CASES = [
    ([5, 5], 1),        # the Golomb-Welch cell
    ([4, 6], 2),        # even axes: antipodal digit counts once
    ([3, 4, 5], 1),     # mixed radix
    ([4, 4, 4], 3),     # deep grades; per-axis weight capped at W=2
]


@pytest.mark.parametrize("axes,R", CASES)
def test_fields_parity(gpu_backend_factory, axes, R):
    r = RefLeeBackend(axes, R)
    rng = np.random.default_rng(20)
    code = np.sort(rng.choice(r.space.size, size=6, replace=False))
    with gpu_cell(axes, R) as g:
        g.recount(code)
        r.recount(code)
        assert (g.read_cnt() == r.read_cnt()).all()
        for t in (0, 1, 2):
            assert g.count_eq(t) == r.count_eq(t)
        every = np.arange(r.space.size, dtype=np.int64)
        g.gain_map()
        r.gain_map()
        assert (g.map_read_at(every) == r.map_read_at(every)).all()
        g.loss_map()
        r.loss_map()
        assert (g.map_read_at(code) == r.map_read_at(code)).all()
        probe = np.sort(rng.choice(r.space.size, size=15, replace=False))
        for t in (0, 1):
            assert (g.ball_gather(probe, t) == r.ball_gather(probe, t)).all()


def test_incremental_parity(gpu_backend_factory):
    """GPU support-walk updates against a from-scratch GPU recount and
    the reference, on an even-axis R=2 cell."""
    axes, R = [4, 6], 2
    r = RefLeeBackend(axes, R)
    rng = np.random.default_rng(4)
    code = np.sort(rng.choice(r.space.size, size=8, replace=False))
    with gpu_cell(axes, R) as g:
        g.recount(code[:4])
        g.ball_update(code[4:], +1)
        inc = g.read_cnt().copy()
        g.recount(code)
        assert (inc == g.read_cnt()).all()
        r.recount(code)
        assert (inc == r.read_cnt()).all()


def test_lee_search_finds_perfect_5(gpu_backend_factory):
    """End-to-end on GPU: the perfect Lee code of Z_5^2, R=1 (size 5 =
    the sphere bound 25/|B_1|) through the untouched search core."""
    from coldcore.search import Searcher
    with gpu_cell([5, 5], 1) as g:
        s = Searcher(g, seed=7)
        s.load(np.zeros(0, dtype=np.int64))
        assert s.greedy_fill(10 ** 6) == 0
        s.peel()
        unc = s.lns(5, round_budget_s=45)
        assert unc == 0 and len(s.code) == 5
        assert s.verify_full() == 0

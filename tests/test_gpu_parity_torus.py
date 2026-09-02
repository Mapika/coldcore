"""GPU parity on the other v1 problems: torus domination + mixed radix.

Function-scoped cells (init/free per test) because the v1 ABI is a
per-process singleton. All cells are tiny (<= 81 points, seconds).
"""
from contextlib import contextmanager

import numpy as np
import pytest

from coldcore.reference import RefHammingBackend, RefTorusBackend

pytestmark = pytest.mark.gpu


@contextmanager
def gpu_cell(axes, R, problem):
    from coldcore.gpu import GpuBackend
    g = GpuBackend(axes, R, problem=problem)
    try:
        yield g
    finally:
        g.close()


CASES = [
    ("torus_linf", [9, 9], 1, RefTorusBackend),
    ("torus_linf", [5, 7], 1, RefTorusBackend),
    ("hamming", [2, 3, 4], 1, RefHammingBackend),
]


@pytest.mark.parametrize("problem,axes,R,refcls", CASES)
def test_fields_parity(gpu_backend_factory, problem, axes, R, refcls):
    r = refcls(axes, R)
    rng = np.random.default_rng(20)
    code = np.sort(rng.choice(r.space.size, size=6, replace=False))
    with gpu_cell(axes, R, problem) as g:
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


def test_torus_search_finds_9(gpu_backend_factory):
    """End-to-end on GPU: 9-vertex domination of the 9x9 king torus."""
    from coldcore.search import Searcher
    with gpu_cell([9, 9], 1, "torus_linf") as g:
        s = Searcher(g, seed=7)
        s.load(np.zeros(0, dtype=np.int64))
        assert s.greedy_fill(10 ** 6) == 0
        s.peel()
        unc = s.lns(9, round_budget_s=45)
        assert unc == 0 and len(s.code) <= 9
        assert s.verify_full() == 0

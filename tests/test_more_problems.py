"""Torus-domination and mixed-radix Hamming reference backends + search.

Same field identities as the covering tests, on the other v1 problems;
ground truth for the torus: the 9x9 king torus has a perfect 9-vertex
dominating set (3x3 shifted-diagonal pattern).
"""
import time

import numpy as np
import pytest

from coldcore.patterns import support_patterns
from coldcore.reference import RefHammingBackend, RefTorusBackend
from coldcore.search import Searcher


@pytest.fixture(params=["torus99", "mixed234"])
def b(request):
    if request.param == "torus99":
        return RefTorusBackend([9, 9], 1)
    return RefHammingBackend([2, 3, 4], 1)


def _rand_code(b, m, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(b.space.size, size=m, replace=False))


def test_support_table_matches_ball(b):
    """The packed support table enumerates exactly the coverage set."""
    problem = "torus_linf" if isinstance(b, RefTorusBackend) else "hamming"
    pats = support_patterns(b.space.axes, b.R, problem)
    axes = np.array(b.space.axes, dtype=np.int64)
    center = np.array([a // 2 for a in b.space.axes], dtype=np.int64)
    seen = set()
    for p in pats:
        w = center.copy()
        v = int(p)
        while v:
            byte = v & 0xFF
            pos, delta = byte >> 4, byte & 0xF
            w[pos] = (w[pos] + delta) % axes[pos]
            v >>= 8
        seen.add(int(b.space.word_index(w[None, :])[0]))
    cidx = int(b.space.word_index(center[None, :])[0])
    assert seen == set(b.ball_indices(cidx).tolist())


def test_field_identities(b):
    code = _rand_code(b, 6, 1)
    b.recount(code)
    every = np.arange(b.space.size, dtype=np.int64)
    b.gain_map()
    assert (b.map_read_at(every) == b.ball_gather(every, 0)).all()
    b.loss_map()
    assert (b.map_read_at(code) == b.ball_gather(code, 1)).all()


def test_incremental_matches_recount(b):
    code = _rand_code(b, 8, 2)
    b.recount(code[:4])
    b.ball_update(code[4:], +1)
    inc = b.cnt.copy()
    b.recount(code)
    assert (inc == b.cnt).all()


def test_torus_king_domination_optimum():
    """Search must reproduce gamma(C9 x C9 king) = 9."""
    b = RefTorusBackend([9, 9], 1)
    s = Searcher(b, seed=7, deadline=time.time() + 90)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    s.peel()
    assert s.verify_full() == 0
    unc = s.lns(9, round_budget_s=45)
    assert unc == 0 and len(s.code) <= 9
    assert s.verify_full() == 0


def test_mixed_radix_search_covers():
    b = RefHammingBackend([2, 3, 4, 5], 1)
    s = Searcher(b, seed=8)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    s.peel()
    assert s.verify_full() == 0


def test_search_without_hist_capability():
    """map_hist is optional in v1: greedy must work without it."""
    b = RefTorusBackend([7, 7], 1)
    b.supports_hist = False
    s = Searcher(b, seed=9)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    assert s.verify_full() == 0

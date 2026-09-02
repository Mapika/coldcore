"""Search core end-to-end on the reference backend — no GPU required.

Ground truth: K_2(4,1) = 4 (binary, length 4, radius 1) and
K_3(4,1) = 9 (the perfect ternary Hamming code).
"""
import time

import numpy as np

from coldcore.reference import RefCoveringBackend
from coldcore.search import Searcher


def _valid_cover(b, code):
    b.recount(code)
    return b.uncovered() == 0


def test_greedy_builds_cover():
    b = RefCoveringBackend(3, 4, 1)
    s = Searcher(b, seed=1)
    s.load(np.zeros(0, dtype=np.int64))
    unc = s.greedy_fill(target_m=10 ** 6)
    assert unc == 0
    assert _valid_cover(b, s.code)
    # exact greedy on 81 words should not be far off the optimum of 9
    assert len(s.code) <= 15


def test_peel_keeps_feasibility():
    b = RefCoveringBackend(3, 4, 1)
    s = Searcher(b, seed=2)
    rng = np.random.default_rng(0)
    # deliberately redundant cover: all 81 words minus a few
    code = np.sort(rng.choice(81, size=60, replace=False))
    s.load(code)
    assert s.uncov == 0
    removed = s.peel()
    assert removed > 0
    assert s.verify_full() == 0


def test_lns_solves_binary_optimum():
    # K_2(4,1) = 4 on a 16-word space: LNS must find it fast
    b = RefCoveringBackend(2, 4, 1)
    s = Searcher(b, seed=3, deadline=time.time() + 60)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 6)
    assert s.uncov == 0
    s.peel()
    unc = s.lns(4, round_budget_s=30)
    assert unc == 0
    assert len(s.code) <= 4
    assert s.verify_full() == 0


def test_descend_ternary_reaches_low_m():
    b = RefCoveringBackend(3, 4, 1)
    solved = []
    s = Searcher(b, seed=4, deadline=time.time() + 90,
                 on_solved=lambda m, idx: solved.append((m, idx.copy())))
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 6)
    assert s.uncov == 0
    s.descend(floor=9, notch_budget_s=10)
    assert solved, "descend must report at least the initial solution"
    ms = [m for m, _ in solved]
    assert ms == sorted(ms, reverse=True)      # monotone improvements
    m_best, code_best = solved[-1]
    assert _valid_cover(b, code_best)
    assert m_best <= 11                        # optimum is 9; 11 = generous


def test_ruin_modes_and_restore():
    b = RefCoveringBackend(3, 4, 1)
    s = Searcher(b, seed=5)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 6)
    m0 = len(s.code)
    for mode in ("low", "random", "cluster", "hole"):
        s.ruin(3, mode)
        assert len(s.code) == m0 - 3
        assert s.uncov == b.uncovered()        # incremental cnt stayed exact
        s.greedy_fill(10 ** 6)
        if s.uncov:                            # repair fully for next mode
            s.lns(m0, round_budget_s=10)
        m0 = len(s.code)
        assert s.verify_full() == 0

"""The wall-breaking passes on the GPU plugin.

Skipped automatically when no CUDA device / no built plugin (conftest).
Seconds-scale and megabytes of device memory, so it is safe alongside
other GPU tenants.

What only a GPU run can check: that ``uncovered_words`` reads the real
device ``cnt``, that a *failed* ``two_swap`` puts the device field back
bit-for-bit (the undo is a chain of incremental ``ball_update`` calls,
not a recount), and that ``map_extract``'s unordered device slot order
does not leak into the pass's decisions.

NOTE: the vendored plugin ABI is a per-process singleton — this module
keeps exactly one live cell.
"""
import time

import numpy as np
import pytest

from coldcore.reference import RefCoveringBackend
from coldcore.search import Searcher

pytestmark = pytest.mark.gpu

Q, N, R, OPT = 3, 4, 1, 9          # K_3(4,1) = 9


@pytest.fixture(scope="module")
def cell(gpu_backend_factory):
    g = gpu_backend_factory(Q, N, R)
    yield g
    g.close()


def _is_cover(code):
    """Brute force, on the CPU, from raw balls — never the GPU's cnt."""
    r = RefCoveringBackend(Q, N, R)
    seen = np.zeros(r.space.size, dtype=bool)
    for w in np.atleast_1d(np.asarray(code, dtype=np.int64)):
        seen[r.ball_indices(int(w))] = True
    return bool(seen.all())


def _start(size, m, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(size, size=m, replace=False)).astype(np.int64)


def _searcher(b, seed=1, secs=90):
    return Searcher(b, seed=seed, deadline=time.time() + secs)


def test_uncovered_words_match_the_device_field(cell):
    s = _searcher(cell, seed=1)
    s.load(_start(cell.space.size, 6, 3))
    assert s.uncov > 0 and s.supports_deficit()
    truth = np.flatnonzero(np.asarray(cell.read_cnt()) == 0)
    words = s.uncovered_words(limit=len(truth) + 4)
    assert sorted(words.tolist()) == sorted(truth.tolist())


def test_deficit_ruin_runs_on_the_device(cell):
    s = _searcher(cell, seed=2)
    s.load(_start(cell.space.size, 12, 4))
    removed = s.ruin(4, "deficit")
    assert len(removed) == 4
    assert s.stats["deficit_ruins"] == 1
    assert s.uncov == s.verify_full()


def test_failed_two_swap_restores_the_device_field_exactly(cell):
    """The undo path is incremental ball_updates, so a bug here would
    corrupt cnt silently. Compare the whole field, byte for byte."""
    s = _searcher(cell, seed=3)
    s.load(_start(cell.space.size, 5, 7))       # far too small to cover
    before_cnt = np.asarray(cell.read_cnt()).copy()
    before_code, u0 = s.code.copy(), s.uncov
    unc = s.two_swap(5, budget_s=5.0, max_uncov=10 ** 6, seed=11)
    assert s.code.tolist() == before_code.tolist() or unc < u0
    if unc == u0:                                # nothing better found
        assert (np.asarray(cell.read_cnt()) == before_cnt).all()
    assert unc == s.verify_full()


def test_two_swap_and_break_wall_reach_the_optimum(cell):
    for seed in (1, 2):
        s = _searcher(cell, seed=seed)
        s.load(_start(cell.space.size, OPT + 2, seed))
        unc = s.break_wall(OPT, budget_s=20, burst_s=2.5, seed=seed)
        assert unc == 0, f"gpu wall-breaker stalled at {unc}"
        assert len(s.code) == OPT and s.verify_full() == 0
        assert _is_cover(s.code)


def test_shell_polish_on_the_device(cell):
    """The local neighbourhood costs a whole candidate batch in one
    device gather. What is checked here is that the device path is
    exact and actually accepts moves; the head-to-head against the pool
    neighbourhood is made where it is not a coin flip — on the C9 x C9
    plateau in ``test_search_wall.py`` and in ``scripts/bench_wall.py``.
    This cell (81 words, optimum 9) is small enough that either
    neighbourhood may solve it outright."""
    for hood in ("pool", "shell"):
        s = _searcher(cell, seed=6)
        s.load(_start(cell.space.size, OPT, 13))
        unc = s.anneal(OPT, budget_s=8, mode="lahc", neighbourhood=hood,
                       seed=6)
        assert unc == s.verify_full() and len(s.code) == OPT
        assert (unc == 0) == _is_cover(s.code)
        if hood == "shell":
            assert s.stats["shift_moves"] > 0
            assert s.stats["shift_accepts"] > 0


def test_polish_auto_cadence_on_the_device(cell):
    """On a small GPU cell a transform costs about what a move costs, so
    the measured cadence must land far below the fixed default of 64."""
    s = _searcher(cell, seed=5)
    s.load(_start(cell.space.size, OPT, 9))
    unc = s.anneal(OPT, budget_s=10, mode="lahc", pool_refresh="auto", seed=5)
    assert unc == s.verify_full()
    assert 1 <= s.stats["pool_refresh"] < 64

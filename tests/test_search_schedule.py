"""v0.3 descent schedule options and the random-repair ablation, on the
reference backends (no GPU required).

Every "solved" claim is re-checked with a brute-force union of balls.
"""
import time

import numpy as np
import pytest

from coldcore.reference import RefCoveringBackend, RefLeeBackend
from coldcore.reference.grid import RefGridBackend
from coldcore.search import Searcher


def _is_cover(b, code):
    seen = np.zeros(b.space.size, dtype=bool)
    for w in np.atleast_1d(np.asarray(code, dtype=np.int64)):
        seen[b.ball_indices(int(w))] = True
    return bool(seen.all())


def _built(b, seed=1, secs=60, **kw):
    solved = []
    s = Searcher(b, seed=seed, deadline=time.time() + secs,
                 on_solved=lambda m, idx: solved.append((m, idx.copy())),
                 **kw)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 6)
    return s, solved


# ------------------------------------------------------------ schedule
def test_v03_schedule_is_valid_and_monotone():
    b = RefCoveringBackend(3, 4, 1)
    s, solved = _built(b, seed=4, secs=20)
    s.loop_reserve_s, s.build_reserve_s = 2.0, 1.0
    t0 = time.time()
    m = s.descend(floor=9, notch_budget_s=2, salvage=True, persist=True,
                  patience=True, reserve_s=1.0, min_notch_s=0.5)
    ms = [x for x, _ in solved]
    assert ms == sorted(ms, reverse=True)
    for size, code in solved:
        assert len(code) == size and _is_cover(b, code)
    assert m == ms[-1] <= 11
    # persist: the descent either reached the floor or used the deadline
    assert m == 9 or time.time() - t0 > 15


def test_salvage_completes_a_failed_notch():
    """A state of size M - step with u uncovered words is completed by
    the exact greedy; when the result is smaller than M it is returned."""
    b = RefCoveringBackend(2, 6, 1)
    s, _ = _built(b, seed=2)
    s.peel()
    M = len(s.code)
    s.ruin(4, "low")                    # M - 4 words, some uncovered
    assert s.uncov > 0
    got = s._salvage(M + 10)            # any completion beats M + 10
    assert got is not None and got == len(s.code)
    assert s.verify_full() == 0 and _is_cover(b, s.code)
    assert s._salvage(1) is None        # nothing to do when feasible


def test_lns_stall_ends_the_pass():
    """K_2(6,1) = 12: at M = 8 no cover exists, so with stall_s the pass
    stops long before its round budget."""
    b = RefCoveringBackend(2, 6, 1)
    s, _ = _built(b, seed=3)
    s.ruin(len(s.code) - 8, "low")
    t0 = time.time()
    unc = s.lns(8, round_budget_s=60, kmin=1, kmax=2, stall_s=0.5)
    assert unc > 0
    assert time.time() - t0 < 30
    assert s.stats.get("stalls", 0) == 1


def test_portfolio_stall_ends_the_pass():
    """Same infeasible target as above, with a portfolio of streams: the
    stall limit applies to the portfolio too."""
    b = RefCoveringBackend(2, 6, 1)
    s, _ = _built(b, seed=3)
    s.ruin(len(s.code) - 8, "low")
    t0 = time.time()
    unc = s.portfolio(8, budget_s=60, streams=2, slice_s=0.2, kmin=1,
                      kmax=2, stall_s=1.0)
    assert unc > 0
    assert time.time() - t0 < 30


def test_default_descent_schedule_is_unchanged():
    """The library default stays the v0.2 schedule: the new keywords at
    their defaults move nothing."""
    out = []
    for kw in ({}, {"salvage": False, "persist": False, "patience": False,
                    "reserve_s": 30.0, "min_notch_s": 20.0}):
        b = RefCoveringBackend(3, 4, 1)
        s, _ = _built(b, seed=4, secs=90)
        m = s.descend(floor=9, notch_budget_s=3, **kw)
        out.append((m, s.code.tolist()))
    assert out[0] == out[1]


# ------------------------------------------------------- random repair
def test_random_repair_inserts_only_coverers_of_deficient_words():
    b = RefCoveringBackend(3, 4, 1)
    s = Searcher(b, seed=5, repair="random")
    rng = np.random.default_rng(0)
    for u in rng.choice(b.space.size, size=10, replace=False):
        for _ in range(20):
            w = s._random_coverer(int(u))
            assert int(u) in set(b.ball_indices(w).tolist())


def test_random_coverer_is_uniform_over_the_ball():
    b = RefCoveringBackend(3, 4, 1)
    s = Searcher(b, seed=6, repair="random")
    draws = [s._random_coverer(40) for _ in range(9000)]
    vals, cnt = np.unique(draws, return_counts=True)
    assert len(vals) == 9                        # |B_1| = 1 + 4*2
    assert cnt.min() > 800 and cnt.max() < 1200  # ~1000 each


def test_random_coverer_on_a_small_open_grid():
    """Open 2x2 board, R = 1: all four cells cover the corner (1, 1), and
    the draw is uniform over them (the wrapped pattern sign is
    ambiguous when an axis has at most 2R + 1 cells)."""
    b = RefGridBackend([2, 2], 1)
    s = Searcher(b, seed=8, repair="random")
    u = b.space.size - 1
    draws = [s._random_coverer(u) for _ in range(4000)]
    vals, cnt = np.unique(draws, return_counts=True)
    assert vals.tolist() == [0, 1, 2, 3]
    assert cnt.min() > 850 and cnt.max() < 1150
    for w in vals:
        assert u in set(b.ball_indices(int(w)).tolist())


def test_random_coverer_on_an_open_grid_is_uniform():
    b = RefGridBackend([6, 6], 1)
    s = Searcher(b, seed=9, repair="random")
    u = 0                                        # corner: 4 coverers
    vals, cnt = np.unique([s._random_coverer(u) for _ in range(4000)],
                          return_counts=True)
    assert len(vals) == 4 and cnt.min() > 850
    u = 2 * 6 + 2                                # interior: 9 coverers
    vals = np.unique([s._random_coverer(u) for _ in range(4000)])
    assert len(vals) == 9
    for w in vals:
        assert u in set(b.ball_indices(int(w)).tolist())


@pytest.mark.parametrize("make", [lambda: RefCoveringBackend(3, 4, 1),
                                  lambda: RefLeeBackend([5, 5], 1)],
                         ids=["hamming", "lee"])
def test_random_repair_builds_valid_covers(make):
    b = make()
    s, solved = _built(b, seed=7, secs=60, repair="random")
    assert s.uncov == 0 and s.verify_full() == 0
    assert _is_cover(b, s.code)
    unc = s.lns(len(s.code), round_budget_s=2)
    assert unc == 0 and _is_cover(b, s.code)


def test_random_repair_is_worse_than_exact_on_average():
    """Sanity check of the ablation: over a few seeds the random build is
    larger than the exact greedy build."""
    sizes = {"exact": [], "random": []}
    for mode in sizes:
        for seed in range(4):
            b = RefCoveringBackend(3, 4, 1)
            s, _ = _built(b, seed=seed, repair=mode)
            sizes[mode].append(len(s.code))
    assert np.mean(sizes["random"]) > np.mean(sizes["exact"])


def test_unknown_repair_mode_rejected():
    with pytest.raises(ValueError):
        Searcher(RefCoveringBackend(2, 3, 1), repair="greedy")


def test_random_repair_needs_a_radius():
    from coldcore.problems.covering_design import CoveringDesignBackend
    b = CoveringDesignBackend(7, 3, 2, device="cpu")
    s = Searcher(b, seed=1, repair="random")
    with pytest.raises(ValueError):
        s._ball_table()


def test_cli_repair_random_and_legacy_descent(tmp_path, capsys):
    from coldcore.cli import main
    for extra in (["--repair", "random"], ["--legacy-descent"],
                  ["--start", "greedy"]):
        out = tmp_path / "sol.txt"
        main(["solve", "--backend", "ref", "-p", "q=2,n=4,R=1",
              "--floor", "4", "--notch-budget", "1", "--hours", "0.01",
              "--seed", "3", "--out", str(out)] + extra)
        text = capsys.readouterr().out
        assert "descent ended at M=" in text
        words = np.loadtxt(out, dtype=np.int64, ndmin=2)
        b = RefCoveringBackend(2, 4, 1)
        assert _is_cover(b, b.word_index(words))


def test_carry_at_most_once_in_a_row():
    """K_2(6,1) = 12: a floor of 8 forces a run of failed notches; carried
    attempts never follow each other, and every reported code is valid."""
    b = RefCoveringBackend(2, 6, 1)
    s, solved = _built(b, seed=8, secs=12)
    s.loop_reserve_s, s.build_reserve_s = 0.5, 0.2
    s.descend(floor=8, notch_budget_s=0.5, step0=1, kmin=1, carry=True,
              salvage=True, persist=True, patience=True, reserve_s=0.5,
              min_notch_s=0.1)
    fails = s.stats.get("notch_fails", 0)
    carried = s.stats.get("carried", 0)
    assert fails >= 3
    assert 1 <= carried <= (fails + 1) // 2
    for size, code in solved:
        assert len(code) == size and _is_cover(b, code)

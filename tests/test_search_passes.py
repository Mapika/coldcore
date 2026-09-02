"""The added search passes on the reference backends — no GPU required.

Covers the three additions to the pass library:

  * adaptive (bandit) ruin-operator selection inside ``lns``
  * ``anneal`` — Metropolis / late-acceptance polish at fixed M
  * ``portfolio`` — time-sliced restarts around a shared incumbent

Ground truth for the standard smokes: K_2(4,1) = 4 (hamming),
gamma(C9 x C9 king) = 9 (torus_linf), and the Golomb-Welch perfect Lee
code in Z_5^2 at R=1 (size 5).

Every "solved" claim is re-checked with :func:`_is_cover`, which unions
raw balls and never looks at the backend's incremental ``cnt`` — so a
bookkeeping bug cannot make an invalid cover look valid.
"""
import time

import numpy as np
import pytest

from coldcore.reference import (RefCoveringBackend, RefLeeBackend,
                                RefTorusBackend)
from coldcore.search import RUIN_MODES, OperatorBandit, Searcher

# (name, backend factory, optimum) — the standard smokes
SMOKES = [
    ("hamming K_2(4,1)", lambda: RefCoveringBackend(2, 4, 1), 4),
    ("torus 9x9 king", lambda: RefTorusBackend([9, 9], 1), 9),
    ("lee Z_5^2 R=1", lambda: RefLeeBackend([5, 5], 1), 5),
]


def _is_cover(b, code):
    """Independent brute-force check: the union of the balls is the space."""
    seen = np.zeros(b.space.size, dtype=bool)
    for w in np.atleast_1d(np.asarray(code, dtype=np.int64)):
        seen[b.ball_indices(int(w))] = True
    return bool(seen.all())


def _random_start(b, m, seed):
    """A random size-m point set — almost never a cover, so the pass
    under test has to do the work rather than inherit it from greedy."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(b.space.size, size=m, replace=False))


def _searcher(b, seed=1, secs=60, log=None, on_solved=None):
    return Searcher(b, seed=seed, deadline=time.time() + secs,
                    on_log=(log.append if log is not None else None),
                    on_solved=on_solved)


# ---------------------------------------------------------------- bandit
def test_bandit_starts_at_the_proven_mix():
    """Untouched scores reproduce the hand-tuned frequencies: `hole` is
    drawn about twice as often as the other three."""
    ban = OperatorBandit(np.random.default_rng(0), floor=0.0)
    p = ban.probabilities()
    assert list(ban.ops) == list(RUIN_MODES)
    assert p.sum() == pytest.approx(1.0)
    assert p[ban.ops.index("hole")] == pytest.approx(0.4)
    assert p[ban.ops.index("low")] == pytest.approx(0.2)


def test_bandit_shifts_weight_towards_rewarded_operator():
    ban = OperatorBandit(np.random.default_rng(0), floor=0.15, decay=0.5)
    before = ban.probabilities()[ban.ops.index("random")]
    for _ in range(20):
        ban.update("random", OperatorBandit.REWARD_BEST)
        ban.update("low", OperatorBandit.REWARD_WORSE)
    after = ban.probabilities()
    assert after[ban.ops.index("random")] > before
    assert after[ban.ops.index("random")] == max(after)
    # exploration floor: even a fully punished operator keeps a share
    assert after[ban.ops.index("low")] >= 0.15 / len(ban.ops) - 1e-12


def test_bandit_select_and_summary_are_seeded():
    picks = []
    for _ in range(2):
        ban = OperatorBandit(np.random.default_rng(7))
        seq = [ban.select() for _ in range(50)]
        assert set(seq) <= set(RUIN_MODES)
        for op in seq:
            ban.update(op, OperatorBandit.REWARD_EQUAL)
        picks.append(seq)
        summ = ban.summary()
        assert sum(v["uses"] for v in summ.values()) == 50
        assert all(0.0 <= v["p"] <= 1.0 for v in summ.values())
    assert picks[0] == picks[1]


def test_adaptive_lns_solves_and_records_operator_stats():
    """Adaptive selection must still reach the optimum, and must leave an
    auditable per-operator record behind."""
    b = RefCoveringBackend(2, 4, 1)
    s = _searcher(b, seed=3)
    s.load(_random_start(b, 4, 11))
    assert s.lns(4, round_budget_s=10, kmin=1, kmax=2,
                 select="adaptive", seed=5) == 0
    assert _is_cover(b, s.code) and len(s.code) <= 4
    ops = s.stats["ops"]
    assert set(ops) == set(RUIN_MODES)
    assert sum(v["uses"] for v in ops.values()) == s.stats["rounds"]


def test_fixed_selection_is_the_default_and_unchanged():
    """The default LNS path must be bit-identical to the explicit
    ``select="fixed"`` one — no new randomness, no reordered draws."""
    out = []
    for kw in ({}, {"select": "fixed"}):
        b = RefCoveringBackend(3, 4, 1)
        s = _searcher(b, seed=6)
        s.load(_random_start(b, 12, 4))
        unc = s.lns(12, round_budget_s=10, kmin=2, kmax=4, **kw)
        out.append((unc, s.code.tolist(), s.stats["rounds"]))
    assert out[0] == out[1]
    assert "ops" not in s.stats            # fixed selection stays bandit-free


def test_unknown_selection_rejected():
    s = _searcher(RefCoveringBackend(2, 4, 1))
    s.load(np.zeros(0, dtype=np.int64))
    with pytest.raises(ValueError):
        s.lns(4, round_budget_s=1, select="bandit")


# ---------------------------------------------------------------- polish
@pytest.mark.parametrize("name,make,opt", SMOKES, ids=[s[0] for s in SMOKES])
@pytest.mark.parametrize("mode", ["anneal", "lahc"])
def test_polish_reaches_the_known_optimum(name, make, opt, mode):
    """From a random infeasible point set of exactly the optimal size,
    the polish pass must walk all the way to a valid cover."""
    b = make()
    for start in (1, 2, 3):
        s = _searcher(b, seed=start)
        s.load(_random_start(b, opt, start))
        assert s.uncov > 0, "start should be infeasible or the test is void"
        unc = s.anneal(opt, budget_s=15, mode=mode, seed=start)
        assert unc == 0, f"{name}/{mode}: stalled at {unc} uncovered"
        assert len(s.code) == opt
        assert _is_cover(b, s.code)
        assert s.verify_full() == 0


@pytest.mark.parametrize("mode", ["anneal", "lahc"])
def test_polish_never_reports_a_bad_cover(mode):
    """Under a budget far too small to succeed, the pass must still hand
    back an exact state: the returned count is the true uncovered count,
    and it only ever claims zero when the cover really is one."""
    b = RefCoveringBackend(3, 4, 1)
    for start in range(6):
        s = _searcher(b, seed=start)
        s.load(_random_start(b, 9, 100 + start))
        unc = s.anneal(9, budget_s=0.02, mode=mode, seed=start,
                       pool_refresh=1, sample=2)
        assert unc == s.verify_full()
        assert (unc == 0) == _is_cover(b, s.code)
        assert len(s.code) == 9


@pytest.mark.parametrize("mode", ["anneal", "lahc"])
def test_polish_escapes_a_swap_local_optimum(mode):
    """The point of the pass. This Z_5^2 state leaves 3 points uncovered
    and *no* swap improves it — a strict local optimum of the polish
    pass's own neighbourhood, so a downhill-only version of it would sit
    there forever. The test proves the wall by brute force first, then
    requires the polish to accept its way off it to a perfect code."""
    b = RefLeeBackend([5, 5], 1)
    wall = np.array([0, 1, 8, 17, 19], dtype=np.int64)
    b.recount(wall)
    u0 = b.uncovered()
    assert u0 == 3
    for i in range(len(wall)):
        for v in range(b.space.size):
            if v in wall:
                continue
            swapped = wall.copy()
            swapped[i] = v
            b.recount(swapped)
            assert b.uncovered() >= u0, "start state is not a wall"

    s = _searcher(b, seed=1)
    s.load(wall)
    assert s.uncov == u0
    assert s.anneal(5, budget_s=15, mode=mode, seed=2) == 0
    assert _is_cover(b, s.code) and len(s.code) == 5


def test_polish_lahc_is_reproducible_from_its_seed():
    b = RefCoveringBackend(3, 4, 1)
    runs = []
    for _ in range(2):
        s = _searcher(b, seed=99)
        s.load(_random_start(b, 9, 21))
        unc = s.anneal(9, budget_s=15, mode="lahc", seed=4)
        runs.append((unc, s.code.tolist(), s.stats["moves"]))
    assert runs[0] == runs[1]
    assert runs[0][2] > 0, "the pass must actually have moved"


def test_polish_honours_its_tuning_knobs():
    """Every knob off its default, on a case that must still solve: tiny
    LAHC list, one-move pool refresh, minimal sample, steep schedule,
    and per-move logging."""
    b = RefTorusBackend([9, 9], 1)
    log = []
    s = _searcher(b, seed=8, log=log)
    s.load(_random_start(b, 9, 31))
    unc = s.anneal(9, budget_s=15, mode="anneal", t0=8.0, t_end=0.001,
                   lahc_len=1, pool_refresh=1, sample=2, seed=8, log_every=1)
    assert unc == 0 and _is_cover(b, s.code)
    assert s.stats["transforms"] >= s.stats["accepts"]   # refreshed per move
    assert any("move=" in line for line in log)
    assert log[-1].startswith("anneal M=9 done")


def test_polish_shrinks_an_oversized_solution_to_M():
    """Called at an M below the current size, the pass ruins down first."""
    b = RefCoveringBackend(3, 4, 1)
    s = _searcher(b, seed=2)
    s.load(_random_start(b, 30, 2))         # redundant cover of Z_3^4
    assert s.uncov == 0 and len(s.code) == 30
    unc = s.anneal(9, budget_s=15, mode="lahc", seed=3)
    assert len(s.code) == 9
    assert unc == s.verify_full() and (unc == 0) == _is_cover(b, s.code)


def test_unknown_polish_mode_rejected():
    s = _searcher(RefCoveringBackend(2, 4, 1))
    s.load(np.zeros(0, dtype=np.int64))
    with pytest.raises(ValueError):
        s.anneal(4, budget_s=1, mode="tabu")


# ------------------------------------------------------------- portfolio
@pytest.mark.parametrize("name,make,opt", SMOKES, ids=[s[0] for s in SMOKES])
def test_portfolio_reaches_the_known_optimum(name, make, opt):
    b = make()
    s = _searcher(b, seed=4)
    s.load(_random_start(b, opt, 7))
    unc = s.portfolio(opt, budget_s=20, streams=3, slice_s=0.5, kmin=1,
                      kmax=3, ruin_frac=0.4, select="adaptive", seed=5)
    assert unc == 0, f"{name}: portfolio stalled at {unc}"
    assert len(s.code) <= opt and _is_cover(b, s.code)
    assert s.verify_full() == 0


def test_portfolio_seeding_is_reproducible_and_shares_the_incumbent():
    """With a zero budget the pass only seeds its streams and returns the
    best of them — a timing-free view of the seeding and best-of logic."""
    b = RefCoveringBackend(3, 4, 1)
    runs = []
    for _ in range(2):
        s = _searcher(b, seed=12)
        s.load(_random_start(b, 11, 3))
        unc = s.portfolio(11, budget_s=0.0, streams=4, ruin_frac=0.5, seed=6)
        runs.append((unc, s.code.tolist()))
        assert unc == s.uncov == s.best_uncov
        assert unc == s.verify_full()
    assert runs[0] == runs[1]


def test_portfolio_restarts_the_worst_stream():
    """A notch it cannot make forces restart rounds; the incumbent must
    never get worse and the reported state must stay exact."""
    b = RefCoveringBackend(2, 6, 1)
    log = []
    s = _searcher(b, seed=13, log=log)
    s.load(_random_start(b, 8, 5))          # K_2(6,1) = 12: 8 is impossible
    unc = s.portfolio(8, budget_s=4, streams=3, slice_s=0.3, kmin=1, kmax=2,
                      seed=9)
    assert unc > 0 and unc == s.verify_full()
    assert s.stats["restarts"] > 0
    assert any(line.startswith("portfolio M=8") for line in log)
    assert not _is_cover(b, s.code)         # honest about the failure


# ---------------------------------------------------------------- descend
def test_descend_defaults_are_unchanged():
    """Adding the new knobs must not move the default descent one bit."""
    out = []
    for kw in ({}, {"select": "fixed", "polish": None, "polish_share": 0.5,
                    "portfolio": 1, "seed": None}):
        b = RefCoveringBackend(3, 4, 1)
        s = _searcher(b, seed=4, secs=90)
        s.load(np.zeros(0, dtype=np.int64))
        s.greedy_fill(10 ** 6)
        m = s.descend(floor=9, notch_budget_s=3, **kw)
        out.append((m, s.code.tolist()))
    assert out[0] == out[1]


@pytest.mark.parametrize("kw", [
    {"polish": "lahc"},
    {"polish": "anneal", "polish_share": 0.7},
    {"portfolio": 2},
    {"portfolio": 2, "polish": "lahc", "select": "adaptive"},
], ids=["lahc", "anneal", "portfolio", "combined"])
def test_descend_with_new_passes_stays_valid_and_monotone(kw):
    b = RefCoveringBackend(3, 4, 1)
    solved = []
    s = _searcher(b, seed=15, secs=90,
                  on_solved=lambda m, idx: solved.append((m, idx.copy())))
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 6)
    m = s.descend(floor=9, notch_budget_s=3, seed=2, **kw)
    assert solved, "descend must report at least the initial solution"
    ms = [x for x, _ in solved]
    assert ms == sorted(ms, reverse=True)
    for size, code in solved:
        assert len(code) == size and _is_cover(b, code)
    assert m == ms[-1] <= 11                # optimum is 9; 11 is generous


def test_descend_rejects_unknown_polish():
    b = RefCoveringBackend(2, 4, 1)
    s = _searcher(b, seed=1)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 6)
    with pytest.raises(ValueError):
        s.descend(floor=1, notch_budget_s=1, polish="greedy")


# -------------------------------------------------------------------- cli
def test_cli_defaults_and_new_flags(tmp_path, capsys):
    """The new flags reach the descent, and the run still writes a
    verified solution file."""
    from coldcore.cli import main

    for extra in ([], ["--ruin-select", "adaptive", "--polish", "lahc",
                       "--polish-share", "0.4", "--portfolio", "2"]):
        out = tmp_path / "sol.txt"
        main(["solve", "--backend", "ref", "-p", "q=2,n=4,R=1",
              "--floor", "4", "--notch-budget", "1", "--hours", "0.02",
              "--seed", "3", "--out", str(out)] + extra)
        text = capsys.readouterr().out
        assert "SOLVED M=" in text and "descent ended at M=" in text
        words = np.loadtxt(out, dtype=np.int64, ndmin=2)
        b = RefCoveringBackend(2, 4, 1)
        assert _is_cover(b, b.word_index(words))

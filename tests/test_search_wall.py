"""The wall-breaking passes on the reference backends — no GPU required.

A *wall* is the state that binds record production: a cover of size M+1
is in hand, one of size M is believed to exist, and the search sits at
1-3 uncovered words indefinitely. These are the four additions aimed at
it:

  * ``uncovered_words`` — the exact deficient words, which is what turns
    "something is uncovered" into a target;
  * ``ruin(k, "deficit")`` and ``lns(..., deficit=True)`` — ruin aimed at
    the incumbents nearest those words;
  * ``anneal(..., pool_refresh="auto")`` — measured pool cadence;
  * ``anneal(..., neighbourhood="shell"|"auto")`` — the local candidate
    batch, which is the difference between a polish that accepts and one
    that is inert;
  * ``two_swap`` — exact bounded (add, drop) search for the endgame;
  * ``break_wall`` — the driver that rotates them.

As in ``test_search_passes.py``, every "solved" claim is re-checked by
:func:`_is_cover`, which unions raw balls and never consults the
backend's incremental ``cnt``.
"""
import time

import numpy as np
import pytest

from coldcore.reference import (RefCoveringBackend, RefLeeBackend,
                                RefTorusBackend)
from coldcore.search import (RUIN_MIX_EXT, RUIN_MODES, RUIN_MODES_EXT,
                             OperatorBandit, Searcher)

# The Z_5^2 Lee state from test_search_passes.py: 3 uncovered, and no
# single swap improves it (that test brute-forces the proof).
LEE_WALL = np.array([0, 1, 8, 17, 19], dtype=np.int64)


def _is_cover(b, code):
    seen = np.zeros(b.space.size, dtype=bool)
    for w in np.atleast_1d(np.asarray(code, dtype=np.int64)):
        seen[b.ball_indices(int(w))] = True
    return bool(seen.all())


def _random_start(b, m, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(b.space.size, size=m, replace=False))


def _searcher(b, seed=1, secs=60, log=None, on_solved=None):
    return Searcher(b, seed=seed, deadline=time.time() + secs,
                    on_log=(log.append if log is not None else None),
                    on_solved=on_solved)


# ------------------------------------------------------- deficit words
def test_uncovered_words_are_exactly_the_uncovered_words():
    b = RefLeeBackend([5, 5], 1)
    s = _searcher(b)
    s.load(LEE_WALL)
    assert s.uncov == 3
    words = s.uncovered_words(limit=8)
    truth = np.flatnonzero(b.cnt == 0)
    assert sorted(words.tolist()) == sorted(truth.tolist())
    assert s.uncovered_words(limit=2).tolist() == sorted(words.tolist())[:2]


def test_uncovered_words_empty_when_feasible_and_capped_when_asked():
    b = RefCoveringBackend(2, 4, 1)
    s = _searcher(b)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 6)
    assert s.uncov == 0
    assert not len(s.uncovered_words())
    # a scan cap that reads nothing must report nothing, never lie
    s.load(_random_start(b, 3, 1))
    assert s.uncov > 0
    assert not len(s.uncovered_words(scan_cap=0))


def test_uncovered_words_are_cached_against_the_solution():
    """The wall case: round after round rejected back to the same
    incumbent must not re-scan the field."""
    b = RefCoveringBackend(3, 4, 1)
    s = _searcher(b)
    s.load(_random_start(b, 9, 2))
    reads = {"n": 0}
    raw = b.read_cnt

    def counted(*a, **k):
        reads["n"] += 1
        return raw(*a, **k)
    b.read_cnt = counted
    first = s.uncovered_words()
    assert reads["n"] > 0
    n0 = reads["n"]
    for _ in range(5):
        assert s.uncovered_words().tolist() == first.tolist()
    assert reads["n"] == n0                     # served from the cache
    s.ruin(2, "random")                         # different solution
    s.uncovered_words()
    assert reads["n"] > n0


def test_deficit_ruin_removes_points_near_the_deficit():
    """The point of the operator: the removed incumbents must be closer
    to the uncovered words than a random selection of the same size."""
    b = RefCoveringBackend(3, 5, 1)
    s = _searcher(b, seed=5)
    s.load(_random_start(b, 30, 9))
    assert s.uncov > 0
    words = s.uncovered_words(limit=4)
    assert len(words)
    code = s.code.copy()
    digs = b.index_word(code).astype(np.int16)
    tw = b.index_word(words).astype(np.int16)
    dist = np.min(np.stack([(digs != t[None, :]).sum(axis=1) for t in tw]),
                  axis=0)
    removed = s.ruin(6, "deficit")
    assert len(removed) == 6 and len(s.code) == len(code) - 6
    picked = dist[np.isin(code, removed)]
    assert picked.mean() < dist.mean()
    assert s.stats["deficit_ruins"] == 1
    assert s.uncov == s.verify_full()


def test_deficit_ruin_falls_back_without_the_capability():
    """``read_cnt`` is an optional backend capability. A backend without
    it must degrade to `hole` — not crash, and not silently do nothing.
    Same start, same seed, so the fallback is provably the `hole` draw."""
    outs = []
    for capable in (False, True):
        b = RefCoveringBackend(3, 4, 1)
        s = _searcher(b, seed=1)
        s.load(_random_start(b, 12, 3))
        if not capable:
            s.supports_deficit = lambda: False
        removed = s.ruin(3, "deficit" if not capable else "hole")
        assert len(removed) == 3 and s.uncov == s.verify_full()
        outs.append(sorted(removed.tolist()))
    assert outs[0] == outs[1]                   # the fallback IS `hole`


# --------------------------------------------------- deficit inside LNS
def test_lns_deficit_is_opt_in_and_default_is_unchanged():
    out = []
    for kw in ({}, {"deficit": False}):
        b = RefCoveringBackend(3, 4, 1)
        s = _searcher(b, seed=6)
        s.load(_random_start(b, 12, 4))
        unc = s.lns(12, round_budget_s=5, kmin=2, kmax=4, **kw)
        out.append((unc, s.code.tolist(), s.stats["rounds"]))
    assert out[0] == out[1]
    assert "deficit_ruins" not in out and s.stats.get("deficit_ruins", 0) == 0


@pytest.mark.parametrize("select", ["fixed", "adaptive"])
def test_lns_deficit_solves_and_actually_fires(select):
    b = RefCoveringBackend(3, 4, 1)
    s = _searcher(b, seed=3)
    s.load(_random_start(b, 9, 11))
    unc = s.lns(9, round_budget_s=15, kmin=1, kmax=3, select=select,
                seed=5, deficit=True)
    assert unc == 0 and _is_cover(b, s.code)
    assert s.stats.get("deficit_ruins", 0) > 0
    if select == "adaptive":
        assert set(s.stats["ops"]) == set(RUIN_MODES_EXT)


def test_bandit_extended_ops_and_boost_hint():
    ban = OperatorBandit(np.random.default_rng(0), ops=RUIN_MODES_EXT,
                         prior=RUIN_MIX_EXT, floor=0.0)
    p = ban.probabilities()
    assert list(ban.ops) == list(RUIN_MODES_EXT)
    assert p[ban.ops.index("deficit")] == pytest.approx(2.0 / 7.0)
    boosted = ban.probabilities({"deficit": 4.0})
    assert boosted[ban.ops.index("deficit")] > p[ban.ops.index("deficit")]
    assert boosted.sum() == pytest.approx(1.0)
    # a crushed operator must still be reachable when the driver hints
    for _ in range(50):
        ban.update("deficit", OperatorBandit.REWARD_WORSE)
    assert ban.probabilities({"deficit": 4.0})[ban.ops.index("deficit")] > \
        ban.probabilities()[ban.ops.index("deficit")]
    # no boost => bit-identical to the plain rule (the default path)
    assert np.array_equal(ban.probabilities(None), ban.probabilities())
    plain = OperatorBandit(np.random.default_rng(0))
    assert list(plain.ops) == list(RUIN_MODES)


# ------------------------------------------------- adaptive pool cadence
def test_auto_refresh_formula_tracks_the_cost_ratio():
    f = Searcher._auto_refresh
    # a transform costing 9x a move, kept to 10% of the clock -> P = 81
    assert f(9e-3, 1e-3, 0.10) == 81
    # cheap transform, cheap moves -> the floor
    assert f(1e-6, 1e-3, 0.10) == 1
    # a very expensive transform climbs, but is capped
    assert f(10.0, 1e-6, 0.10) == 1 << 16
    # a bigger share allowance means refreshing more often
    assert f(9e-3, 1e-3, 0.5) < f(9e-3, 1e-3, 0.05)


def test_auto_pool_refresh_solves_and_reports_its_cadence():
    b = RefTorusBackend([9, 9], 1)
    s = _searcher(b, seed=8)
    s.load(_random_start(b, 9, 31))
    unc = s.anneal(9, budget_s=15, mode="lahc", pool_refresh="auto", seed=8)
    assert unc == 0 and _is_cover(b, s.code)
    assert s.stats["pool_refresh"] >= 1


def test_pool_refresh_default_is_the_fixed_reproducible_one():
    """Auto reads the clock, so it cannot be the default: the documented
    seed-reproducibility of `lahc` depends on the fixed cadence."""
    b = RefCoveringBackend(3, 4, 1)
    runs = []
    for _ in range(2):
        s = _searcher(b, seed=99)
        s.load(_random_start(b, 9, 21))
        runs.append((s.anneal(9, budget_s=10, mode="lahc", seed=4),
                     s.code.tolist()))
    assert runs[0] == runs[1]
    with pytest.raises(ValueError):
        _searcher(b).anneal(9, budget_s=1, pool_refresh=0)


# ------------------------------------------------------------ two-swap
def test_two_swap_breaks_the_proven_single_swap_wall():
    """The state no single swap improves. Depth 1 (one swap) must fail on
    it — that is what makes it a wall — and depth 2 must break it."""
    b = RefLeeBackend([5, 5], 1)
    s = _searcher(b, seed=1)
    s.load(LEE_WALL)
    assert s.two_swap(5, budget_s=10, max_uncov=3, depth=1, seed=3) == 3
    assert s.code.tolist() == LEE_WALL.tolist()      # state restored
    assert s.verify_full() == 3

    s.load(LEE_WALL)
    assert s.two_swap(5, budget_s=10, max_uncov=3, depth=2, seed=3) == 0
    assert len(s.code) == 5 and _is_cover(b, s.code)
    assert s.verify_full() == 0
    assert s.stats["two_swap_hits"] == 1


def test_two_swap_is_reproducible_from_its_seed():
    b = RefLeeBackend([5, 5], 1)
    runs = []
    for _ in range(2):
        s = _searcher(b, seed=1)
        s.load(LEE_WALL)
        runs.append((s.two_swap(5, budget_s=10, max_uncov=3, seed=7),
                     s.code.tolist(), s.stats["two_swap_nodes"]))
    assert runs[0] == runs[1]
    assert runs[0][2] > 0


def test_two_swap_leaves_the_state_exact_when_it_fails():
    """No budget, an impossible M, and a deficit outside its range: in
    every case the state it hands back must be the state it got, exactly
    — same code, same cnt, same reported count."""
    b = RefCoveringBackend(2, 6, 1)                  # K_2(6,1) = 12
    for kw in ({"budget_s": 0.0}, {"nodes": 0}, {"max_uncov": 1},
               {"branch": 0}):
        s = _searcher(b, seed=2)
        s.load(_random_start(b, 8, 5))
        before, u0 = s.code.copy(), s.uncov
        base = {"budget_s": 5.0, "max_uncov": 99, "seed": 3}
        base.update(kw)
        unc = s.two_swap(8, **base)
        assert unc == u0 == s.verify_full()
        assert s.code.tolist() == before.tolist()
        assert not _is_cover(b, s.code)


def test_two_swap_preserves_M_and_rejects_a_mismatched_M():
    b = RefLeeBackend([5, 5], 1)
    s = _searcher(b, seed=1)
    s.load(LEE_WALL)
    with pytest.raises(ValueError):
        s.two_swap(6, budget_s=1, max_uncov=3)
    with pytest.raises(ValueError):
        s.two_swap(5, budget_s=1, max_uncov=3, depth=0)
    assert s.two_swap(None, budget_s=5, max_uncov=3, seed=1) in (0, 1, 2, 3)
    assert len(s.code) == 5


def test_two_swap_is_a_no_op_outside_its_range():
    b = RefCoveringBackend(2, 4, 1)
    s = _searcher(b, seed=1)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 6)
    assert s.uncov == 0
    assert s.two_swap(len(s.code), budget_s=5) == 0
    assert s.stats.get("two_swap_nodes", 0) == 0


# --------------------------------------------------------- wall breaker
@pytest.mark.parametrize("name,make,opt", [
    ("hamming K_2(4,1)", lambda: RefCoveringBackend(2, 4, 1), 4),
    ("torus 9x9 king", lambda: RefTorusBackend([9, 9], 1), 9),
    ("lee Z_5^2 R=1", lambda: RefLeeBackend([5, 5], 1), 5),
], ids=["hamming", "torus", "lee"])
def test_break_wall_reaches_the_known_optimum(name, make, opt):
    b = make()
    for start in (1, 2):
        s = _searcher(b, seed=start, secs=90)
        s.load(_random_start(b, opt + 2, start))
        unc = s.break_wall(opt, budget_s=20, burst_s=2, seed=start)
        assert unc == 0, f"{name}: wall-breaker stalled at {unc}"
        assert len(s.code) == opt and _is_cover(b, s.code)
        assert s.verify_full() == 0
        assert s.stats["wall"]["M"] == opt and s.stats["wall"]["best"] == 0


def test_break_wall_is_honest_about_an_impossible_target():
    b = RefCoveringBackend(2, 6, 1)                  # K_2(6,1) = 12
    log = []
    s = _searcher(b, seed=3, secs=90, log=log)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 6)
    s.peel()
    unc = s.break_wall(11, budget_s=8, burst_s=1.5, stagnation=1, seed=1)
    assert unc > 0 and unc == s.verify_full()
    assert len(s.code) == 11 and not _is_cover(b, s.code)
    assert s.stats["restarts"] > 0               # stagnation forced restarts
    w = s.stats["wall"]
    assert w["bursts"] > 1 and w["best"] == unc
    assert [u for _, u in w["traj"]] == sorted(  # best never gets worse
        [u for _, u in w["traj"]], reverse=True) or w["best"] == min(
        u for _, u in w["traj"])
    assert any(line.startswith("wall M=11") for line in log)


def test_break_wall_drops_an_oversized_incumbent_to_the_target():
    b = RefCoveringBackend(3, 4, 1)
    s = _searcher(b, seed=2, secs=90)
    s.load(_random_start(b, 30, 2))
    assert s.uncov == 0 and len(s.code) == 30
    unc = s.break_wall(9, budget_s=20, burst_s=2, seed=3)
    assert len(s.code) == 9
    assert unc == s.verify_full() and (unc == 0) == _is_cover(b, s.code)


def test_break_wall_rejects_unknown_polish():
    b = RefCoveringBackend(2, 4, 1)
    s = _searcher(b, seed=1)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 6)
    with pytest.raises(ValueError):
        s.break_wall(4, budget_s=1, polish="tabu")


def test_break_wall_works_without_polish_or_two_swap():
    """Each stage of the rotation must be individually switchable off —
    that is what the benchmark ablations rely on."""
    b = RefTorusBackend([9, 9], 1)
    s = _searcher(b, seed=4, secs=60)
    s.load(_random_start(b, 11, 4))
    unc = s.break_wall(9, budget_s=15, burst_s=2, polish=None,
                       two_swap_at=0, deficit=False, seed=2)
    assert unc == s.verify_full() and len(s.code) == 9
    assert s.stats.get("two_swap_nodes", 0) == 0
    assert s.stats.get("deficit_ruins", 0) == 0


# ------------------------------------------------------- descend + cli
def test_descend_deficit_stays_valid_and_monotone():
    b = RefCoveringBackend(3, 4, 1)
    solved = []
    s = _searcher(b, seed=15, secs=90,
                  on_solved=lambda m, idx: solved.append((m, idx.copy())))
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 6)
    m = s.descend(floor=9, notch_budget_s=3, seed=2, deficit=True,
                  select="adaptive", portfolio=2)
    ms = [x for x, _ in solved]
    assert ms == sorted(ms, reverse=True)
    for size, code in solved:
        assert len(code) == size and _is_cover(b, code)
    assert m == ms[-1] <= 11


def test_cli_wall_flag(tmp_path, capsys):
    from coldcore.cli import main
    out = tmp_path / "wall.txt"
    main(["solve", "--backend", "ref", "-p", "q=2,n=4,R=1", "--wall", "4",
          "--wall-budget", "10", "--wall-burst", "1.5", "--hours", "0.05",
          "--seed", "3", "--deficit", "--polish-neighbourhood", "shell",
          "--out", str(out)])
    text = capsys.readouterr().out
    assert "wall-breaking at M=4" in text
    assert "wall M=4 ended at uncov=0" in text
    words = np.loadtxt(out, dtype=np.int64, ndmin=2)
    b = RefCoveringBackend(2, 4, 1)
    assert _is_cover(b, b.word_index(words))


def test_cli_descend_deficit_flag(tmp_path, capsys):
    from coldcore.cli import main
    out = tmp_path / "sol.txt"
    main(["solve", "--backend", "ref", "-p", "q=2,n=4,R=1", "--floor", "4",
          "--notch-budget", "1", "--hours", "0.02", "--seed", "3",
          "--deficit", "--ruin-select", "adaptive",
          "--polish", "lahc", "--polish-neighbourhood", "auto",
          "--out", str(out)])
    text = capsys.readouterr().out
    assert "SOLVED M=" in text and "descent ended at M=" in text
    words = np.loadtxt(out, dtype=np.int64, ndmin=2)
    b = RefCoveringBackend(2, 4, 1)
    assert _is_cover(b, b.word_index(words))


# --------------------------------------------- local (shell) neighbourhood
def test_shell1_is_the_exact_distance_1_neighbourhood():
    """Mixed radix included: the shell must be every word differing in
    exactly one digit, and nothing else."""
    b = RefCoveringBackend(3, 4, 1)
    s = _searcher(b)
    for w in (0, 17, 80):
        got = np.sort(s._shell1(w))
        digs = b.space.all_words().astype(np.int64)
        d = (digs != digs[w][None, :]).sum(axis=1)
        assert got.tolist() == np.flatnonzero(d == 1).tolist()
        assert len(got) == 4 * (3 - 1)


def test_shell1_handles_mixed_radix_axes():
    from coldcore.reference import RefGridBackend
    b = RefGridBackend([2, 3, 4], 1)
    s = _searcher(b)
    digs = b.space.all_words().astype(np.int64)
    for w in (0, 5, 23):
        got = np.sort(s._shell1(w))
        d = (digs != digs[w][None, :]).sum(axis=1)
        assert got.tolist() == np.flatnonzero(d == 1).tolist()


def test_shell_k_samples_within_distance_k():
    b = RefCoveringBackend(3, 5, 1)
    s = _searcher(b)
    rng = np.random.default_rng(0)
    digs = b.space.all_words().astype(np.int64)
    got = s._shell_k(100, 2, 64, rng)
    assert len(got) == 64
    d = (digs[got] != digs[100][None, :]).sum(axis=1)
    assert d.max() <= 2 and (d > 0).any()


def test_shell_polish_beats_the_pool_on_a_plateau():
    """The whole point. On this cell the pool neighbourhood's accepts
    collapse and it stalls; the local one keeps moving and solves."""
    b = RefTorusBackend([9, 9], 1)
    out = {}
    for hood in ("pool", "shell"):
        s = _searcher(b, seed=2, secs=90)
        s.load(_random_start(b, 9, 3))
        unc = s.anneal(9, budget_s=10, mode="lahc", neighbourhood=hood, seed=4)
        assert unc == s.verify_full() and len(s.code) == 9
        out[hood] = (unc, s.stats["accepts"] / max(1, s.stats["moves"]))
    assert out["shell"][0] < out["pool"][0], out
    assert out["shell"][1] > 10 * out["pool"][1], out
    assert out["shell"][0] == 0


@pytest.mark.parametrize("name,make,opt", [
    ("hamming K_2(4,1)", lambda: RefCoveringBackend(2, 4, 1), 4),
    ("torus 9x9 king", lambda: RefTorusBackend([9, 9], 1), 9),
    ("lee Z_5^2 R=1", lambda: RefLeeBackend([5, 5], 1), 5),
], ids=["hamming", "torus", "lee"])
@pytest.mark.parametrize("mode", ["anneal", "lahc"])
def test_shell_polish_reaches_the_known_optimum(name, make, opt, mode):
    b = make()
    for start in (1, 2, 3):
        s = _searcher(b, seed=start)
        s.load(_random_start(b, opt, start))
        assert s.uncov > 0
        unc = s.anneal(opt, budget_s=15, mode=mode, neighbourhood="shell",
                       seed=start)
        assert unc == 0, f"{name}/{mode}: shell polish stalled at {unc}"
        assert len(s.code) == opt and _is_cover(b, s.code)
        assert s.verify_full() == 0


def test_shell_polish_escapes_the_pinned_swap_local_optimum():
    b = RefLeeBackend([5, 5], 1)
    s = _searcher(b, seed=1)
    s.load(LEE_WALL)
    assert s.uncov == 3
    assert s.anneal(5, budget_s=15, mode="lahc", neighbourhood="shell",
                    seed=2) == 0
    assert _is_cover(b, s.code) and len(s.code) == 5


def test_shell_polish_never_reports_a_bad_cover():
    b = RefCoveringBackend(3, 4, 1)
    for start in range(5):
        s = _searcher(b, seed=start)
        s.load(_random_start(b, 9, 100 + start))
        unc = s.anneal(9, budget_s=0.05, mode="lahc", neighbourhood="shell",
                       seed=start, loss_refresh=1, pool_refresh=1)
        assert unc == s.verify_full()
        assert (unc == 0) == _is_cover(b, s.code)
        assert len(s.code) == 9


def test_shell_polish_reheats_from_its_best_and_shrinks_to_M():
    b = RefCoveringBackend(2, 6, 1)              # K_2(6,1) = 12
    s = _searcher(b, seed=3, secs=90)
    s.load(_random_start(b, 20, 2))
    log = []
    s.on_log = log.append
    unc = s.anneal(11, budget_s=6, mode="lahc", neighbourhood="shell",
                   restart_stale=50, seed=1)
    assert len(s.code) == 11 and unc == s.verify_full() and unc > 0
    assert any("reheat" in line for line in log)
    assert s.stats["shift_moves"] > 0 and s.stats["shift_accepts"] > 0


def test_auto_neighbourhood_switches_when_accepts_collapse():
    """gamma(C9 x C9 king) = 9, so M = 8 is unreachable: the pool
    neighbourhood's accepts collapse there and `auto` must notice."""
    b = RefTorusBackend([9, 9], 1)
    s = _searcher(b, seed=2, secs=90)
    s.load(_random_start(b, 8, 3))
    log = []
    s.on_log = log.append
    unc = s.anneal(8, budget_s=10, mode="lahc", neighbourhood="auto", seed=4)
    assert unc == s.verify_full() > 0 and len(s.code) == 8
    assert s.stats["polish_neighbourhood"] == "shell"
    assert any("switching to the shell" in line for line in log)
    assert s.stats["shift_moves"] > 0


def test_auto_neighbourhood_stays_on_the_pool_when_it_works():
    """K_3(4,1) from a random 9-set: the pool neighbourhood solves it in
    a handful of accepts, so `auto` must not pay for a switch."""
    b = RefCoveringBackend(3, 4, 1)
    s = _searcher(b, seed=2, secs=60)
    s.load(_random_start(b, 9, 3))
    unc = s.anneal(9, budget_s=8, mode="lahc", neighbourhood="auto", seed=4)
    assert unc == 0 and _is_cover(b, s.code)
    assert s.stats["polish_neighbourhood"] == "pool"


def test_unknown_neighbourhood_rejected():
    b = RefCoveringBackend(2, 4, 1)
    s = _searcher(b)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 6)
    with pytest.raises(ValueError):
        s.anneal(4, budget_s=1, neighbourhood="local")
    with pytest.raises(ValueError):
        s.break_wall(4, budget_s=1, polish_neighbourhood="local")
    with pytest.raises(ValueError):
        s.descend(floor=1, notch_budget_s=1, polish_neighbourhood="local")


def test_pool_neighbourhood_is_the_default_and_unchanged():
    """The frozen path: no neighbourhood argument must be bit-identical
    to the explicit pool one, and must not touch the shell counters."""
    out = []
    for kw in ({}, {"neighbourhood": "pool"}):
        b = RefCoveringBackend(3, 4, 1)
        s = _searcher(b, seed=99)
        s.load(_random_start(b, 9, 21))
        unc = s.anneal(9, budget_s=6, mode="lahc", seed=4, **kw)
        out.append((unc, s.code.tolist(), s.stats["moves"],
                    s.stats["accepts"]))
        assert "shift_moves" not in s.stats
    assert out[0] == out[1]


def test_break_wall_plumbs_the_neighbourhood_and_defaults_to_auto():
    """At an unreachable M the driver gets past the LNS burst and into
    the polish — which must be the neighbourhood it was asked for."""
    import inspect
    sig = inspect.signature(Searcher.break_wall)
    assert sig.parameters["polish_neighbourhood"].default == "auto"

    b = RefTorusBackend([9, 9], 1)              # gamma = 9, so 8 is out
    s = _searcher(b, seed=4, secs=90)
    s.load(_random_start(b, 11, 4))
    unc = s.break_wall(8, budget_s=12, burst_s=4, seed=2,
                       polish_neighbourhood="shell")
    assert unc > 0 and unc == s.verify_full() and len(s.code) == 8
    assert s.stats["shift_moves"] > 0 and s.stats["shift_accepts"] > 0
    assert s.stats["polish_neighbourhood"] == "shell"
    assert not _is_cover(b, s.code)

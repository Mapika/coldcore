"""The coset start (coldcore.cosets) on small cells — no GPU required.

Every structural claim is checked against brute force over the whole
space: the syndrome map partitions the space into |Q| cosets of |H|
words, D is the syndrome set of the radius-R ball, the expansion of a
quotient cover is a cover of the space, and the quotient backend's
fields agree with the explicit-incidence backend on the same incidence.
"""
import time

import numpy as np
import pytest

from coldcore.cosets import (CosetModel, QuotientBackend, choose_m,
                             coset_stage, h_sizes, prime_powers)
from coldcore.incidence import SparseIncidenceBackend
from coldcore.reference import RefCoveringBackend
from coldcore.search import Searcher
from coldcore.space import ProductSpace

# (q, n, R, syndrome digits): prime, prime power, and two mixed alphabets
CELLS = [(3, 4, 1, {3: 2}), (4, 4, 1, {2: 3}), (8, 3, 1, {2: 4}),
         (6, 3, 1, {2: 2, 3: 1}), (10, 3, 1, {2: 1, 5: 2}),
         (5, 4, 2, {5: 2})]


def _is_cover(q, n, R, words):
    sp = ProductSpace(q, n)
    W = sp.all_words().astype(np.int64)
    C = np.asarray(words, dtype=np.int64)
    d = (W[:, None, :] != C[None, :, :]).sum(axis=2).min(axis=1)
    return bool((d <= R).all())


def test_prime_powers():
    assert prime_powers(2) == [(2, 1)]
    assert prime_powers(8) == [(2, 3)]
    assert prime_powers(10) == [(2, 1), (5, 1)]
    assert prime_powers(12) == [(2, 2), (3, 1)]
    assert prime_powers(9) == [(3, 2)]


@pytest.mark.parametrize("q,n,R,m", CELLS)
def test_syndromes_partition_the_space(q, n, R, m):
    rng = np.random.default_rng(1)
    M = CosetModel(q, n, R, m, rng)
    W = ProductSpace(q, n).all_words().astype(np.int64)
    syn = M.index(M.syndrome_digits(W))
    assert M.size * M.h_size == q ** n
    counts = np.bincount(syn, minlength=M.size)
    assert (counts == M.h_size).all()
    # D is exactly the syndrome set of the words of weight <= R
    D = np.zeros(M.size, dtype=bool)
    D[syn[(W != 0).sum(axis=1) <= R]] = True
    assert (D == M.D).all()


@pytest.mark.parametrize("q,n,R,m", CELLS)
def test_coset_words_are_the_cosets(q, n, R, m):
    rng = np.random.default_rng(2)
    M = CosetModel(q, n, R, m, rng)
    picks = rng.choice(M.size, size=min(3, M.size), replace=False)
    cw = M.coset_words(picks)
    sp = ProductSpace(q, n)
    assert len(cw) == len(picks) * M.h_size
    assert len(np.unique(sp.word_index(cw))) == len(cw)
    assert set(M.index(M.syndrome_digits(cw)).tolist()) == set(
        picks.tolist())


@pytest.mark.parametrize("q,n,R,m", CELLS)
def test_quotient_greedy_expands_to_a_cover(q, n, R, m):
    rng = np.random.default_rng(3)
    M = CosetModel(q, n, R, m, rng)
    picks, trace = M.greedy(rng)
    assert trace[-1] == 0
    assert _is_cover(q, n, R, M.coset_words(picks))


@pytest.mark.parametrize("q,n,R,m", CELLS)
def test_quotient_backend_matches_incidence_backend(q, n, R, m):
    rng = np.random.default_rng(4)
    M = CosetModel(q, n, R, m, rng)
    qb = QuotientBackend(M)
    ib = SparseIncidenceBackend(M.incidence(), M.size)
    code = rng.choice(M.size, size=max(1, M.size // 5), replace=False)
    for b in (qb, ib):
        b.recount(code)
    assert (qb.cnt == ib.cnt).all()
    for f in ("gain_map", "loss_map"):
        getattr(qb, f)()
        getattr(ib, f)()
        assert (qb.map == ib.map).all(), f
    cand = rng.choice(M.size, size=17)
    for t in (0, 1, 2):
        assert (qb.ball_gather(cand, t) == ib.ball_gather(cand, t)).all()
    for d in (+1, -1):
        qb.ball_update(cand[:4], d)
        ib.ball_update(cand[:4], d)
        assert (qb.cnt == ib.cnt).all()
    assert qb.uncovered() == ib.uncovered()


def test_search_core_runs_on_the_quotient():
    """Greedy + LNS of the ordinary searcher on Q give a cover of Q, and
    its expansion is a cover of the space."""
    rng = np.random.default_rng(5)
    M = CosetModel(3, 5, 1, {3: 3}, rng)
    qb = QuotientBackend(M)
    s = Searcher(qb, seed=1, deadline=time.time() + 30)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    assert s.uncov == 0
    s.peel()
    assert s.verify_full() == 0
    assert _is_cover(3, 5, 1, M.coset_words(s.code))


def test_choose_m_and_h_sizes():
    for m in choose_m(6, 4, 1, q_max=1 << 10):
        N = int(np.prod([p ** v for p, v in m.items()]))
        assert 1 < N <= 1 << 10
    hs = h_sizes(3, 8, 600, lo=2, hi=400)
    assert [h for h, _ in hs] == [3, 9, 27, 81, 243]
    assert all(2 <= 600 / h <= 400 for h, _ in hs)


def test_coset_stage_returns_a_verified_cover():
    b = RefCoveringBackend(3, 5, 1)
    s = Searcher(b, seed=2, deadline=time.time() + 60)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    s.peel()
    m_ref = len(s.code)
    got = coset_stage(s, 3, 5, 1, budget_s=4.0, m_ref=m_ref)
    assert got is not None
    assert np.array_equal(np.sort(got), np.sort(s.code))
    assert s.verify_full() == 0
    assert _is_cover(3, 5, 1, b.index_word(got))


def test_cli_coset_start(tmp_path, capsys):
    from coldcore.cli import main
    out = tmp_path / "sol.txt"
    main(["solve", "--backend", "cpu", "-p", "q=3,n=5,R=1",
          "--hours", "0.004", "--seed", "2", "--start", "cosets",
          "--coset-share", "0.3", "--out", str(out)])
    text = capsys.readouterr().out
    assert "coset start:" in text and "descent ended at M=" in text
    words = np.loadtxt(out, dtype=np.int64, ndmin=2)
    assert _is_cover(3, 5, 1, words)


def test_cli_start_auto_needs_a_deadline(tmp_path, capsys):
    """Without --hours the default start is the plain greedy (the coset
    stage is time-boxed by the run budget)."""
    from coldcore.cli import main
    out = tmp_path / "sol.txt"
    main(["solve", "--backend", "ref", "-p", "q=2,n=4,R=1", "--floor", "4",
          "--notch-budget", "1", "--seed", "3", "--out", str(out)])
    text = capsys.readouterr().out
    assert "coset start" not in text
    words = np.loadtxt(out, dtype=np.int64, ndmin=2)
    assert _is_cover(2, 4, 1, words)

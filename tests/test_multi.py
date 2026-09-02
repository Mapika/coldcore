"""mu-fold multiple covering K^mu(n; ax; R) — CPU ground truth.

Every word must be covered at least mu times. The generalization lives
entirely at the coverage-predicate seam (docs/multi-coverage.md), so it
must hold for EVERY problem family; these tests pin the three exact
quantities the (untouched) search core runs on

    deficit D = sum_x max(0, mu - cnt(x))     (0 <=> feasible)
    gain(x)   = #{w in B(x) : cnt(w) < mu}    = -delta D of adding x
    loss(c)   = #{w in B(c) : 1 <= cnt(w) <= mu} = +delta D of removing c

against brute force, plus exhaustively-determined small optima.
"""
import time
from itertools import combinations

import numpy as np
import pytest

from coldcore.reference import (RefHammingBackend, RefLeeBackend,
                                RefTorusBackend)
from coldcore.search import Searcher

# (family factory, axes, R) exercised for every mu
FAMILIES = [
    ("hamming", RefHammingBackend, [3, 4, 5], 1),
    ("hamming", RefHammingBackend, [2, 2, 2, 2], 1),
    ("lee", RefLeeBackend, [5, 5], 1),
    ("lee", RefLeeBackend, [4, 6], 2),
    ("torus", RefTorusBackend, [6, 6], 1),
]


def _code(b, m, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(b.space.size, size=m, replace=False))


def _deficit(b):
    """Independent recomputation of the objective from cnt alone."""
    return int(sum(max(0, b.mu - int(c)) for c in b.cnt))


@pytest.mark.parametrize("name,cls,axes,R", FAMILIES)
def test_mu1_is_the_legacy_problem(name, cls, axes, R):
    """mu=1 must reproduce the ordinary covering fields exactly."""
    a, b = cls(axes, R), cls(axes, R, mu=1)
    code = _code(a, 7, 1)
    every = np.arange(a.space.size, dtype=np.int64)
    for x in (a, b):
        x.recount(code)
    assert (a.cnt == b.cnt).all()
    a.gain_map(); b.gain_map()
    assert (a.map == b.map).all()
    assert (a.map_read_at(every) == a.ball_gather(every, 0)).all()
    a.loss_map(); b.loss_map()
    assert (a.map == b.map).all()
    assert a.uncovered() == b.uncovered() == a.count_eq(0)
    # the roles are literal cnt levels at mu=1
    assert (a.ball_gather(every, 0) == b.ball_gather(every, 0)).all()
    assert (a.ball_gather(every, 1) == b.ball_gather(every, 1)).all()


@pytest.mark.parametrize("name,cls,axes,R", FAMILIES)
@pytest.mark.parametrize("mu", [2, 3])
def test_field_identities(name, cls, axes, R, mu):
    """gain/loss maps are the ball sums of the deficient/critical
    indicators, and uncovered() is the total deficit."""
    b = cls(axes, R, mu=mu)
    code = _code(b, 6, 2)
    b.recount(code)
    every = np.arange(b.space.size, dtype=np.int64)
    assert b.uncovered() == _deficit(b)
    b.gain_map()
    assert (b.map_read_at(every) == b.ball_gather(every, 0)).all()
    b.loss_map()
    assert (b.map_read_at(every) == b.ball_gather(every, 1)).all()
    # roles, spelled out from cnt directly
    for w in every[::7]:
        ball = b.ball_indices(w)
        assert b.ball_gather([w], 0)[0] == int((b.cnt[ball] < mu).sum())
        assert b.ball_gather([w], 1)[0] == int(
            ((b.cnt[ball] >= 1) & (b.cnt[ball] <= mu)).sum())


@pytest.mark.parametrize("name,cls,axes,R", FAMILIES)
@pytest.mark.parametrize("mu", [2, 3])
def test_gain_and_loss_are_exact_marginals(name, cls, axes, R, mu):
    """The greedy is exact only if gain/loss are the true deltas of the
    objective — check that against a from-scratch recount."""
    b = cls(axes, R, mu=mu)
    code = _code(b, 5, 3)
    b.recount(code)
    d0 = b.uncovered()
    rng = np.random.default_rng(4)
    for x in rng.choice(b.space.size, size=6, replace=False):
        x = int(x)
        if x in code:
            continue
        g = int(b.ball_gather([x], 0)[0])
        b.recount(np.append(code, x))
        assert b.uncovered() == d0 - g
        b.recount(code)
    for c in code:
        lo = int(b.ball_gather([int(c)], 1)[0])
        b.recount(code[code != c])
        assert b.uncovered() == d0 + lo
        b.recount(code)


@pytest.mark.parametrize("name,cls,axes,R", FAMILIES)
def test_peel_safety(name, cls, axes, R):
    """A codeword is removable iff no word of its ball sits at cnt <= mu
    — exactly the loss == 0 test the peel uses."""
    mu = 2
    b = cls(axes, R, mu=mu)
    s = Searcher(b, seed=9, deadline=time.time() + 120)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    for c in s.code:
        safe = bool((b.cnt[b.ball_indices(int(c))] > mu).all())
        assert (int(b.ball_gather([int(c)], 1)[0]) == 0) == safe
    n = s.peel()
    assert s.verify_full() == 0                  # peel preserved mu-coverage
    assert n >= 0 and (b.cnt >= mu).all()


# ---------------------------------------------------------------- optima
def _exhaustive_mu_optimum(b, mu):
    """Smallest mu-fold cover by exhaustive subset search (tiny spaces)."""
    size = b.space.size
    balls = [b.ball_indices(w) for w in range(size)]
    for k in range(1, size + 1):
        for c in combinations(range(size), k):
            cnt = np.zeros(size, dtype=np.int64)
            for w in c:
                cnt[balls[w]] += 1
            if (cnt >= mu).all():
                return k, c
    return None, None


@pytest.mark.parametrize("mu,expect", [(1, 2), (2, 4), (3, 6)])
def test_binary_n3_R1_exhaustive_optimum(mu, expect):
    """K_2^mu(3,1): |X| = 8, |B_1| = 4, so the sphere bound is 2*mu and
    it is attained by unions of the 4 disjoint perfect codes (cosets of
    the repetition code). Determined here by exhaustive search over all
    subsets, then re-derived by the search core."""
    b = RefHammingBackend([2] * 3, 1, mu=mu)
    assert len(b.ball_indices(0)) == 4
    k, c = _exhaustive_mu_optimum(b, mu)
    assert k == expect == 2 * mu               # == sphere bound mu*|X|/|B|
    s = Searcher(b, seed=3, deadline=time.time() + 60)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    s.peel()
    if len(s.code) > expect:
        s.ruin(len(s.code) - expect)
    assert s.lns(expect, round_budget_s=30) == 0
    assert len(s.code) <= expect and s.verify_full() == 0


def test_mu2_lee_z5_sphere_bound():
    """K^2(Z_5^2, Lee, R=1) = 10: |B_1| = 5 gives the bound 2*25/5 = 10,
    and the two cosets {(i,2i)}, {(i,2i+1)} of the perfect Golomb-Welch
    code are disjoint perfect covers whose union attains it."""
    b = RefLeeBackend([5, 5], 1, mu=2)
    twin = b.word_index(np.array([[i, (2 * i + t) % 5]
                                  for t in (0, 1) for i in range(5)]))
    b.recount(twin)
    assert b.uncovered() == 0 and (b.cnt == 2).all()   # perfect 2-fold cover
    s = Searcher(b, seed=5, deadline=time.time() + 300)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    s.peel()
    if len(s.code) > 10:
        s.ruin(len(s.code) - 10)
    assert s.lns(10, round_budget_s=90) == 0
    assert len(s.code) == 10 and s.verify_full() == 0


def test_mu2_torus_king_6x6():
    """K^2 of the 6x6 king torus = 8: |B_1| = 9 divides 36, the four
    3x3 boxes centred at {1,4}^2 tile the torus, and the (1,0)-shift of
    that tiling is a second perfect cover — 8 kings, every square
    dominated exactly twice; the sphere bound 2*36/9 = 8 makes it
    optimal."""
    b = RefTorusBackend([6, 6], 1, mu=2)
    twin = b.word_index(np.array([[(i + s) % 6, j] for s in (0, 1)
                                  for i in (1, 4) for j in (1, 4)]))
    b.recount(twin)
    assert b.uncovered() == 0 and (b.cnt == 2).all()
    s = Searcher(b, seed=5, deadline=time.time() + 180)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    s.peel()
    if len(s.code) > 8:
        s.ruin(len(s.code) - 8)
    assert s.lns(8, round_budget_s=60) == 0
    assert len(s.code) == 8 and s.verify_full() == 0

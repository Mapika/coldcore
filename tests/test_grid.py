"""Open-boundary grid domination (`grid_linf`, problem id 3) — CPU.

Chebyshev balls on a bounded board: no wraparound, so the per-axis
window clamps at the edges and the coverage relation is NOT shift
invariant. That is what makes this family the generality test for the
plugin seams — everything else in the framework is circulant.

Ground truth here is the independent brute-force RefGridBackend; the
optima are pinned by matching packing lower bounds and explicit
constructions, both checked in-test.
"""
import time
from itertools import product

import numpy as np
import pytest

from coldcore.patterns import support_patterns
from coldcore.reference import RefGridBackend
from coldcore.search import Searcher


def _expected_ball(axes, R, digits):
    """prod_j |[a_j-R, a_j+R] cap [0, m_j)| — the clamped box."""
    out = 1
    for a, m in zip(digits, axes):
        a = int(a)  # digits may arrive as np.uint8; max(0, a-R) must not wrap
        out *= min(m - 1, a + R) - max(0, a - R) + 1
    return out


@pytest.mark.parametrize("axes,R", [([8, 8], 1), ([5, 9], 1), ([4, 4], 2),
                                    ([6, 3, 3], 1), ([3, 3], 3)])
def test_balls_are_clamped_boxes(axes, R):
    b = RefGridBackend(axes, R)
    for w in range(b.space.size):
        d = b.space.index_word([w])[0]
        assert len(b.ball_indices(w)) == _expected_ball(axes, R, d)


def test_shift_invariance_is_broken():
    """The whole point of the family: corner balls are smaller than
    interior ones (on a torus every ball has the same size)."""
    b = RefGridBackend([8, 8], 1)
    corner = len(b.ball_indices(int(b.word_index(np.array([[0, 0]]))[0])))
    inner = len(b.ball_indices(int(b.word_index(np.array([[4, 4]]))[0])))
    edge = len(b.ball_indices(int(b.word_index(np.array([[0, 4]]))[0])))
    assert (corner, edge, inner) == (4, 6, 9)


@pytest.mark.parametrize("axes,R", [([8, 8], 1), ([5, 9], 1), ([4, 4], 2),
                                    ([6, 3, 3], 1), ([3, 3], 3)])
def test_support_table_walk_matches_ball(axes, R):
    """The packed table holds circular shifts (as for the torus); the
    WALK resolves each against the centre digit and drops patterns that
    would wrap past a boundary. Replicated here in numpy, for every
    centre — corners and edges included — against the brute-force ball.
    """
    b = RefGridBackend(axes, R)
    pats = support_patterns(axes, R, "grid_linf")
    assert len(set(int(p) for p in pats)) == len(pats)      # no duplicates
    for w in range(b.space.size):
        dig = b.space.index_word([w])[0].astype(np.int64)
        seen = set()
        for p in pats:
            word, v, good = dig.copy(), int(p), True
            while v:
                byte = v & 0xFF
                pos, delta = byte >> 4, byte & 0xF
                nd = word[pos] + delta
                off = delta if nd < axes[pos] else delta - axes[pos]
                if abs(off) > R:                 # wrapped out of the ball
                    good = False
                    break
                word[pos] += off
                v >>= 8
            if good:
                seen.add(int(b.space.word_index(word[None, :])[0]))
        assert seen == set(b.ball_indices(w).tolist())


def test_field_identities():
    b = RefGridBackend([5, 9], 1)
    rng = np.random.default_rng(1)
    code = np.sort(rng.choice(b.space.size, size=6, replace=False))
    b.recount(code)
    every = np.arange(b.space.size, dtype=np.int64)
    b.gain_map()
    assert (b.map_read_at(every) == b.ball_gather(every, 0)).all()
    b.loss_map()
    assert (b.map_read_at(code) == b.ball_gather(code, 1)).all()


def test_incremental_matches_recount():
    b = RefGridBackend([8, 8], 1)
    rng = np.random.default_rng(2)
    code = np.sort(rng.choice(b.space.size, size=8, replace=False))
    b.recount(code[:4])
    b.ball_update(code[4:], +1)
    inc = b.cnt.copy()
    b.recount(code)
    assert (inc == b.cnt).all()


# ------------------------------------------------------------- optima
def _packing_bound(b, cells):
    """|cells| is a lower bound on any dominating set if the closed
    neighbourhoods of `cells` are pairwise disjoint (each needs its own
    dominator). Verified here, not assumed."""
    balls = [set(b.ball_indices(int(w)).tolist())
             for w in b.word_index(np.array(cells))]
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            assert not (balls[i] & balls[j])
    return len(balls)


def _is_cover(b, cells, mu=1):
    b.recount(b.word_index(np.array(cells)))
    return b.uncovered() == 0 and int(b.cnt.min()) >= mu


@pytest.mark.parametrize("axes,rows,cols,pack_r,pack_c,opt", [
    # 8x8 king graph: gamma = ceil(8/3)^2 = 9
    ([8, 8], (1, 4, 6), (1, 4, 6), (0, 3, 6), (0, 3, 6), 9),
    # non-square board 5x9: gamma = ceil(5/3)*ceil(9/3) = 2*3 = 6
    ([5, 9], (1, 4), (1, 4, 7), (0, 3), (0, 3, 6), 6),
])
def test_king_domination_number(axes, rows, cols, pack_r, pack_c, opt):
    """Both bounds are established in-test: a construction that is
    verified to dominate, and a packing of that many cells whose closed
    neighbourhoods are verified pairwise disjoint. The engine must then
    reach exactly `opt`."""
    b = RefGridBackend(axes, 1)
    assert _is_cover(b, list(product(rows, cols)))          # upper bound
    assert _packing_bound(b, list(product(pack_r, pack_c))) == opt
    assert len(rows) * len(cols) == opt
    s = Searcher(b, seed=2, deadline=time.time() + 300)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    s.peel()
    if len(s.code) > opt:
        s.ruin(len(s.code) - opt)
    assert s.lns(opt, round_budget_s=90) == 0
    assert len(s.code) == opt and s.verify_full() == 0
    assert (b.cnt >= 1).all()


def test_mu_fold_grid_domination():
    """mu composes with the new family. On the 5x5 board with R=1 the
    2-fold domination number is exactly 8:

    upper bound — {0,3}x{0,3} and {1,4}x{1,4} are two DISJOINT
    dominating sets (each row/column window pair covers 0..4), so their
    8-cell union dominates every square twice;
    lower bound — the four cells {0,3}^2 have pairwise disjoint closed
    neighbourhoods and each needs mu = 2 dominators inside its own
    neighbourhood, giving 4 * 2 = 8.
    """
    b = RefGridBackend([5, 5], 1, mu=2)
    one = RefGridBackend([5, 5], 1)
    cells = [(r, c) for r in (0, 3) for c in (0, 3)]
    other = [(r, c) for r in (1, 4) for c in (1, 4)]
    assert _is_cover(one, cells) and _is_cover(one, other)
    assert not set(cells) & set(other)
    assert _is_cover(b, cells + other, mu=2)
    assert _packing_bound(one, cells) * 2 == 8          # lower bound
    s = Searcher(b, seed=4, deadline=time.time() + 300)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    s.peel()
    if len(s.code) > 8:
        s.ruin(len(s.code) - 8)
    assert s.lns(8, round_budget_s=90) == 0
    assert len(s.code) == 8 and s.verify_full() == 0
    assert (b.cnt >= 2).all()

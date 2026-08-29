"""Lee-metric reference backend: ball sizes, identities, small optima.

The third problem family. Everything here is CPU-only ground truth: the
brute-force RefLeeBackend (independent double-loop over words) checked
against the closed-form weight-enumerator ball sizes and against known /
brute-force-verified optima (Golomb-Welch perfect codes).
"""
import time
from itertools import combinations

import numpy as np
import pytest

from coldcore.patterns import lee_ball_size, support_patterns
from coldcore.reference import RefLeeBackend
from coldcore.search import Searcher


def _rand_code(b, m, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(b.space.size, size=m, replace=False))


# (axes, R) -> |B_R| computed by hand from the per-axis weight
# enumerators L_a(z) = 1 + 2z + ... (+ z^{a/2} once for even a):
#   [5,5]  R=1: (1+2z+2z^2)^2      -> 1 + 4        = 5   (Golomb-Welch)
#   [4,4]  R=1: (1+2z+z^2)^2       -> 1 + 4        = 5
#   [4,4]  R=2:                    -> 1 + 4 + 6    = 11  (antipodal w=2
#                                     has multiplicity 1, not 2)
#   [6,6]  R=2: (1+2z+2z^2+z^3)^2  -> 1 + 4 + 8    = 13
#   [7,7]  R=2: (1+2z+2z^2+2z^3)^2 -> 1 + 4 + 8    = 13  (= 2R(R+1)+1)
#   [3,4,5] R=1: c1 = 2+2+2        -> 1 + 6        = 7
#   [4,4]  R=4: whole space        ->                16
BALL_CASES = [
    ([5, 5], 1, 5),
    ([4, 4], 1, 5),
    ([4, 4], 2, 11),
    ([6, 6], 2, 13),
    ([7, 7], 2, 13),
    ([3, 4, 5], 1, 7),
    ([4, 4], 4, 16),
]


@pytest.mark.parametrize("axes,R,expect", BALL_CASES)
def test_ball_sizes(axes, R, expect):
    """Enumerator formula == brute-force ball == the hand value; every
    ball the same size (the metric is translation invariant)."""
    assert lee_ball_size(axes, R) == expect
    b = RefLeeBackend(axes, R)
    sizes = {len(b.ball_indices(w)) for w in range(b.space.size)}
    assert sizes == {expect}


def test_ball_is_brute_force_lee_ball():
    """ball_indices really is the Lee ball (independent recomputation)."""
    b = RefLeeBackend([4, 6, 5], 2)
    digs = b.space.all_words().astype(np.int64)
    ax = np.array([4, 6, 5])
    for w in (0, 17, 63, b.space.size - 1):
        d = np.abs(digs - digs[w][None, :])
        lee = np.minimum(d, ax - d).sum(axis=1)
        assert (np.flatnonzero(lee <= b.R) == b.ball_indices(w)).all()


@pytest.mark.parametrize("axes,R", [([5, 5], 1), ([4, 6], 2), ([3, 4, 5], 1)])
def test_support_table_matches_ball(axes, R):
    """The packed v1 support table enumerates exactly the Lee ball."""
    b = RefLeeBackend(axes, R)
    pats = support_patterns(axes, R, "lee")
    axarr = np.array(axes, dtype=np.int64)
    center = np.array([a // 2 for a in axes], dtype=np.int64)
    seen = set()
    for p in pats:
        w = center.copy()
        v = int(p)
        while v:
            byte = v & 0xFF
            pos, delta = byte >> 4, byte & 0xF
            w[pos] = (w[pos] + delta) % axarr[pos]
            v >>= 8
        seen.add(int(b.space.word_index(w[None, :])[0]))
    assert len(seen) == len(pats)           # no duplicate patterns
    cidx = int(b.space.word_index(center[None, :])[0])
    assert seen == set(b.ball_indices(cidx).tolist())


@pytest.fixture(params=["lee55", "lee45even"])
def b(request):
    if request.param == "lee55":
        return RefLeeBackend([5, 5], 1)
    return RefLeeBackend([4, 5], 2)         # even axis: antipodal edge case


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


def test_perfect_code_z5():
    """{(i, 2i mod 5)} is the perfect Lee code in Z_5^2 (Golomb-Welch):
    every word covered exactly once."""
    b = RefLeeBackend([5, 5], 1)
    code = b.word_index(np.array([[i, (2 * i) % 5] for i in range(5)]))
    b.recount(code)
    assert b.count_eq(1) == b.space.size    # perfect: multiplicity 1
    assert b.uncovered() == 0


def test_lee_search_finds_perfect_5():
    """Search must reproduce the optimum 5 in Z_5^2, R=1: the sphere
    bound gives >= 25/|B_1| = 5, and a 5-cover exists (perfect code)."""
    b = RefLeeBackend([5, 5], 1)
    assert b.space.size == 5 * lee_ball_size([5, 5], 1)   # bound == 5
    s = Searcher(b, seed=7, deadline=time.time() + 90)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    s.peel()
    assert s.verify_full() == 0
    unc = s.lns(5, round_budget_s=45)
    assert unc == 0 and len(s.code) == 5
    assert s.verify_full() == 0


def test_lee_search_matches_bruteforce_optimum_even_q():
    """Even-q cell [4,4], R=1: exhaustively determine the optimal cover
    size (16 words, |B_1|=5), then search must reach it."""
    b = RefLeeBackend([4, 4], 1)
    size = b.space.size
    masks = [0] * size
    for w in range(size):
        for u in b.ball_indices(w):
            masks[w] |= 1 << int(u)
    full = (1 << size) - 1
    opt = None
    for k in range(1, size + 1):
        if any_cover_of_size(masks, k, full):
            opt = k
            break
    assert opt is not None and opt >= -(-size // 5)       # sphere bound
    s = Searcher(b, seed=11, deadline=time.time() + 90)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    s.peel()
    unc = s.lns(opt, round_budget_s=45)
    assert unc == 0 and len(s.code) <= opt
    assert s.verify_full() == 0


def any_cover_of_size(masks, k, full):
    return any(_or_all(c) == full for c in combinations(masks, k))


def _or_all(c):
    m = 0
    for x in c:
        m |= x
    return m

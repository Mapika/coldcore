"""Internal consistency of the numpy reference backend (no GPU needed).

These identities are exactly what the GPU parity tests check against, so
the reference must satisfy them by construction.
"""
import numpy as np
import pytest

from coldcore.patterns import ball_size
from coldcore.reference import RefCoveringBackend


@pytest.fixture(scope="module")
def b():
    return RefCoveringBackend(3, 4, 1)


def _rand_code(b, m, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(b.space.size, size=m, replace=False))


def test_recount_matches_bruteforce(b):
    code = _rand_code(b, 7, 0)
    b.recount(code)
    # brute force: cnt[u] = #codewords within distance R
    digs = b.space.all_words().astype(np.int16)
    cd = digs[code]
    d = (digs[:, None, :] != cd[None, :, :]).sum(axis=2)
    assert (b.cnt == (d <= b.R).sum(axis=1)).all()


def test_gain_equals_ball_gather_zero(b):
    code = _rand_code(b, 6, 1)
    b.recount(code)
    b.gain_map()
    every = np.arange(b.space.size, dtype=np.int64)
    assert (b.map_read_at(every) == b.ball_gather(every, 0)).all()


def test_loss_equals_ball_gather_one(b):
    code = _rand_code(b, 9, 2)
    b.recount(code)
    b.loss_map()
    assert (b.map_read_at(code) == b.ball_gather(code, 1)).all()


def test_ball_update_matches_recount(b):
    code = _rand_code(b, 8, 3)
    b.recount(code[:5])
    b.ball_update(code[5:], +1)
    cnt_inc = b.cnt.copy()
    b.recount(code)
    assert (cnt_inc == b.cnt).all()
    b.ball_update(code[5:], -1)
    cnt_dec = b.cnt.copy()
    b.recount(code[:5])
    assert (cnt_dec == b.cnt).all()


def test_count_eq_and_empty_code(b):
    b.recount(np.zeros(0, dtype=np.int64))
    assert b.uncovered() == b.space.size
    b.gain_map()
    # with nothing covered, gain of every word is its ball size
    assert b.map_max() == ball_size(b.q, b.n, b.R)


def test_map_extract_and_hist(b):
    code = _rand_code(b, 5, 4)
    b.recount(code)
    b.gain_map()
    mx = b.map_max()
    idx, val, found = b.map_extract(mx)      # top slice
    assert found == len(idx) and (val == mx).all()
    hist = b.map_hist(16, mx)
    assert hist.sum() == int((b.map > 0).sum())
    # top bin counts exactly the values in ((15*mx/16), mx]
    v = b.map[b.map > 0]
    assert hist[15] == int(((v - 1) * 16 // mx == 15).sum() +
                           ((v - 1) * 16 // mx > 15).sum())

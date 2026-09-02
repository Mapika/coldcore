import numpy as np

from coldcore.patterns import ball_patterns, ball_patterns_ox, ball_size
from coldcore.space import ProductSpace


def test_index_roundtrip():
    sp = ProductSpace(5, 6)
    idx = np.array([0, 1, 5, 5 ** 6 - 1, 12345], dtype=np.int64)
    words = sp.index_word(idx)
    assert (sp.word_index(words) == idx).all()


def test_all_words_unique():
    sp = ProductSpace(3, 4)
    w = sp.all_words()
    assert w.shape == (81, 4)
    assert len(np.unique(sp.word_index(w))) == 81


def _decode_v1(pat, n):
    """Decode a packed v1 pattern into {pos: delta}."""
    out = {}
    p = int(pat)
    while p:
        b = p & 0xFF
        out[b >> 4] = b & 0xF
        p >>= 8
    return out


def test_ball_patterns_enumerate_ball():
    q, n, R = 3, 4, 2
    pats = ball_patterns(q, n, R)
    assert len(pats) == ball_size(q, n, R)
    # applying every pattern to a center hits distinct words, all within R
    sp = ProductSpace(q, n)
    center = np.array([1, 0, 2, 1], dtype=np.int64)
    seen = set()
    for p in pats:
        d = _decode_v1(p, n)
        w = center.copy()
        for pos, delta in d.items():
            w[pos] = (w[pos] + delta) % q
        assert len(d) <= R
        seen.add(int(sp.word_index(w[None, :])[0]))
    assert len(seen) == ball_size(q, n, R)


def test_ball_patterns_ox_matches_v1():
    q, n, R = 4, 5, 2
    pats = ball_patterns(q, n, R)
    masks, offs = ball_patterns_ox(q, n, R)
    assert len(masks) == len(pats) == ball_size(q, n, R)
    for k in range(len(pats)):
        d = _decode_v1(pats[k], n)
        mask = 0
        for pos, delta in d.items():
            mask |= 1 << pos
            assert (int(offs[k]) >> (4 * pos)) & 0xF == delta
        assert mask == int(masks[k])

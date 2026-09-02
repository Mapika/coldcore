"""Parity gate for the fast CPU backend.

``NumpyDPBackend`` replaces an O(|X|^2) brute force with the plugin's
O(n|X|) axis DP. The only acceptable evidence for that is *identical
numbers*: every field, every query and every incremental operation is
compared against the corresponding ``Ref*Backend`` on randomised
configurations, mu-fold cases included — and then a whole ``Searcher``
run is compared move for move.
"""
import numpy as np
import pytest

from coldcore.dp import NumpyDPBackend
from coldcore.reference import (RefGridBackend, RefHammingBackend,
                                RefLeeBackend, RefTorusBackend)
from coldcore.search import Searcher

#: (problem, ref class, axes, R) — every shipped family, homogeneous and
#: mixed radix, plus the awkward corners (R=0, oversized grid window,
#: axis of size 2, a full-radius Lee ball).
CELLS = [
    ("hamming", RefHammingBackend, [2] * 8, 1),
    ("hamming", RefHammingBackend, [3] * 5, 2),
    ("hamming", RefHammingBackend, [4] * 4, 1),
    ("hamming", RefHammingBackend, [2, 3, 4, 5], 2),
    ("hamming", RefHammingBackend, [5, 2, 3], 0),
    ("hamming", RefHammingBackend, [6, 6], 2),       # R = n: the whole space
    ("lee", RefLeeBackend, [5, 5], 1),
    ("lee", RefLeeBackend, [7, 7], 2),
    ("lee", RefLeeBackend, [4, 5, 6], 3),
    ("lee", RefLeeBackend, [8, 8], 4),
    ("torus_linf", RefTorusBackend, [9, 9], 1),
    ("torus_linf", RefTorusBackend, [7, 7, 7], 1),
    ("torus_linf", RefTorusBackend, [11, 5], 2),
    ("grid_linf", RefGridBackend, [8, 8], 1),
    ("grid_linf", RefGridBackend, [5, 6, 4], 2),
    ("grid_linf", RefGridBackend, [7, 7], 5),        # window wider than axis
]

IDS = [f"{p}-{'x'.join(map(str, a))}-R{r}" for p, _, a, r in CELLS]


def _pair(problem, cls, axes, R, mu=1):
    return cls(axes, R, mu=mu), NumpyDPBackend(axes, R, problem=problem, mu=mu)


def _assert_fields_match(ref, dp):
    assert np.array_equal(np.asarray(ref.cnt), np.asarray(dp.cnt))
    assert ref.uncovered() == dp.uncovered()
    for t in range(0, 4):
        assert ref.count_eq(t) == dp.count_eq(t)
    for which in ("gain_map", "loss_map"):
        getattr(ref, which)()
        getattr(dp, which)()
        assert np.array_equal(np.asarray(ref.map), np.asarray(dp.map)), which
        assert ref.map_max() == dp.map_max()
        mx = ref.map_max()
        if mx > 0:
            assert np.array_equal(ref.map_hist(16, mx), dp.map_hist(16, mx))
            for thr in (1, max(1, mx // 2), mx):
                ri, rv, rf = ref.map_extract(thr)
                di, dv, df = dp.map_extract(thr)
                assert rf == df
                assert np.array_equal(ri, di) and np.array_equal(rv, dv)


@pytest.mark.parametrize("problem,cls,axes,R", CELLS, ids=IDS)
@pytest.mark.parametrize("mu", [1, 2])
def test_fields_and_queries_match_brute_force(problem, cls, axes, R, mu):
    rng = np.random.default_rng(hash((problem, tuple(axes), R, mu)) % 2 ** 32)
    ref, dp = _pair(problem, cls, axes, R, mu=mu)
    size = ref.space.size
    for trial in range(3):
        m = int(rng.integers(1, max(2, size // 4)))
        code = np.unique(rng.integers(0, size, size=m).astype(np.int64))
        ref.recount(code)
        dp.recount(code)
        _assert_fields_match(ref, dp)
        # incremental: same adds/removes, same field afterwards
        add = rng.integers(0, size, size=3).astype(np.int64)
        ref.ball_update(add, +1)
        dp.ball_update(add, +1)
        assert np.array_equal(np.asarray(ref.cnt), np.asarray(dp.cnt))
        probe = rng.integers(0, size, size=min(size, 32)).astype(np.int64)
        for target in (0, 1, 2):
            assert np.array_equal(ref.ball_gather(probe, target),
                                  dp.ball_gather(probe, target))
        ref.ball_update(add, -1)
        dp.ball_update(add, -1)
        assert np.array_equal(np.asarray(ref.cnt), np.asarray(dp.cnt))
        assert np.array_equal(ref.read_cnt(), dp.read_cnt())
        off = int(rng.integers(0, size))
        assert np.array_equal(ref.read_cnt(off, size - off),
                              dp.read_cnt(off, size - off))


@pytest.mark.parametrize("problem,cls,axes,R", CELLS, ids=IDS)
def test_ball_indices_match_brute_force(problem, cls, axes, R):
    ref, dp = _pair(problem, cls, axes, R)
    rng = np.random.default_rng(7)
    for w in rng.integers(0, ref.space.size, size=12):
        assert np.array_equal(ref.ball_indices(int(w)),
                              dp.ball_indices(int(w)))


@pytest.mark.parametrize("problem,cls,axes,R", CELLS, ids=IDS)
def test_indep_dist_and_capabilities_match(problem, cls, axes, R):
    ref, dp = _pair(problem, cls, axes, R)
    assert ref.indep_dist == dp.indep_dist
    assert dp.supports_hist
    assert list(dp.space.axes) == list(ref.space.axes)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_dtype_does_not_change_the_answer(dtype):
    ref = RefHammingBackend([3] * 4, 1)
    dp = NumpyDPBackend([3] * 4, 1, dtype=dtype)
    code = np.arange(0, 81, 7, dtype=np.int64)
    ref.recount(code)
    dp.recount(code)
    assert np.array_equal(np.asarray(ref.cnt), np.asarray(dp.cnt))


@pytest.mark.parametrize("problem,cls,axes,R,mu", [
    ("hamming", RefHammingBackend, [3] * 4, 1, 1),
    ("hamming", RefHammingBackend, [2] * 6, 1, 2),
    ("lee", RefLeeBackend, [5, 5], 1, 1),
    ("torus_linf", RefTorusBackend, [9, 9], 1, 1),
    ("grid_linf", RefGridBackend, [8, 8], 1, 1),
], ids=["K3(4,1)", "K2^2(6,1)", "lee5x5", "king9x9", "grid8x8"])
def test_a_whole_search_run_is_identical_on_both_backends(problem, cls, axes,
                                                          R, mu):
    """The point of the parity: `cpu` is a drop-in for `ref`.

    Same seed, same passes -> the *same* solution, not merely one of
    equal size. That is only possible if every field value, every
    ball_gather and even map_extract's ordering agree.

    The round count is fixed rather than the wall-clock budget: the
    time-budgeted passes deliberately run *more* rounds on the faster
    backend, so a time-bounded run could only ever be compared by
    quality, not by trace."""
    out = []
    for mk in (lambda: cls(axes, R, mu=mu),
               lambda: NumpyDPBackend(axes, R, problem=problem, mu=mu)):
        b = mk()
        s = Searcher(b, seed=3)
        s.load(np.zeros(0, dtype=np.int64))
        s.greedy_fill(10 ** 6)
        s.peel()
        trace = [(int(s.uncov), s.code.tolist())]
        target = max(1, len(s.code) - 1)
        # a fixed ruin/recreate round sequence: every operator, exact
        # repair, best-snapshot bookkeeping — all clock-free
        for i in range(12):
            s.ruin(1 + i % 3, ["low", "hole", "cluster", "random"][i % 4])
            s.greedy_fill(target)
            trace.append((int(s.uncov), s.code.tolist()))
        out.append(trace)
    assert out[0] == out[1]


@pytest.mark.parametrize("problem,cls,axes,R", CELLS, ids=IDS)
def test_support_set_equals_the_packed_plugin_table(problem, cls, axes, R):
    """The column table must enumerate exactly the plugin's packed one.

    Decoding the v1 bytes independently and walking both tables from the
    same centre is the provenance check: the CPU backend covers the same
    ball the CUDA plugin does, pattern for pattern."""
    from coldcore.patterns import support_patterns
    if max(axes) > 16:
        pytest.skip("packed format caps axes at 16")
    pats = support_patterns(axes, R, problem)
    dp = NumpyDPBackend(axes, R, problem=problem)
    sp = dp.space
    w = int(sp.size // 3)
    dig = sp.index_word(np.array([w]))[0].astype(np.int64)
    packed = set()
    for v in pats.tolist():
        x, ok = w, True
        for k in range(8):
            b = (v >> (8 * k)) & 255
            if not b:
                break
            p, d = b >> 4, b & 15
            nd = (int(dig[p]) + d) % axes[p]
            sh = nd - int(dig[p])
            if problem == "grid_linf" and abs(sh) > R:
                ok = False
                break
            x += sh * int(sp.pow_q[p])
        if ok:
            packed.add(x)
    assert packed == set(dp.ball_indices(w).tolist())


@pytest.mark.parametrize("problem,cls,axes,R", CELLS, ids=IDS)
@pytest.mark.parametrize("target", [0, 1, 2])
def test_both_gather_paths_agree(problem, cls, axes, R, target,
                                 monkeypatch):
    """The mask-vs-values choice in ``ball_gather`` is a cost policy.

    Forcing each path over the same states must give identical counts —
    including on ``grid_linf``, whose balls are not translation
    invariant, on mu > 1, and on states where the role set is empty."""
    import coldcore.dp as dpmod
    ref, dp = _pair(problem, cls, axes, R, mu=2)
    rng = np.random.default_rng(11)
    size = ref.space.size
    for m in (0, 1, size // 8 or 1, size // 2, size):
        code = np.unique(rng.integers(0, size, size=max(m, 1)))[:m]
        ref.recount(code)
        dp.recount(code)
        for k in (1, 7, 64):
            probe = rng.integers(0, size, size=min(size, k)).astype(np.int64)
            want = ref.ball_gather(probe, target)
            # break-even 0 forces the mask path, a huge one forces values
            monkeypatch.setattr(dpmod, "_MASK_BREAKEVEN", 0)
            assert np.array_equal(dp.ball_gather(probe, target), want)
            monkeypatch.setattr(dpmod, "_MASK_BREAKEVEN", 10 ** -9)
            assert np.array_equal(dp.ball_gather(probe, target), want)


def test_rejects_unknown_problem_and_bad_mu():
    with pytest.raises(ValueError):
        NumpyDPBackend([3] * 3, 1, problem="nope")
    with pytest.raises(ValueError):
        NumpyDPBackend([3] * 3, 1, mu=0)
    with pytest.raises(ValueError):        # torus window must not self-wrap
        NumpyDPBackend([4, 4], 2, problem="torus_linf")


def test_extract_shuffle_is_seeded_like_the_reference():
    dp = NumpyDPBackend([3] * 3, 1, seed_shuffle=5)
    dp.recount(np.array([0], dtype=np.int64))
    dp.gain_map()
    a, _, fa = dp.map_extract(1)
    dp.gain_map()
    b, _, fb = dp.map_extract(1)
    assert fa == fb and sorted(a.tolist()) == sorted(b.tolist())

"""GPU fast paths (v0.3) against the CPU axis-DP backend.

* set-side gather (``DCT_SET_GATHER``: 0 = support walk only, 2 = set side
  whenever the predicate set fits): every problem family, mu = 1 and
  mu > 1, from near-cover states (small deficient set) and from random
  states (the set overflows and the walk takes over);
* asynchronous single-word ``ball_update`` immediately followed by reads;
* the support-table locality order: the binding's reordered table must be
  a permutation of the builder's.

Function-scoped init/free (v1 singleton ABI). The CPU backend is the
exact reference (it is itself pinned to the brute-force reference by
tests/test_dp_backend.py).
"""
import ctypes
from contextlib import contextmanager

import numpy as np
import pytest

from coldcore.dp import NumpyDPBackend
from coldcore.patterns import support_patterns

pytestmark = pytest.mark.gpu

CELLS = [
    ("hamming", [3] * 6, 1, 1),
    ("hamming", [4] * 5, 2, 1),
    ("hamming", [3, 4, 5, 6], 2, 1),
    ("hamming", [3] * 5, 1, 2),
    ("lee", [5, 6, 7], 2, 1),
    ("lee", [4, 4, 4, 4], 3, 1),
    ("torus_linf", [7, 9, 11], 2, 1),
    ("grid_linf", [4, 5, 6], 3, 1),
    ("grid_linf", [8, 8], 1, 2),
]


@contextmanager
def gpu_cell(monkeypatch, problem, axes, R, mu, policy):
    from coldcore.gpu import GpuBackend
    monkeypatch.setenv("DCT_SET_GATHER", str(policy))
    g = GpuBackend(axes, R, problem=problem, mu=mu)
    try:
        yield g
    finally:
        g.close()


def _stats(g):
    if not hasattr(g._lib, "dct_stats"):
        return None
    buf = (ctypes.c_longlong * 8)()
    g._lib.dct_stats(buf, 8)
    return list(buf)


def _near_cover(c, rng):
    """Random greedy-ish cover of the CPU cell, minus a few words."""
    code = []
    c.recount(np.zeros(0, dtype=np.int64))
    while c.uncovered() > 0:
        unc = np.flatnonzero(c.cnt < c.mu)
        code.append(int(rng.choice(unc)))
        c.ball_update(np.array(code[-1:], dtype=np.int64), +1)
    code = np.array(code, dtype=np.int64)
    return code[rng.permutation(len(code))[:max(1, len(code) - 3)]]


@pytest.mark.parametrize("policy", [0, 2])
@pytest.mark.parametrize("problem,axes,R,mu", CELLS)
def test_gather_paths_match_cpu(monkeypatch, gpu_backend_factory, problem,
                                axes, R, mu, policy):
    c = NumpyDPBackend(axes, R, problem=problem, mu=mu)
    rng = np.random.default_rng(31)
    size = c.space.size
    starts = [_near_cover(c, rng),
              np.sort(rng.choice(size, size=max(2, size // 40),
                                 replace=False)).astype(np.int64)]
    with gpu_cell(monkeypatch, problem, axes, R, mu, policy) as g:
        for code in starts:
            g.recount(code)
            c.recount(code)
            for step in range(4):
                probe = rng.integers(0, size, size=int(rng.integers(1, 300)))
                probe = probe.astype(np.int64)
                for role in (0, 1):
                    assert (g.ball_gather(probe, role)
                            == c.ball_gather(probe, role)).all()
                # mutate between gathers (cache invalidation)
                w = rng.integers(0, size, size=1).astype(np.int64)
                g.ball_update(w, +1)
                c.ball_update(w, +1)
                assert (g.ball_gather(probe, 0)
                        == c.ball_gather(probe, 0)).all()
                g.ball_update(code[step:step + 1], -1)
                c.ball_update(code[step:step + 1], -1)
                assert g.uncovered() == c.uncovered()
                assert (g.ball_gather(probe, 0)
                        == c.ball_gather(probe, 0)).all()
                g.ball_update(code[step:step + 1], +1)
                c.ball_update(code[step:step + 1], +1)
                g.ball_update(w, -1)
                c.ball_update(w, -1)
            assert (g.read_cnt() == c.read_cnt()).all()
        st = _stats(g)
    if st is not None:
        if policy == 0:
            assert st[0] == 0               # set side never used
        else:
            assert st[0] > 0                # near-cover start used it


def test_tiled_set_gather(monkeypatch, gpu_backend_factory):
    """A deficient set above 32,768 words takes the tiled kernel (and
    grows the list buffer); counts equal the CPU backend's."""
    problem, axes, R = "hamming", [5] * 8, 3
    c = NumpyDPBackend(axes, R)
    rng = np.random.default_rng(8)
    size = c.space.size
    code = np.sort(rng.choice(size, 120, replace=False)).astype(np.int64)
    monkeypatch.setenv("DCT_SET_RATIO", "32")      # allow |Z| up to 32 |B|
    with gpu_cell(monkeypatch, problem, axes, R, 1, 2) as g:
        g.recount(code)
        c.recount(code)
        assert 32768 < c.uncovered() < 32 * 4065   # tiled, within ratio
        probe = rng.integers(0, size, size=200).astype(np.int64)
        assert (g.ball_gather(probe, 0) == c.ball_gather(probe, 0)).all()
        w = probe[:3]
        g.ball_update(w, +1)
        c.ball_update(w, +1)
        assert (g.ball_gather(probe, 0) == c.ball_gather(probe, 0)).all()
        st = _stats(g)
    if st is not None:
        assert st[7] > 0


def test_async_updates_then_reads(gpu_backend_factory):
    q, n, R = 3, 6, 1
    c = NumpyDPBackend([q] * n, R)
    rng = np.random.default_rng(5)
    g = gpu_backend_factory(q, n, R)
    try:
        code = np.sort(rng.choice(c.space.size, 30, replace=False))
        g.recount(code)
        c.recount(code)
        live = list(code)
        for i in range(400):            # many back-to-back async launches
            if i % 3 == 2 and live:
                w = np.array([live.pop(int(rng.integers(len(live))))])
                g.ball_update(w, -1)
                c.ball_update(w, -1)
            else:
                w = rng.integers(0, c.space.size, size=1 + i % 70)
                live.extend(int(x) for x in w)
                g.ball_update(w.astype(np.int64), +1)
                c.ball_update(w.astype(np.int64), +1)
            if i % 50 == 0:
                assert g.count_eq(0) == c.count_eq(0)
        assert (g.read_cnt() == c.read_cnt()).all()
        g.gain_map()
        c.gain_map()
        every = np.arange(c.space.size, dtype=np.int64)
        assert (g.map_read_at(every) == c.map_read_at(every)).all()
    finally:
        g.close()


@pytest.mark.parametrize("problem,axes,R", [
    ("hamming", [5] * 6, 3), ("hamming", [3, 4, 5, 6], 2),
    ("lee", [7, 7, 7], 3), ("torus_linf", [5, 7, 9], 1),
    ("grid_linf", [4, 5, 6], 3)])
def test_locality_order_is_permutation(problem, axes, R):
    from coldcore.gpu import locality_order
    pats = support_patterns(axes, R, problem)
    out = locality_order(pats, axes)
    assert out[0] == 0                      # the centre stays first
    assert np.array_equal(np.sort(out), np.sort(pats))


TRANSFORM_CELLS = [
    ([2] * 9, 3, 1), ([2] * 10, 2, 1), ([3] * 7, 2, 1), ([3] * 2, 1, 1),
    ([4] * 5, 4, 1), ([5] * 5, 3, 1), ([4, 4, 5, 5, 3], 2, 1),
    ([6] * 4, 2, 1), ([7] * 4, 3, 1), ([8] * 4, 4, 1), ([9] * 4, 3, 1),
    ([10] * 4, 4, 1), ([3] * 4, 4, 1), ([7] * 3, 3, 1), ([2, 3, 4, 5, 6, 7], 3, 1),
    ([10, 9, 8, 7, 6], 5, 1), ([3] * 6, 2, 2), ([4] * 4, 1, 3),
]


@pytest.mark.parametrize("variant", ["pairs", "single", "legacy"])
@pytest.mark.parametrize("axes,R,mu", TRANSFORM_CELLS)
def test_hamming_transform_matches_cpu(monkeypatch, gpu_backend_factory,
                                       axes, R, mu, variant):
    """v0.3 transform (exact-Q templates, no fills, fused init, two axes
    per kernel where adjacent axes are equal), its one-axis-per-kernel
    form (DCT_PAIR_PASSES=0) and the legacy path (DCT_LEGACY_TRANSFORM=1)
    all reproduce the CPU fields: cnt from a multiset code, the gain map
    and the loss map, including R >= n cells (every layer nonzero), odd
    and even n, and mixed radix."""
    from coldcore.gpu import GpuBackend
    monkeypatch.setenv("DCT_LEGACY_TRANSFORM",
                       "1" if variant == "legacy" else "0")
    monkeypatch.setenv("DCT_PAIR_PASSES", "0" if variant == "single" else "1")
    c = NumpyDPBackend(axes, R, mu=mu)
    rng = np.random.default_rng(sum(axes) + R)
    size = c.space.size
    every = np.arange(size, dtype=np.int64)
    g = GpuBackend(axes, R, mu=mu)
    try:
        for m in (1, max(2, size // 50), max(3, size // 7)):
            code = rng.integers(0, size, size=m).astype(np.int64)  # multiset
            g.recount(code)
            c.recount(code)
            assert (g.read_cnt() == c.read_cnt()).all()
            g.gain_map()
            c.gain_map()
            assert (g.map_read_at(every) == c.map_read_at(every)).all()
            g.loss_map()
            c.loss_map()
            assert (g.map_read_at(every) == c.map_read_at(every)).all()
    finally:
        g.close()


@pytest.mark.parametrize("policy", [0, 1])
@pytest.mark.parametrize("problem,axes,R,mu", CELLS)
def test_update_delta_and_swap_eval(monkeypatch, gpu_backend_factory,
                                    problem, axes, R, mu, policy):
    """ball_update_delta reports exactly the change of uncovered();
    swap_eval reproduces the drop / gather / undo sequence of the CPU
    backend and leaves cnt untouched. policy 0 disables the metric fast
    path (swap_eval then runs its mutate-gather-undo fallback)."""
    c = NumpyDPBackend(axes, R, problem=problem, mu=mu)
    rng = np.random.default_rng(77)
    size = c.space.size
    starts = [_near_cover(c, rng),
              np.sort(rng.choice(size, size=max(2, size // 40),
                                 replace=False)).astype(np.int64)]
    with gpu_cell(monkeypatch, problem, axes, R, mu, policy) as g:
        if g.swap_eval is None:
            pytest.skip("plugin lacks dct_swap_eval")
        for code in starts:
            g.recount(code)
            c.recount(code)
            # --- update_delta: singles and a batch, both signs
            for w in (code[:1], code[1:4], rng.integers(0, size, 3)):
                w = np.asarray(w, dtype=np.int64)
                sign = -1 if np.isin(w, code).all() else +1
                u0 = c.uncovered()
                c.ball_update(w, sign)
                dd = g.ball_update_delta(w, sign)
                assert dd == c.uncovered() - u0
                assert g.uncovered() == c.uncovered()
                c.ball_update(w, -sign)
                assert g.ball_update_delta(w, -sign) == -dd
            # --- swap_eval: per-drop candidate lists and a shared list
            k = min(5, len(code))
            drops = rng.choice(code, size=k, replace=False).astype(np.int64)
            cands = [rng.integers(0, size, size=int(rng.integers(0, 60)))
                     .astype(np.int64) for _ in range(k)]
            cnt0 = g.read_cnt().copy()
            loss, gains = g.swap_eval(drops, cands)
            assert (g.read_cnt() == cnt0).all()
            for i, w in enumerate(drops):
                wa = np.array([w], dtype=np.int64)
                assert loss[i] == c.ball_gather(wa, 1)[0]
                c.ball_update(wa, -1)
                if len(cands[i]):
                    assert (gains[i] == c.ball_gather(cands[i], 0)).all()
                c.ball_update(wa, +1)
            shared = rng.integers(0, size, size=40).astype(np.int64)
            loss2, g2 = g.swap_eval(drops, shared)
            assert g2.shape == (k, 40)
            for i, w in enumerate(drops):
                wa = np.array([w], dtype=np.int64)
                c.ball_update(wa, -1)
                assert (g2[i] == c.ball_gather(shared, 0)).all()
                c.ball_update(wa, +1)
            assert (loss2 == loss).all()
            # the single-drop form the polish uses (own fast path)
            for i in range(k):
                l1, g1 = g.swap_eval(drops[i:i + 1], [cands[i]])
                assert l1[0] == loss[i] and (g1[0] == gains[i]).all()
                l1, g1 = g.swap_eval(drops[i:i + 1], shared)
                assert g1.shape == (1, 40) and (g1[0] == g2[i]).all()
            l1, g1 = g.swap_eval(drops[:1], [np.zeros(0, dtype=np.int64)])
            assert l1[0] == loss[0] and len(g1[0]) == 0
        st = g.gpu_stats()
    if st and policy == 0:
        assert st["swap_eval_fast"] == 0
    if st and policy == 1:
        assert st["swap_eval_fast"] > 0


def test_swap_eval_rejects_inactive_drop(gpu_backend_factory):
    g = gpu_backend_factory(3, 5, 1)
    try:
        if g.swap_eval is None:
            pytest.skip("plugin lacks dct_swap_eval")
        g.recount(np.array([0, 7], dtype=np.int64))
        with pytest.raises(RuntimeError):
            g.swap_eval(np.array([5]), np.array([1, 2]))
        with pytest.raises(RuntimeError):
            g.ball_update_delta(np.array([5]), -1)
    finally:
        g.close()


def test_generic_reference_forms_match_cpu():
    """swap_eval_generic / ball_update_delta_generic on the CPU backend
    equal the step-by-step sequence and leave cnt unchanged."""
    from coldcore.gpu import ball_update_delta_generic, swap_eval_generic
    c = NumpyDPBackend([4] * 5, 2)
    rng = np.random.default_rng(4)
    code = _near_cover(c, rng)
    c.recount(code)
    cnt0 = c.read_cnt().copy()
    drops = code[:3]
    cands = [rng.integers(0, c.space.size, size=k).astype(np.int64)
             for k in (5, 0, 17)]
    loss, gains = swap_eval_generic(c, drops, cands)
    assert (c.read_cnt() == cnt0).all()
    for i, w in enumerate(drops):
        wa = np.array([w])
        assert loss[i] == c.ball_gather(wa, 1)[0]
        c.ball_update(wa, -1)
        if len(cands[i]):
            assert (gains[i] == c.ball_gather(cands[i], 0)).all()
        c.ball_update(wa, +1)
    u0 = c.uncovered()
    dd = ball_update_delta_generic(c, drops[:1], -1)
    assert dd == c.uncovered() - u0 == loss[0]


@pytest.mark.parametrize("cap", [1, 7, 100, 4096])
def test_extract_overflow_counts_exactly(gpu_backend_factory, cap):
    """found is the exact number of entries >= thr even when the output
    fills early (hits past the cap are tallied, not claimed); the stored
    entries are distinct, valid positions with their exact values."""
    c = NumpyDPBackend([5] * 6, 2)
    rng = np.random.default_rng(cap)
    code = np.sort(rng.choice(c.space.size, 30, replace=False))
    g = gpu_backend_factory(5, 6, 2)
    try:
        g.recount(code)
        c.recount(code)
        g.gain_map()
        c.gain_map()
        for thr in (1, 2, 5):
            idx, val, found = g.map_extract(thr, cap=cap)
            ci, cv, cf = c.map_extract(thr)
            assert found == cf
            assert len(idx) == min(cap, cf)
            assert len(np.unique(idx)) == len(idx)
            assert (c.map[idx] == val).all() and (val >= thr).all()
        assert g.count_eq(0) == c.count_eq(0)       # listing pass, overflow
    finally:
        g.close()

"""GPU plugin vs numpy reference on tiny instances.

Skipped automatically when no CUDA device / no built plugin (conftest).
Each test is seconds-scale and uses only megabytes of device memory, so
it is safe alongside other GPU tenants.

NOTE: the vendored plugin ABI is a per-process singleton — all GPU tests
share one session-scoped cell (q=3, n=4, R=1).
"""
import numpy as np
import pytest

from coldcore.reference import RefCoveringBackend

pytestmark = pytest.mark.gpu

Q, N, R = 3, 4, 1


@pytest.fixture(scope="module")     # singleton ABI: one live cell at a time
def cell(gpu_backend_factory):
    g = gpu_backend_factory(Q, N, R)
    yield g, RefCoveringBackend(Q, N, R)
    g.close()


def _rand_code(size, m, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(size, size=m, replace=False)).astype(np.int64)


def test_cnt_parity(cell):
    g, r = cell
    code = _rand_code(g.space.size, 7, 10)
    g.recount(code)
    r.recount(code)
    assert (g.read_cnt() == r.read_cnt()).all()
    for t in (0, 1, 2):
        assert g.count_eq(t) == r.count_eq(t)


def test_gain_map_parity(cell):
    g, r = cell
    code = _rand_code(g.space.size, 6, 11)
    g.recount(code)
    r.recount(code)
    g.gain_map()
    r.gain_map()
    every = np.arange(g.space.size, dtype=np.int64)
    assert (g.map_read_at(every) == r.map_read_at(every)).all()
    assert g.map_max() == r.map_max()


def test_loss_map_parity(cell):
    g, r = cell
    code = _rand_code(g.space.size, 9, 12)
    g.recount(code)
    r.recount(code)
    g.loss_map()
    r.loss_map()
    assert (g.map_read_at(code) == r.map_read_at(code)).all()


def test_ball_ops_parity(cell):
    g, r = cell
    code = _rand_code(g.space.size, 8, 13)
    g.recount(code[:5])
    r.recount(code[:5])
    g.ball_update(code[5:], +1)
    r.ball_update(code[5:], +1)
    assert (g.read_cnt() == r.read_cnt()).all()
    probe = _rand_code(g.space.size, 20, 14)
    for t in (0, 1):
        assert (g.ball_gather(probe, t) == r.ball_gather(probe, t)).all()


def test_extract_and_hist_parity(cell):
    g, r = cell
    code = _rand_code(g.space.size, 5, 15)
    g.recount(code)
    r.recount(code)
    g.gain_map()
    r.gain_map()
    mx = r.map_max()
    for thr in (1, max(1, mx // 2), mx):
        gi, gv, gf = g.map_extract(thr)
        ri, rv, rf = r.map_extract(thr)
        assert gf == rf
        # unordered: compare as sets of (idx, val)
        assert (np.sort(gi) == np.sort(ri)).all()
        assert sorted(zip(gi.tolist(), gv.tolist())) == \
               sorted(zip(ri.tolist(), rv.tolist()))
    assert (g.map_hist(16, mx) == r.map_hist(16, mx)).all()


def test_search_end_to_end_parity(cell):
    """Same seed, same algorithm: GPU and reference searches both produce
    valid covers of comparable size (paths may diverge via extract
    order, so we compare quality, not trajectories)."""
    from coldcore.search import Searcher
    g, r = cell
    sizes = {}
    for name, backend in (("gpu", g), ("ref", r)):
        s = Searcher(backend, seed=42)
        s.load(np.zeros(0, dtype=np.int64))
        assert s.greedy_fill(10 ** 6) == 0
        s.peel()
        assert s.verify_full() == 0
        sizes[name] = len(s.code)
    assert abs(sizes["gpu"] - sizes["ref"]) <= 2

"""GPU parity for open-boundary grid domination (PROB_GRID_LINF).

The first NON-shift-invariant family in the engine, so this is the real
ABI generality test: the clamped window pass and — the part the three
circulant families never needed — the boundary-aware support walk, which
must drop exactly those packed circular shifts that would wrap past an
edge. Cells include a non-square board, a 3-D board, and boards with
2R+1 > axis (the window clamps to the whole axis), plus mu = 2 to show
that multiple covering composes with a new family for free.
"""
from contextlib import contextmanager
from itertools import product

import numpy as np
import pytest

from coldcore.reference import RefGridBackend

pytestmark = pytest.mark.gpu


@contextmanager
def gpu_cell(axes, R, mu=1):
    from coldcore.gpu import GpuBackend
    g = GpuBackend(axes, R, problem="grid_linf", mu=mu)
    try:
        yield g
    finally:
        g.close()


CASES = [
    ([8, 8], 1),        # the chessboard king graph
    ([5, 9], 1),        # non-square board
    ([4, 4], 2),        # 2R+1 > axis: the window clamps to everything
    ([6, 3, 3], 1),     # 3-D, mixed extents
    ([3, 3], 3),        # R beyond the board
    ([16, 4], 1),       # max axis size
]


@pytest.mark.parametrize("axes,R", CASES)
@pytest.mark.parametrize("mu", [1, 2])
def test_fields_parity(gpu_backend_factory, axes, R, mu):
    r = RefGridBackend(axes, R, mu=mu)
    rng = np.random.default_rng(11)
    code = np.sort(rng.choice(r.space.size, size=7, replace=False))
    every = np.arange(r.space.size, dtype=np.int64)
    with gpu_cell(axes, R, mu) as g:
        g.recount(code)
        r.recount(code)
        assert (g.read_cnt() == r.read_cnt()).all()
        assert g.uncovered() == r.uncovered()
        for t in (0, 1, 2):
            assert g.count_eq(t) == r.count_eq(t)
        g.gain_map()
        r.gain_map()
        assert (g.map_read_at(every) == r.map_read_at(every)).all()
        g.loss_map()
        r.loss_map()
        assert (g.map_read_at(every) == r.map_read_at(every)).all()
        for t in (0, 1):
            assert (g.ball_gather(every, t) == r.ball_gather(every, t)).all()
        assert g.map_max() == r.map_max()


@pytest.mark.parametrize("axes,R", CASES)
def test_ball_walk_is_boundary_aware(gpu_backend_factory, axes, R):
    """One codeword at a time: cnt must be the clamped box, never a
    wrapped one. A single +1 at a corner is the sharpest form of the
    test — the circulant walk would light up cells on the far edge."""
    r = RefGridBackend(axes, R)
    with gpu_cell(axes, R) as g:
        for w in (0, r.space.size - 1, r.space.size // 2):
            g.recount(np.array([w], dtype=np.int64))
            ball = r.ball_indices(w)
            cnt = g.read_cnt()
            assert (cnt[ball] == 1).all()
            assert int(cnt.sum()) == len(ball)
            # incremental walk agrees with the transform
            g.recount(np.zeros(0, dtype=np.int64))
            g.ball_update(np.array([w], dtype=np.int64), +1)
            assert (g.read_cnt() == cnt).all()
            g.ball_update(np.array([w], dtype=np.int64), -1)
            assert int(g.read_cnt().sum()) == 0


def test_incremental_parity(gpu_backend_factory):
    axes, R = [5, 9], 1
    r = RefGridBackend(axes, R)
    rng = np.random.default_rng(6)
    code = np.sort(rng.choice(r.space.size, size=9, replace=False))
    with gpu_cell(axes, R) as g:
        g.recount(code[:4])
        g.ball_update(code[4:], +1)
        inc = g.read_cnt().copy()
        g.recount(code)
        assert (inc == g.read_cnt()).all()
        r.recount(code)
        assert (inc == r.read_cnt()).all()


@pytest.mark.parametrize("axes,rows,cols,opt", [
    ([8, 8], (1, 4, 6), (1, 4, 6), 9),      # 8x8 king graph: gamma = 9
    ([5, 9], (1, 4), (1, 4, 7), 6),         # 5x9 board: gamma = 6
])
def test_king_domination_end_to_end(gpu_backend_factory, axes, rows, cols, opt):
    """The untouched search core must reach the domination number on the
    GPU (both bounds are established in tests/test_grid.py; the explicit
    construction is re-verified here through the plugin's own fields)."""
    import time

    from coldcore.search import Searcher
    with gpu_cell(axes, 1) as g:
        cells = g.word_index(np.array(list(product(rows, cols))))
        g.recount(cells)
        assert g.uncovered() == 0 and len(cells) == opt
        s = Searcher(g, seed=2, deadline=time.time() + 300)
        s.load(np.zeros(0, dtype=np.int64))
        assert s.greedy_fill(10 ** 6) == 0
        s.peel()
        if len(s.code) > opt:
            s.ruin(len(s.code) - opt)
        assert s.lns(opt, round_budget_s=90) == 0
        assert len(s.code) == opt and s.verify_full() == 0
        assert (g.read_cnt() >= 1).all()


def test_grid_rejects_the_circulant_ox_format(gpu_backend_factory):
    """dct_use_fmt(1) must be refused for this problem: the ox walk is
    register-only and circulant, so it cannot express the clamp."""
    import ctypes

    from coldcore.patterns import support_patterns_ox
    with gpu_cell([8, 8], 1) as g:
        if not (hasattr(g._lib, "dct_use_fmt")
                and hasattr(g._lib, "dct_set_tbl2")):
            pytest.skip("plugin lacks the optional ox-format symbols")
        mask, offs = support_patterns_ox([8, 8], 1, "grid_linf")
        rc = g._lib.dct_set_tbl2(
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            offs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_longlong(len(mask)))
        assert rc == 0                       # the table itself is fine
        assert g._lib.dct_use_fmt(ctypes.c_int(1)) != 0    # the walk is not
        assert g._lib.dct_use_fmt(ctypes.c_int(0)) == 0

"""Counter boundary and multiple-cover owner regressions for ABI 1."""
import ctypes

import numpy as np
import pytest

from coldcore.gpu import GpuBackend

pytestmark = pytest.mark.gpu


def test_counter_limit_is_exact_and_rejection_preserves_state(gpu_backend_factory):
    # Repeated centers exercise the maximum count without a large space.
    code = np.zeros(65535, dtype=np.int64)
    with GpuBackend([3, 3], 1, mu=65535) as g:
        g.recount(code)
        before = g.read_cnt().copy()
        assert before[0] == 65535
        for operation in (
            lambda: g.recount(np.zeros(65536, dtype=np.int64)),
            lambda: g.ball_update([0], +1),
            lambda: g.ball_update([1], -1),
            lambda: g.ball_update([0], 2),
            lambda: g.recount([-1]),
            lambda: g.recount([9]),
        ):
            with pytest.raises(RuntimeError):
                operation()
            np.testing.assert_array_equal(g.read_cnt(), before)
        g.ball_update([0], -1)
        assert g.read_cnt(0, 1)[0] == 65534
        g.ball_update([0], +1)
        np.testing.assert_array_equal(g.read_cnt(), before)
        # Replacing the code also replaces the tracked removal multiplicities.
        g.recount([8])
        before = g.read_cnt().copy()
        with pytest.raises(RuntimeError):
            g.ball_update([8, 8], -1)
        np.testing.assert_array_equal(g.read_cnt(), before)
        g.ball_update([8], -1)
        assert not g.read_cnt().any()


@pytest.mark.parametrize('problem,axes,R', [
    ('hamming', [3, 3], 1), ('lee', [5, 5], 1),
    ('torus_linf', [5, 5], 1), ('grid_linf', [5, 5], 1),
])
@pytest.mark.parametrize('mu', [1, 2, 3])
def test_owner_loss_counts_every_critical_incidence(
        gpu_backend_factory, problem, axes, R, mu):
    code = np.array([0, 1, 2], dtype=np.int64)
    # Compute marginal deficit directly by removing each center in a CPU oracle.
    from coldcore.reference import (RefGridBackend, RefHammingBackend,
                                    RefLeeBackend, RefTorusBackend)
    cls = {'hamming': RefHammingBackend, 'lee': RefLeeBackend,
           'torus_linf': RefTorusBackend, 'grid_linf': RefGridBackend}[problem]
    ref = cls(axes, R, mu=mu)
    ref.recount(code)
    baseline = ref.uncovered()
    expected = []
    for i in range(len(code)):
        ref.recount(np.delete(code, i))
        expected.append(ref.uncovered() - baseline)
    with GpuBackend(axes, R, problem=problem, mu=mu, use_owner=True) as g:
        out = np.empty(len(code), dtype=np.int32)
        rc = g._lib.dct_loss_owner(
            code.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)), len(code),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        assert rc == 0
        np.testing.assert_array_equal(out, expected)
        g.ball_update(code, -1)
        assert not g.read_cnt().any()


def test_distinct_codeword_overflow_reproducer(gpu_backend_factory):
    # All 65,536 distinct centers cover 0 in H(12,3), radius 6.
    indices = np.arange(3 ** 12, dtype=np.int64)
    digits = indices.copy()
    weight = np.zeros(len(indices), dtype=np.int16)
    for _ in range(12):
        weight += digits % 3 != 0
        digits //= 3
    code = indices[weight <= 6][:65536]
    assert len(code) == 65536
    with GpuBackend([3] * 12, 6, mu=65535) as g:
        g.recount(code[:-1])
        assert g.read_cnt(0, 1)[0] == 65535
        with pytest.raises(RuntimeError):
            g.recount(code)
        assert g.read_cnt(0, 1)[0] == 65535
        g.ball_update(code[:1], -1)
        assert g.read_cnt(0, 1)[0] == 65534

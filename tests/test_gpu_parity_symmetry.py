"""The symmetry layer on the GPU backend: same numbers, no new kernels.

The claim `docs/symmetry.md` makes about scope is that the orbit
machinery is a *client* of the backend protocol — it adds no CUDA and no
ABI surface, so it must run unchanged on a GPU plugin and produce
byte-identical per-orbit quantities. That is what this pins:

  * the per-orbit gain/loss bounds (a whole-space transform plus a
    segment sum over the orbit CSR) agree exactly with the reference;
  * the exact orbit marginals (`add_orbit` / `drop_orbit`, which are a
    `ball_update` plus a deficit readback) agree exactly;
  * orbit search on the GPU cell finds the same structured optimum.

Skipped automatically when no CUDA device / no built plugin (conftest).
Seconds-scale and a few megabytes of device memory, so it is safe
alongside other GPU tenants.

NOTE: the vendored plugin ABI is a per-process singleton — one live cell
per module.
"""
import numpy as np
import pytest

from coldcore.reference import RefCoveringBackend
from coldcore.symmetry import SymmetricSearcher, parse_group

pytestmark = pytest.mark.gpu

Q, N, R = 3, 4, 1
SPEC = "translate:1011"           # one generator of the ternary Hamming code


@pytest.fixture(scope="module")   # singleton ABI: one live cell at a time
def cell(gpu_backend_factory):
    g = gpu_backend_factory(Q, N, R)
    yield g, RefCoveringBackend(Q, N, R)
    g.close()


def test_orbit_bounds_match_the_reference(cell):
    g, ref = cell
    sg = SymmetricSearcher(g, parse_group(g.space, SPEC), seed=1)
    sr = SymmetricSearcher(ref, parse_group(ref.space, SPEC), seed=1)
    assert (sg.part.labels == sr.part.labels).all()
    rng = np.random.default_rng(4)
    for o in rng.choice(len(sg.part), size=4, replace=False):
        assert sg.add_orbit(int(o)) == sr.add_orbit(int(o))   # exact marginals
        assert sg.uncov == sr.uncov
        assert (sg.gain_bounds() == sr.gain_bounds()).all()
        assert (sg.loss_bounds() == sr.loss_bounds()).all()
    for o in np.flatnonzero(sg.selected):
        assert sg.drop_orbit(int(o)) == sr.drop_orbit(int(o))
        assert sg.uncov == sr.uncov


def test_orbit_search_on_the_gpu_backend(cell):
    """K_3(4,1) = 9 as a union of three translate orbits, found by the
    unchanged orbit searcher driving the CUDA plugin."""
    g, _ref = cell
    s = SymmetricSearcher(g, parse_group(g.space, SPEC), seed=1)
    assert s.greedy_fill(10 ** 9) == 0
    s.peel()
    assert len(s.code) == 9 and int(s.selected.sum()) == 3
    assert s.verify_full() == 0
    # a perfect code: the GPU's own cnt field says every word once
    assert g.count_eq(1) == g.space.size

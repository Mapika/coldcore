"""GPU backend for explicit incidence problems: ctypes binding of plugin
ABI 2 (flat ``sparse_*`` symbols; docs/PLUGIN.md, "ABI 2: explicit
incidence").

Same shape as :mod:`coldcore.gpu`: a singleton cell per loaded ``.so``
per process, int return codes (``-100-e`` = CUDA error e), caller-
allocated outputs. The Python surface is the backend protocol of
:mod:`coldcore.protocol`, identical to
:class:`coldcore.incidence.SparseIncidenceBackend` — every query is
parity-tested against it (``tests/test_incidence.py``).

Placement: ``cnt`` (over the universe), ``map`` (over the candidates)
and the extraction buffers are always in HBM. The incidence matrix goes
to HBM when the whole cell fits the budget (env ``COLDCORE_HBM_BUDGET``,
default 80 GB, or the ``hbm_budget`` argument), else to CPU memory
pinned to the LPDDR NUMA node and read over C2C/ATS (``inc_mode=1``);
pass ``inc_mode`` to override.
"""
import ctypes
import os
import sys

import numpy as np

from .gpu import HBM_BUDGET, _check
from .incidence import IncidenceSpace
from .protocol import Backend

_libs = {}

_LL = ctypes.c_longlong
_I32P = ctypes.POINTER(ctypes.c_int32)
_I64P = ctypes.POINTER(ctypes.c_int64)
_LLP = ctypes.POINTER(_LL)

PLUGIN_NAME = "libcoldcore_sparse.so"


def _bind(path):
    lib = ctypes.CDLL(path)
    lib.sparse_abi_version.restype = ctypes.c_int
    lib.sparse_init.restype = ctypes.c_int
    lib.sparse_init.argtypes = [_LL, ctypes.c_int, _LL, _I32P, ctypes.c_int,
                                ctypes.c_int, _LLP, _LLP]
    lib.sparse_init_csr.restype = ctypes.c_int
    lib.sparse_init_csr.argtypes = [_LL, _LL, _LL, _I64P, _I32P, _I32P, _I32P,
                                    ctypes.c_int, ctypes.c_int, _LLP, _LLP]
    lib.sparse_is_csr.restype = ctypes.c_int
    for f in ("sparse_free", "sparse_set_mu", "sparse_get_mu",
              "sparse_inc_mode", "sparse_field", "sparse_map_max",
              "sparse_map_hist"):
        getattr(lib, f).restype = ctypes.c_int
    lib.sparse_recount.restype = ctypes.c_int
    lib.sparse_recount.argtypes = [_LLP, _LL]
    lib.sparse_ball_update.restype = ctypes.c_int
    lib.sparse_ball_update.argtypes = [_LLP, _LL, ctypes.c_int]
    lib.sparse_ball_gather.restype = ctypes.c_int
    lib.sparse_ball_gather.argtypes = [_LLP, _LL, ctypes.c_int, _I32P]
    lib.sparse_map_read_at.restype = ctypes.c_int
    lib.sparse_map_read_at.argtypes = [_LLP, _LL, _I32P]
    lib.sparse_read_cnt.restype = ctypes.c_int
    lib.sparse_read_cnt.argtypes = [_LL, _LL, _I32P]
    lib.sparse_map_extract.restype = _LL
    lib.sparse_map_extract.argtypes = [ctypes.c_int32, _I64P, _I32P,
                                       ctypes.c_int]
    for f in ("sparse_count_eq", "sparse_count_deficit", "sparse_dev_bytes",
              "sparse_host_bytes"):
        getattr(lib, f).restype = _LL
    return lib


def load_sparse_plugin(path):
    """Load (and cache) the ABI 2 shared library."""
    path = os.path.abspath(path)
    if path not in _libs:
        _libs[path] = _bind(path)
    return _libs[path]


def default_sparse_plugin_path():
    """``COLDCORE_SPARSE_PLUGIN_PATH`` env, else the in-tree CMake output,
    else the install locations :func:`coldcore.gpu.default_plugin_path`
    uses."""
    env = os.environ.get("COLDCORE_SPARSE_PLUGIN_PATH")
    if env:
        return env
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    candidates = [
        os.path.join(here, "build", "plugins", PLUGIN_NAME),
        os.path.join(sys.prefix, "lib", "coldcore", PLUGIN_NAME),
        os.path.join("/usr/local/lib/coldcore", PLUGIN_NAME),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


class GpuIncidenceBackend(Backend):
    """One explicit-incidence cell resident on the GPU.

    Parameters
    ----------
    inc : (n_cand, d) integer array, entries in ``[0, n_univ)``
        Uploaded at init (converted to int32 if needed); the caller's
        array is not referenced afterwards.
    n_univ, mu, extract_cap
        As for :class:`~coldcore.incidence.SparseIncidenceBackend`.
    inc_mode : None | 0 | 1
        Incidence placement: 0 HBM, 1 pinned CPU memory; None picks by
        size against ``hbm_budget``.
    plugin_path : path of ``libcoldcore_sparse.so`` (default: in-tree build)
    """

    indep_dist = None
    supports_hist = True

    def __init__(self, inc, n_univ, mu=1, extract_cap=1 << 22, inc_mode=None,
                 plugin_path=None, hbm_budget=None):
        inc = np.asarray(inc)
        if inc.ndim != 2 or not np.issubdtype(inc.dtype, np.integer):
            raise ValueError("inc must be a 2-D integer array (n_cand, d)")
        self.n_univ = int(n_univ)
        if not (1 <= self.n_univ <= np.iinfo(np.int32).max):
            raise ValueError("n_univ must be in 1..2^31-1 (int32 indices)")
        if inc.size and (int(inc.min()) < 0 or int(inc.max()) >= self.n_univ):
            raise ValueError("incidence entries must lie in [0, n_univ)")
        inc = np.ascontiguousarray(inc, dtype=np.int32)
        self.n_cand, self.d = (int(x) for x in inc.shape)
        self.mu = int(mu)
        if self.mu < 1:
            raise ValueError("mu must be >= 1")
        self.space = IncidenceSpace(self.n_cand)
        self.extract_cap = int(extract_cap)
        self._lib = lib = load_sparse_plugin(plugin_path
                                             or default_sparse_plugin_path())
        budget = HBM_BUDGET if hbm_budget is None else int(hbm_budget)
        inc_bytes = self.n_cand * self.d * 4
        other = (self.n_cand + self.n_univ) * 4 + self.extract_cap * 12
        if inc_mode is None:
            inc_mode = 0 if inc_bytes + other <= budget else 1
        if inc_mode not in (0, 1):
            raise ValueError("inc_mode must be 0 (HBM) or 1 (pinned host)")
        bd, bh = _LL(0), _LL(0)
        self._closed = True
        rc = lib.sparse_init(self.n_cand, self.d, self.n_univ,
                             inc.ctypes.data_as(_I32P), int(inc_mode),
                             self.extract_cap, ctypes.byref(bd),
                             ctypes.byref(bh))
        _check("sparse_init", rc)
        self._closed = False
        _check("sparse_set_mu", lib.sparse_set_mu(self.mu))
        self.inc_mode = int(lib.sparse_inc_mode())
        self.gpu_bytes = bd.value
        self.host_bytes = bh.value

    def close(self):
        if not self._closed:
            self._lib.sparse_free()
            self._closed = True

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _idx(idx):
        return np.ascontiguousarray(np.atleast_1d(idx), dtype=np.int64).ravel()

    # ---------------------------------------------------------- fields
    def recount(self, idx):
        idx = self._idx(idx)
        _check("sparse_recount", self._lib.sparse_recount(
            idx.ctypes.data_as(_LLP), len(idx)))

    def gain_map(self):
        _check("sparse_field(gain)", self._lib.sparse_field(0))

    def loss_map(self):
        _check("sparse_field(loss)", self._lib.sparse_field(1))

    # ----------------------------------------------------- incremental
    def ball_update(self, idx, delta):
        idx = self._idx(idx)
        _check("sparse_ball_update", self._lib.sparse_ball_update(
            idx.ctypes.data_as(_LLP), len(idx), int(delta)))

    def ball_gather(self, idx, target):
        """target is a coverage ROLE: 0 = deficient (cnt < mu), 1 =
        critical (1 <= cnt <= mu), anything else = literal cnt ==
        target (all literal at mu=1)."""
        idx = self._idx(idx)
        out = np.empty(len(idx), dtype=np.int32)
        _check("sparse_ball_gather", self._lib.sparse_ball_gather(
            idx.ctypes.data_as(_LLP), len(idx), int(target),
            out.ctypes.data_as(_I32P)))
        return out

    # ---------------------------------------------------------- queries
    def _ll(self, name, r):
        if r < 0:
            raise RuntimeError(f"{name} rc={r}")
        return int(r)

    def count_eq(self, target):
        """Literal count of universe elements with cnt == target."""
        return self._ll("sparse_count_eq",
                        self._lib.sparse_count_eq(int(target)))

    def uncovered(self):
        """Total deficit ``sum_u max(0, mu - cnt[u])``; 0 iff feasible."""
        return self._ll("sparse_count_deficit",
                        self._lib.sparse_count_deficit())

    def map_max(self):
        v = ctypes.c_int32(0)
        _check("sparse_map_max", self._lib.sparse_map_max(ctypes.byref(v)))
        return int(v.value)

    def map_hist(self, nbins, vmax):
        out = np.zeros(nbins, dtype=np.int32)
        _check("sparse_map_hist", self._lib.sparse_map_hist(
            int(nbins), int(vmax), out.ctypes.data_as(_I32P)))
        return out

    def map_extract(self, thr, cap=None):
        cap = min(cap or self.extract_cap, self.extract_cap)
        oi = np.empty(cap, dtype=np.int64)
        ov = np.empty(cap, dtype=np.int32)
        found = self._ll("sparse_map_extract", self._lib.sparse_map_extract(
            int(thr), oi.ctypes.data_as(_I64P), ov.ctypes.data_as(_I32P),
            cap))
        m = min(found, cap)
        return oi[:m], ov[:m], found

    def map_read_at(self, idx):
        idx = self._idx(idx)
        out = np.empty(len(idx), dtype=np.int32)
        _check("sparse_map_read_at", self._lib.sparse_map_read_at(
            idx.ctypes.data_as(_LLP), len(idx), out.ctypes.data_as(_I32P)))
        return out

    def read_cnt(self, off=0, length=None):
        """A range of the multiplicity field over the *universe*; a range
        past its end is clipped like a numpy slice (the host backend's
        behaviour), so a caller scanning by ``space.size`` — which is the
        candidate count — gets short or empty blocks, not an error."""
        off = max(0, int(off))
        hi = self.n_univ if length is None else min(self.n_univ, off + int(length))
        n = max(0, hi - off)
        out = np.empty(n, dtype=np.int32)
        if n:
            _check("sparse_read_cnt", self._lib.sparse_read_cnt(
                off, n, out.ctypes.data_as(_I32P)))
        return out.astype(np.uint16)

    def deficient_universe(self, limit, rng=None):
        """Up to ``limit`` universe elements with ``cnt < mu`` (one field
        readback; sorted, or a random subset when ``rng`` is given)."""
        cnt = np.empty(self.n_univ, dtype=np.int32)
        _check("sparse_read_cnt", self._lib.sparse_read_cnt(
            0, self.n_univ, cnt.ctypes.data_as(_I32P)))
        u = np.flatnonzero(cnt < self.mu)
        if rng is not None and len(u) > limit:
            u = np.sort(rng.choice(u, size=int(limit), replace=False))
        return u[:limit].astype(np.int64)

    def verify(self, idx):
        """Feasibility of ``idx`` from a fresh device recount (this
        overwrites ``cnt``). Problem layers override it with a host-side
        recomputation that does not touch the plugin."""
        self.recount(idx)
        return self.uncovered() == 0

    def __repr__(self):
        return (f"{type(self).__name__}(n_cand={self.n_cand}, d={self.d}, "
                f"n_univ={self.n_univ}, mu={self.mu}, inc_mode={self.inc_mode})")


class GpuWeightedIncidenceBackend(GpuIncidenceBackend):
    """The weighted CSR cell of :class:`coldcore.incidence
    .WeightedIncidenceBackend` on the GPU (``sparse_init_csr``): same
    exact weighted marginals, ``uncovered`` = weighted deficiency.
    ``indptr``/``indices``/``w`` follow the incidence placement,
    ``uw`` and the fields live in HBM."""

    def __init__(self, indptr, indices, w, n_univ, uw=None, mu=1,
                 extract_cap=1 << 22, inc_mode=None, plugin_path=None,
                 hbm_budget=None):
        indptr = np.ascontiguousarray(indptr, dtype=np.int64).ravel()
        indices = np.ascontiguousarray(indices, dtype=np.int32).ravel()
        w = np.ascontiguousarray(w, dtype=np.int32).ravel()
        self.n_univ = int(n_univ)
        if not (1 <= self.n_univ <= np.iinfo(np.int32).max):
            raise ValueError("n_univ must be in 1..2^31-1 (int32 indices)")
        if len(indptr) < 2 or indptr[0] != 0 or indptr[-1] != len(indices) or (
                np.diff(indptr) < 0).any():
            raise ValueError("indptr is not a valid CSR offset array")
        if len(w) != len(indices) or (w < 1).any():
            raise ValueError("w must match indices and be >= 1")
        if indices.size and (int(indices.min()) < 0
                             or int(indices.max()) >= self.n_univ):
            raise ValueError("indices must lie in [0, n_univ)")
        uw = (np.ones(self.n_univ, dtype=np.int32) if uw is None
              else np.ascontiguousarray(uw, dtype=np.int32).ravel())
        if len(uw) != self.n_univ or (uw < 1).any():
            raise ValueError("uw must have n_univ entries >= 1")
        self.n_cand = len(indptr) - 1
        self.nnz = len(indices)
        self.d = None
        self.row_len = np.diff(indptr)
        self.mu = int(mu)
        if self.mu < 1:
            raise ValueError("mu must be >= 1")
        self.space = IncidenceSpace(self.n_cand)
        self.extract_cap = int(extract_cap)
        self._lib = lib = load_sparse_plugin(plugin_path
                                             or default_sparse_plugin_path())
        budget = HBM_BUDGET if hbm_budget is None else int(hbm_budget)
        inc_bytes = self.nnz * 8 + (self.n_cand + 1) * 8
        other = (self.n_cand + 2 * self.n_univ) * 4 + self.extract_cap * 12
        if inc_mode is None:
            inc_mode = 0 if inc_bytes + other <= budget else 1
        if inc_mode not in (0, 1):
            raise ValueError("inc_mode must be 0 (HBM) or 1 (pinned host)")
        bd, bh = _LL(0), _LL(0)
        self._closed = True
        rc = lib.sparse_init_csr(self.n_cand, self.n_univ, self.nnz,
                                 indptr.ctypes.data_as(_I64P),
                                 indices.ctypes.data_as(_I32P),
                                 w.ctypes.data_as(_I32P),
                                 uw.ctypes.data_as(_I32P), int(inc_mode),
                                 self.extract_cap, ctypes.byref(bd),
                                 ctypes.byref(bh))
        _check("sparse_init_csr", rc)
        self._closed = False
        _check("sparse_set_mu", lib.sparse_set_mu(self.mu))
        self.inc_mode = int(lib.sparse_inc_mode())
        self.gpu_bytes = bd.value
        self.host_bytes = bh.value
        self.uw = uw

    def __repr__(self):
        return (f"{type(self).__name__}(n_cand={self.n_cand}, nnz={self.nnz}, "
                f"n_univ={self.n_univ}, mu={self.mu}, inc_mode={self.inc_mode})")

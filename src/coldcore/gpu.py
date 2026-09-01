"""GPU backend: ctypes binding of a transform plugin .so (v1 C ABI).

Binds the v1 plugin contract (flat ``dct_*`` symbols, singleton
context; see docs/PLUGIN.md). The ctypes binding style follows the
prototype record engine this framework grew out of. The full prototype
list also lives in include/coldcore/plugin.h.

v1 facts this module encodes:
  - singleton: one live cell per loaded .so per process (handles are v2);
  - required symbols: dct_init3/free, set_code, transform, ball_update,
    ball_gather, count_eq, map_max, map_extract(+64), map_read_at,
    read_cnt; legacy dct_init2/dct_init cover older builds;
  - optional (feature-tested by presence): loss_owner, set_tbl2/use_fmt,
    map_hist, host_bytes, and the mu-fold multiple-covering extension
    set_mu/get_mu/ball_gather_mu/count_deficit (docs/multi-coverage.md);
  - rc convention: 0 ok, negative engine error, -100-e = CUDA error e.
"""
import ctypes
import os
import sys

import numpy as np

from .patterns import PROBLEM_IDS, support_patterns
from .protocol import Backend
from .space import ProductSpace

# HBM budget for the auto placement policy (bytes). GH200: 96 GB HBM;
# leave room for support tables, extraction buffers and other tenants.
HBM_BUDGET = int(float(os.environ.get("COLDCORE_HBM_BUDGET", 80e9)))

_libs = {}          # path -> CDLL

_LL = ctypes.c_longlong
_I32P = ctypes.POINTER(ctypes.c_int32)


def _rc_msg(rc):
    if rc <= -100:
        return f"rc={rc} (CUDA error {-(rc + 100)})"
    return f"rc={rc}"


def _check(name, rc):
    if rc != 0:
        raise RuntimeError(f"{name} {_rc_msg(rc)}")


def _bind(path):
    lib = ctypes.CDLL(path)
    for f in ("dct_free", "dct_set_code", "dct_transform", "dct_ball_update",
              "dct_ball_gather", "dct_map_max", "dct_map_read_at",
              "dct_read_cnt"):
        getattr(lib, f).restype = ctypes.c_int
    lib.dct_count_eq.restype = _LL
    lib.dct_map_extract.restype = _LL
    lib.dct_read_cnt.argtypes = [_LL, _LL, ctypes.POINTER(ctypes.c_uint16)]
    lib.dct_init.restype = ctypes.c_int
    lib.dct_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.POINTER(ctypes.c_uint64), _LL,
                             ctypes.c_int, ctypes.POINTER(_LL)]
    if hasattr(lib, "dct_init2"):
        lib.dct_init2.restype = ctypes.c_int
        lib.dct_init2.argtypes = lib.dct_init.argtypes + [
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.dct_map_extract64.restype = _LL
    if hasattr(lib, "dct_init3"):
        lib.dct_init3.restype = ctypes.c_int
        lib.dct_init3.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
            ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), _LL,
            ctypes.c_int, ctypes.POINTER(_LL),
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
    # optional symbols
    if hasattr(lib, "dct_map_hist"):
        lib.dct_map_hist.restype = ctypes.c_int
    if hasattr(lib, "dct_loss_owner"):
        lib.dct_loss_owner.restype = ctypes.c_int
    if hasattr(lib, "dct_host_bytes"):
        lib.dct_host_bytes.restype = _LL
    for f in ("dct_set_mu", "dct_get_mu",       # mu-fold multiple covering
              "dct_ball_gather_mu"):
        if hasattr(lib, f):
            getattr(lib, f).restype = ctypes.c_int
    if hasattr(lib, "dct_count_deficit"):
        lib.dct_count_deficit.restype = _LL
    return lib


def load_plugin(path):
    """Load (and cache) a plugin shared library."""
    path = os.path.abspath(path)
    if path not in _libs:
        _libs[path] = _bind(path)
    return _libs[path]


def default_plugin_path():
    """COLDCORE_PLUGIN_PATH env, else the first existing candidate:
    the in-tree CMake build output (editable install), then system-ish
    locations for installed deployments."""
    env = os.environ.get("COLDCORE_PLUGIN_PATH")
    if env:
        return env
    # src layout: this file is <repo>/src/coldcore/gpu.py, so the repo
    # root (where the cmake build tree lives) is three levels up.
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    name = "libcoldcore_covering.so"
    candidates = [
        os.path.join(here, "build", "plugins", name),
        os.path.join(sys.prefix, "lib", "coldcore", name),
        os.path.join("/usr/local/lib/coldcore", name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]      # keep the old behavior for the error message


class GpuBackend(Backend):
    """One problem cell resident on the GPU / host-coherent LPDDR.

    problem: "hamming" (covering codes, mixed radix ok), "torus_linf"
    (dominating sets on torus grids, Chebyshev balls), "lee" (Lee-metric
    covering codes: sum of circular per-axis distances) or "grid_linf"
    (the same Chebyshev balls on an OPEN board — no wraparound, so the
    windows clamp at the boundary; see docs/plugin-grid.md).
    mu: covering multiplicity — every word must be covered at least mu
    times (K^mu_q(n,R); mu=1 is the ordinary problem, see
    docs/multi-coverage.md). Needs a plugin exporting dct_set_mu.
    layers_mode / cnt_mode: None = auto placement policy, else
    0 = HBM (cudaMalloc), 1 = LPDDR via ATS, 2 = managed preferred-CPU.
    """

    def __init__(self, axes, R, problem="hamming", plugin_path=None,
                 extract_cap=1 << 22, layers_mode=None, cnt_mode=None,
                 use_owner=False, mu=1):
        self.problem = problem
        self.R = R
        self.mu = mu = int(mu)
        if mu < 1:
            raise ValueError("mu must be >= 1")
        self.space = ProductSpace(axes)
        size = self.space.size
        self.extract_cap = extract_cap
        # sum metrics dominate digit distance -> batched peel stays valid
        self.indep_dist = 2 * R if problem in ("hamming", "lee") else None
        self._lib = lib = load_plugin(plugin_path or default_plugin_path())
        self.supports_hist = hasattr(lib, "dct_map_hist")
        self._has_mu = hasattr(lib, "dct_set_mu")
        self._has_gather_mu = hasattr(lib, "dct_ball_gather_mu")
        self._has_deficit = hasattr(lib, "dct_count_deficit")
        if mu > 1 and not self._has_mu:
            raise RuntimeError(
                "plugin lacks dct_set_mu; mu-fold multiple covering "
                "(mu>1) is unsupported by this build")
        pats = support_patterns(self.space.axes, R, problem)
        # auto placement (PLUGIN.md policy): everything HBM if it fits
        # the budget; else layers -> LPDDR, cnt HBM while 2|X| <= 40 GB.
        nlayers = 1 if problem in ("torus_linf", "grid_linf") else (R + 1)
        total = size * (4 * nlayers + 2)
        if layers_mode is None:
            layers_mode = 0 if total <= HBM_BUDGET else 1
        if cnt_mode is None:
            cnt_mode = 0 if (layers_mode == 0 or size * 2 <= 40e9) else 1
        self.layers_mode, self.cnt_mode = layers_mode, cnt_mode
        by = _LL(0)
        pp = pats.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        if hasattr(lib, "dct_init3"):
            ax = (ctypes.c_int * self.space.n)(*self.space.axes)
            rc = lib.dct_init3(PROBLEM_IDS[problem], self.space.n, ax, R,
                               pp, len(pats), extract_cap, ctypes.byref(by),
                               layers_mode, cnt_mode, 1 if use_owner else 0)
        else:
            if problem != "hamming" or self.space.q is None:
                raise RuntimeError(
                    "plugin predates dct_init3; only homogeneous Hamming "
                    "cells are supported by dct_init/dct_init2")
            needs_hc = (layers_mode or cnt_mode or use_owner
                        or size > 2**32 - 2)
            if needs_hc and not hasattr(lib, "dct_init2"):
                raise RuntimeError(
                    f"cell needs the host-coherent plugin (layers_mode="
                    f"{layers_mode}, space={size:.3g})")
            if hasattr(lib, "dct_init2"):
                rc = lib.dct_init2(self.space.q, self.space.n, R, pp,
                                   len(pats), extract_cap, ctypes.byref(by),
                                   layers_mode, cnt_mode,
                                   1 if use_owner else 0)
            else:
                rc = lib.dct_init(self.space.q, self.space.n, R, pp,
                                  len(pats), extract_cap, ctypes.byref(by))
        _check("init", rc)
        if self._has_mu:            # singleton ABI: always (re)state mu
            _check("set_mu", lib.dct_set_mu(mu))
        self.gpu_bytes = by.value
        self.host_bytes = (int(lib.dct_host_bytes())
                           if hasattr(lib, "dct_host_bytes") else 0)
        self._closed = False

    def close(self):
        if not self._closed:
            self._lib.dct_free()
            self._closed = True

    # ---------------- solution / fields ----------------
    def _set_code(self, idx):
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        _check("set_code", self._lib.dct_set_code(
            idx.ctypes.data_as(ctypes.POINTER(_LL)), len(idx)))

    def recount(self, idx):
        self._set_code(idx)
        _check("transform(code->cnt)", self._lib.dct_transform(0, 2))

    def gain_map(self):
        _check("transform(uncov->gain)", self._lib.dct_transform(1, 1))

    def loss_map(self):
        _check("transform(cnt1->loss)", self._lib.dct_transform(2, 1))

    # ---------------- incremental / exact ----------------
    def ball_update(self, idx, delta):
        idx = np.ascontiguousarray(np.atleast_1d(idx), dtype=np.int64)
        _check("ball_update", self._lib.dct_ball_update(
            idx.ctypes.data_as(ctypes.POINTER(_LL)), len(idx), delta))

    def ball_gather(self, idx, target):
        """Exact per-candidate support-set count.

        target is a coverage ROLE (identical to the literal cnt value at
        mu=1): 0 = deficient (cnt < mu) = exact placement gain, 1 =
        critical (1 <= cnt <= mu) = exact removal loss. Any other target
        is a literal cnt == target query.
        """
        idx = np.ascontiguousarray(np.atleast_1d(idx), dtype=np.int64)
        out = np.empty(len(idx), dtype=np.int32)
        p = idx.ctypes.data_as(ctypes.POINTER(_LL))
        if self.mu == 1 or target not in (0, 1):
            _check("ball_gather", self._lib.dct_ball_gather(
                p, len(idx), target, out.ctypes.data_as(_I32P)))
        elif self._has_gather_mu:
            _check("ball_gather_mu", self._lib.dct_ball_gather_mu(
                p, len(idx), target, out.ctypes.data_as(_I32P)))
        else:                       # fallback: sum literal levels (mu small)
            lo, hi = (0, self.mu - 1) if target == 0 else (1, self.mu)
            out[:] = 0
            tmp = np.empty(len(idx), dtype=np.int32)
            for t in range(lo, hi + 1):
                _check("ball_gather", self._lib.dct_ball_gather(
                    p, len(idx), t, tmp.ctypes.data_as(_I32P)))
                out += tmp
        return out

    # ---------------- queries ----------------
    def count_eq(self, target):
        """Literal count of words with cnt == target (mu-blind)."""
        return int(self._lib.dct_count_eq(target))

    def uncovered(self):
        """Total deficit sum_x max(0, mu - cnt(x)); 0 iff feasible.

        This is the quantity the search core decrements by the placement
        gain, and at mu=1 it is exactly the uncovered-word count.
        """
        if self.mu == 1:
            return int(self._lib.dct_count_eq(0))
        if self._has_deficit:
            return int(self._lib.dct_count_deficit())
        return sum((self.mu - t) * int(self._lib.dct_count_eq(t))
                   for t in range(self.mu))

    def map_max(self):
        v = ctypes.c_int32(0)
        _check("map_max", self._lib.dct_map_max(ctypes.byref(v)))
        return int(v.value)

    def map_hist(self, nbins, vmax):
        if not self.supports_hist:
            raise NotImplementedError("plugin lacks dct_map_hist")
        out = np.zeros(nbins, dtype=np.int32)
        _check("map_hist", self._lib.dct_map_hist(
            nbins, vmax, out.ctypes.data_as(_I32P)))
        return out

    def map_extract(self, thr, cap=None):
        cap = min(cap or self.extract_cap, self.extract_cap)
        ov = np.empty(cap, dtype=np.int32)
        if self.space.size > 2**32 - 2:
            oi = np.empty(cap, dtype=np.int64)
            found = self._lib.dct_map_extract64(
                int(thr), oi.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ov.ctypes.data_as(_I32P), cap)
            m = min(int(found), cap)
            return oi[:m], ov[:m], int(found)
        oi = np.empty(cap, dtype=np.uint32)
        found = self._lib.dct_map_extract(
            int(thr), oi.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ov.ctypes.data_as(_I32P), cap)
        m = min(int(found), cap)
        return oi[:m].astype(np.int64), ov[:m], int(found)

    def map_read_at(self, idx):
        idx = np.ascontiguousarray(np.atleast_1d(idx), dtype=np.int64)
        out = np.empty(len(idx), dtype=np.int32)
        _check("map_read_at", self._lib.dct_map_read_at(
            idx.ctypes.data_as(ctypes.POINTER(_LL)), len(idx),
            out.ctypes.data_as(_I32P)))
        return out

    def read_cnt(self, off=0, length=None):
        length = length if length is not None else self.space.size - off
        out = np.empty(length, dtype=np.uint16)
        _check("read_cnt", self._lib.dct_read_cnt(
            off, length, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))))
        return out


class GpuCoveringBackend(GpuBackend):
    """Homogeneous covering-code cell K_q(n,R) — convenience wrapper."""

    def __init__(self, q, n, R, **kw):
        super().__init__([q] * n, R, problem="hamming", **kw)
        self.q, self.n = q, n

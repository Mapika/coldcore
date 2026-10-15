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
  - optional v0.3: ball_update_delta (an update that reports the exact
    deficit change), swap_eval (read-only batched swap evaluation) and
    stats (fast-path counters); see docs/PLUGIN.md "v0.3 additions";
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
    # v0.3 optional: reporting updates, batched swap evaluation, counters
    if hasattr(lib, "dct_ball_update_delta"):
        lib.dct_ball_update_delta.restype = ctypes.c_int
        lib.dct_ball_update_delta.argtypes = [
            ctypes.POINTER(_LL), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(_LL)]
    if hasattr(lib, "dct_swap_eval"):
        lib.dct_swap_eval.restype = ctypes.c_int
        lib.dct_swap_eval.argtypes = [
            ctypes.POINTER(_LL), ctypes.c_int, ctypes.POINTER(_LL),
            ctypes.POINTER(_LL), _I32P, _I32P]
    if hasattr(lib, "dct_stats"):
        lib.dct_stats.restype = ctypes.c_int
    return lib


def locality_order(pats, axes):
    """Reorder a v1 support table so neighbouring entries touch nearby words.

    Every walk kernel maps table entry ``t`` to thread ``t`` of a block, so
    the table order decides which ``cnt`` addresses one warp touches. The
    builders enumerate the highest changed position fastest, which puts
    the 32 lanes of a warp up to ``ax^(n-1)`` words apart: one 32-byte
    sector per lane. Sorting by the unwrapped offset ``sum delta * stride``
    makes the LOW positions vary fastest, so lanes share sectors and cache
    lines. The walks are order-free (exact integer sums and atomic
    increments), so every result is unchanged; only the memory traffic is.
    ``COLDCORE_BALL_ORDER=native`` keeps the builder's order (for A/B).
    Entry 0 stays the empty pattern (the only key 0)."""
    if os.environ.get("COLDCORE_BALL_ORDER", "locality") == "native":
        return pats
    pats = np.ascontiguousarray(pats, dtype=np.uint64)
    stride = np.cumprod([1] + list(axes[:-1])).astype(np.int64)
    key = np.zeros(len(pats), dtype=np.int64)
    for k in range(8):
        byte = ((pats >> np.uint64(8 * k)) & np.uint64(255)).astype(np.int64)
        pos, delta = byte >> 4, byte & 15
        key += np.where(byte != 0, delta * stride[np.minimum(pos, len(axes) - 1)],
                        0)
    return np.ascontiguousarray(pats[np.argsort(key, kind="stable")])


def swap_eval_generic(backend, drops, cands):
    """Reference form of :meth:`GpuBackend.swap_eval` for any backend.

    Composes the protocol operations exactly as the polish does today:
    ``ball_gather(w, 1)``, ``ball_update(w, -1)``, ``ball_gather(c, 0)``,
    ``ball_update(w, +1)`` per drop word, so ``cnt`` is back to its prior
    state on return. A caller can use it wherever the backend has no
    ``swap_eval`` (the CPU backends), keeping one code path::

        sw = getattr(b, "swap_eval", None) or partial(swap_eval_generic, b)
    """
    drops = np.atleast_1d(np.asarray(drops, dtype=np.int64))
    shared = not isinstance(cands, (list, tuple))
    parts = ([np.atleast_1d(np.asarray(cands, dtype=np.int64))] * len(drops)
             if shared else
             [np.atleast_1d(np.asarray(c, dtype=np.int64)) for c in cands])
    loss = np.empty(len(drops), dtype=np.int32)
    gains = []
    for i, w in enumerate(drops.tolist()):
        wa = np.array([w], dtype=np.int64)
        loss[i] = int(backend.ball_gather(wa, 1)[0])
        if len(parts[i]):
            backend.ball_update(wa, -1)
            gains.append(np.asarray(backend.ball_gather(parts[i], 0),
                                    dtype=np.int32))
            backend.ball_update(wa, +1)
        else:
            gains.append(np.zeros(0, dtype=np.int32))
    if shared:
        return loss, (np.stack(gains) if len(gains)
                      else np.zeros((0, len(parts[0]) if parts else 0),
                                    dtype=np.int32))
    return loss, gains


def ball_update_delta_generic(backend, idx, delta):
    """Reference form of :meth:`GpuBackend.ball_update_delta`: the update
    plus the change of ``uncovered()`` it caused (two whole-field reads on
    a backend without the fused operation)."""
    u0 = backend.uncovered()
    backend.ball_update(idx, delta)
    return backend.uncovered() - u0


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
        if not 1 <= mu <= 65535:
            raise ValueError("mu must be in 1..65535")
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
        # v0.3 optional capabilities (feature-tested; None = absent, the
        # same convention as the protocol's optional locality hooks)
        if not hasattr(lib, "dct_ball_update_delta"):
            self.ball_update_delta = None
        if not hasattr(lib, "dct_swap_eval"):
            self.swap_eval = None
        self._ext_bufs = None       # persistent map_extract buffers
        if mu > 1 and not self._has_mu:
            raise RuntimeError(
                "plugin lacks dct_set_mu; mu-fold multiple covering "
                "(mu>1) is unsupported by this build")
        pats = locality_order(support_patterns(self.space.axes, R, problem),
                              self.space.axes)
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

    # ---------------- v0.3 optional operations ----------------
    def ball_update_delta(self, idx, delta):
        """``ball_update(idx, delta)`` that also returns the exact change
        of :meth:`uncovered` (the total deficit) it caused.

        Same validation and effect as :meth:`ball_update`. The change is
        read off the update's own atomics (no whole-space scan), so for a
        single word ``ball_update_delta([w], -1) == ball_gather([w], 1)``
        before the call, and ``ball_update_delta([v], +1) ==
        -ball_gather([v], 0)``. Synchronous (one device round trip).
        ``None`` on a plugin without ``dct_ball_update_delta``."""
        idx = np.ascontiguousarray(np.atleast_1d(idx), dtype=np.int64)
        dd = _LL(0)
        _check("ball_update_delta", self._lib.dct_ball_update_delta(
            idx.ctypes.data_as(ctypes.POINTER(_LL)), len(idx), int(delta),
            ctypes.byref(dd)))
        return int(dd.value)

    def swap_eval(self, drops, cands):
        """Exact costs of many swaps in one call, without changing state.

        ``drops``: k incumbent words (each must be in the solution).
        ``cands``: one candidate array per drop (a list of k arrays), or a
        single array evaluated against every drop.

        Returns ``(loss, gains)``: ``loss[i]`` is the deficit increase of
        removing one copy of ``drops[i]`` (= ``ball_gather(drops[i], 1)``)
        and ``gains[i][j]`` the deficit decrease of then adding
        ``cands[i][j]`` (= ``ball_gather(cands[i][j], 0)`` evaluated with
        that one copy removed). The swap ``drops[i] -> cands[i][j]`` takes
        the deficit from ``u`` to ``u + loss[i] - gains[i][j]``, exactly the
        number the drop / gather / undo sequence produces. ``gains`` is a
        list of arrays, or a ``(k, len(cands))`` array when one shared
        candidate array was given. ``None`` on a plugin without
        ``dct_swap_eval``."""
        drops = np.ascontiguousarray(np.atleast_1d(drops), dtype=np.int64)
        k = len(drops)
        shared = not isinstance(cands, (list, tuple))
        # plugin limit per call, drops and candidates alike: the plugin
        # validates both with its 65,535-occurrence word check
        cap = (1 << 16) - 1
        if shared:
            c1 = np.ascontiguousarray(np.atleast_1d(cands), dtype=np.int64)
            parts = [c1] * k
        else:
            if len(cands) != k:
                raise ValueError("need one candidate array per drop")
            parts = [np.ascontiguousarray(np.atleast_1d(c), dtype=np.int64)
                     for c in cands]
        if k == 1 and len(parts[0]) <= cap:          # the polish's call
            c = parts[0]
            off = np.array([0, len(c)], dtype=np.int64)
            g = np.empty(max(len(c), 1), dtype=np.int32)
            lo = np.empty(1, dtype=np.int32)
            cp = c if len(c) else np.zeros(1, dtype=np.int64)
            _check("swap_eval", self._lib.dct_swap_eval(
                drops.ctypes.data_as(ctypes.POINTER(_LL)), 1,
                cp.ctypes.data_as(ctypes.POINTER(_LL)),
                off.ctypes.data_as(ctypes.POINTER(_LL)),
                g.ctypes.data_as(_I32P), lo.ctypes.data_as(_I32P)))
            g = g[:len(c)]
            return lo, (g[None, :] if shared else [g])
        loss = np.empty(k, dtype=np.int32)
        gains = [None] * k
        i = 0
        while i < k:
            j, tot = i, 0
            while j < k and j - i < cap and tot + len(parts[j]) <= cap:
                tot += len(parts[j])
                j += 1
            if j == i:
                raise ValueError(f"over {cap} candidates for one drop")
            cat = (np.concatenate(parts[i:j]) if tot
                   else np.zeros(1, dtype=np.int64))
            off = np.zeros(j - i + 1, dtype=np.int64)
            off[1:] = np.cumsum([len(p) for p in parts[i:j]])
            g = np.empty(max(tot, 1), dtype=np.int32)
            lo = np.empty(j - i, dtype=np.int32)
            _check("swap_eval", self._lib.dct_swap_eval(
                drops[i:].ctypes.data_as(ctypes.POINTER(_LL)), j - i,
                cat.ctypes.data_as(ctypes.POINTER(_LL)),
                off.ctypes.data_as(ctypes.POINTER(_LL)),
                g.ctypes.data_as(_I32P), lo.ctypes.data_as(_I32P)))
            loss[i:j] = lo
            for t in range(j - i):
                gains[i + t] = g[off[t]:off[t + 1]].copy()
            i = j
        if shared:
            return loss, (np.stack(gains) if k
                          else np.zeros((0, len(c1)), dtype=np.int32))
        return loss, gains

    def gpu_stats(self):
        """Fast-path counters since init (empty dict on older plugins)."""
        if not hasattr(self._lib, "dct_stats"):
            return {}
        buf = (_LL * 8)()
        self._lib.dct_stats(buf, 8)
        keys = ("gather_set_side", "gather_walk", "set_scans",
                "set_overflows", "swap_eval_fast", "swap_eval_fallback",
                "count_eq0_listed", "gather_set_tiled")
        return {k: int(v) for k, v in zip(keys, buf)}

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

    def _extract_bufs(self, cap):
        # Output buffers. Below 2^20 entries a fresh pair per call is
        # cheapest (results are returned as views, no copy). From 2^20 up
        # (the default cap is extract_cap = 4M) a fresh 16+ MB pair cost
        # ~0.2 ms per call on this host -- more than the extraction itself
        # on small cells -- so those reuse one persistent pair and return
        # copies of the found entries.
        wide = self.space.size > 2**32 - 2
        if cap < (1 << 20):
            return (np.empty(cap, dtype=np.int64 if wide else np.uint32),
                    np.empty(cap, dtype=np.int32)), False
        b = self._ext_bufs
        if b is None or len(b[1]) < cap:
            b = (np.empty(cap, dtype=np.int64 if wide else np.uint32),
                 np.empty(cap, dtype=np.int32))
            self._ext_bufs = b
        return b, True

    def map_extract(self, thr, cap=None):
        cap = min(cap or self.extract_cap, self.extract_cap)
        (oi, ov), shared = self._extract_bufs(cap)
        if self.space.size > 2**32 - 2:
            found = self._lib.dct_map_extract64(
                int(thr), oi.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ov.ctypes.data_as(_I32P), cap)
            if found < 0:
                _check("map_extract64", int(found))
            m = min(int(found), cap)
            if shared:
                return oi[:m].copy(), ov[:m].copy(), int(found)
            return oi[:m], ov[:m], int(found)
        found = self._lib.dct_map_extract(
            int(thr), oi.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ov.ctypes.data_as(_I32P), cap)
        if found < 0:
            _check("map_extract", int(found))
        m = min(int(found), cap)
        return (oi[:m].astype(np.int64), ov[:m].copy() if shared else ov[:m],
                int(found))

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

"""Backend factory: one place that turns a cell description into a live
backend, for the CLI, the campaign runner and scripts alike.

Three kinds, same protocol:

``ref``  brute-force numpy oracle, O(|X|^2), capped at 2^16 words
``cpu``  :class:`~coldcore.dp.NumpyDPBackend` — the plugin's axis DP in
         numpy, O(n|X|); the default for CPU work
``gpu``  the CUDA plugin via ctypes

A **cell spec** is the compact string the campaign tooling passes around:

    K3(4,1)                     homogeneous Hamming, q=3 n=4 R=1
    hamming:3x3x3x3:1           explicit axes
    lee:5x5:1                   Lee metric
    torus_linf:9x9:1  grid_linf:8x8:1
    K2(6,1)mu3                  mu-fold multiple covering

``problem:axes:R`` with optional ``mu<k>`` suffix is the general form;
``Kq(n,R)`` is the covering-code shorthand.
"""
import re

BACKENDS = ("ref", "cpu", "gpu")

_K = re.compile(r"^K(\d+)\((\d+),(\d+)\)$")


def parse_cell(spec):
    """Cell spec string -> ``(problem, axes, R, mu)``."""
    s = str(spec).strip()
    mu = 1
    m = re.search(r"mu(\d+)$", s)
    if m:
        mu = int(m.group(1))
        s = s[:m.start()]
    m = _K.match(s)
    if m:
        q, n, R = (int(x) for x in m.groups())
        return "hamming", [q] * n, R, mu
    bits = s.split(":")
    if len(bits) != 3:
        raise ValueError(f"cannot read cell spec {spec!r}")
    problem, axes, R = bits
    problem = "hamming" if problem == "covering" else problem
    return problem, [int(a) for a in axes.split("x")], int(R), mu


def split_cells(text):
    """Split a comma-separated cell list, respecting ``Kq(n,R)`` parens."""
    out, buf, depth = [], [], 0
    for ch in str(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    out.append("".join(buf).strip())
    return [c for c in out if c]


def cell_label(problem, axes, R, mu=1):
    """The canonical spec string for a cell (round-trips ``parse_cell``)."""
    tail = f"mu{mu}" if mu > 1 else ""
    if problem == "hamming" and len(set(axes)) == 1:
        return f"K{axes[0]}({len(axes)},{R}){tail}"
    return f"{problem}:{'x'.join(str(a) for a in axes)}:{R}{tail}"


def make_backend(kind, axes, R, problem="hamming", mu=1, plugin_path=None,
                 **kw):
    """Build one backend. ``kind`` is one of :data:`BACKENDS`."""
    problem = "hamming" if problem == "covering" else problem
    if kind == "ref":
        from .reference import (RefGridBackend, RefHammingBackend,
                                RefLeeBackend, RefTorusBackend)
        cls = {"hamming": RefHammingBackend, "torus_linf": RefTorusBackend,
               "grid_linf": RefGridBackend, "lee": RefLeeBackend}.get(problem)
        if cls is None:
            raise ValueError(f"no reference backend for problem={problem}")
        return cls(list(axes), R, mu=mu, **kw)
    if kind == "cpu":
        from .dp import NumpyDPBackend
        return NumpyDPBackend(list(axes), R, problem=problem, mu=mu, **kw)
    if kind == "gpu":
        from .gpu import GpuBackend
        return GpuBackend(list(axes), R, problem=problem, mu=mu,
                          plugin_path=plugin_path, **kw)
    raise ValueError(f"unknown backend {kind!r}; pick one of {BACKENDS}")


def backend_for_spec(kind, spec, plugin_path=None, **kw):
    """``make_backend`` straight from a cell spec string."""
    problem, axes, R, mu = parse_cell(spec)
    return make_backend(kind, axes, R, problem=problem, mu=mu,
                        plugin_path=plugin_path, **kw)

"""Host-side support-table ("ball") builders for transform plugins.

Mirrors the prototype record engine's host binding; the packed
formats are part of the v1 plugin contract (docs/PLUGIN.md):

v1 format : one uint64 per pattern; byte k = (pos<<4 | delta), delta a
            circular shift in 1..ax_pos-1, zero byte terminates; <= 8
            bytes per pattern. Entry 0 = empty pattern (the center).
ox format : (mask uint16, nibble-offsets uint64) per pattern; nibble i
            of offs = delta at position i. Same enumeration.

Both formats require ax_j <= 16 and pattern weight <= 8 changed
positions (Hamming: weight <= R; torus Chebyshev: weight <= n; Lee:
weight <= min(n, R), each changed position costing Lee weight >= 1).

``grid_linf`` (open board, no wraparound) shares the torus table: the
deltas are the same circular shifts, and the *walk* — not the table —
rejects the ones that would wrap past a boundary for a given centre, so
balls shrink at the edges (see docs/plugin-grid.md).
"""
from itertools import combinations, product
from math import comb

import numpy as np

PROBLEM_IDS = {"hamming": 0, "torus_linf": 1, "lee": 2, "grid_linf": 3}


def _lee_wt(delta, a):
    """Per-axis Lee weight of a circular shift delta on Z_a."""
    return min(delta, a - delta)


def _delta_sets(axes, R, problem):
    """Per-position admissible circular-shift deltas."""
    if problem == "hamming":
        return [list(range(1, a)) for a in axes]
    if problem == "torus_linf":
        return [list(range(1, R + 1)) + list(range(a - R, a))
                for a in axes]
    if problem == "grid_linf":
        # same shifts as the torus, deduplicated: an oversized window
        # (2R+1 > a) is legal on an open board, and would otherwise list
        # a delta twice.  Boundary rejection happens in the walk.
        return [sorted({d for d in range(1, a) if d <= R or a - d <= R})
                for a in axes]
    if problem == "lee":
        return [[d for d in range(1, a) if _lee_wt(d, a) <= R]
                for a in axes]
    raise ValueError(problem)


def _max_pattern_weight(n, R, problem):
    """Max number of changed positions in one support pattern."""
    if problem == "hamming":
        return R
    if problem == "lee":
        return min(n, R)
    return n                        # torus_linf / grid_linf: any subset


def _admissible(subs, deltas, axes, R, problem):
    """Joint constraint on a delta assignment (per-position sets already
    enforced); only the Lee ball couples positions (total weight <= R)."""
    if problem != "lee":
        return True
    return sum(_lee_wt(dl, axes[p])
               for p, dl in zip(subs, deltas)) <= R


def ball_patterns(q, n, R):
    """Fast numpy path: homogeneous Hamming ball, v1 format."""
    pats = [np.zeros(1, dtype=np.uint64)]          # r = 0
    for r in range(1, R + 1):
        subs = np.array(list(combinations(range(n), r)), dtype=np.uint64)
        deltas = np.indices((q - 1,) * r).reshape(r, -1).T.astype(np.uint64) + 1
        out = np.zeros((len(subs), len(deltas)), dtype=np.uint64)
        for k in range(r):
            byte = (subs[:, k][:, None] << np.uint64(4)) | deltas[None, :, k]
            out |= byte << np.uint64(8 * k)
        pats.append(out.ravel())
    return np.concatenate(pats)


def ball_patterns_ox(q, n, R):
    """Fast numpy path: homogeneous Hamming ball, ox format."""
    masks = [np.zeros(1, dtype=np.uint16)]
    offs = [np.zeros(1, dtype=np.uint64)]
    for r in range(1, R + 1):
        subs = np.array(list(combinations(range(n), r)), dtype=np.uint64)
        deltas = np.indices((q - 1,) * r).reshape(r, -1).T.astype(np.uint64) + 1
        m = np.zeros((len(subs), len(deltas)), dtype=np.uint16)
        o = np.zeros((len(subs), len(deltas)), dtype=np.uint64)
        for k in range(r):
            m |= (np.uint16(1) << subs[:, k].astype(np.uint16))[:, None]
            o |= deltas[None, :, k] << (np.uint64(4) * subs[:, k][:, None])
        masks.append(m.ravel())
        offs.append(o.ravel())
    return np.concatenate(masks), np.concatenate(offs)


def support_patterns(axes, R, problem="hamming"):
    """Generic packed v1 support table (heterogeneous axes, any problem)."""
    n = len(axes)
    if problem == "hamming" and len(set(axes)) == 1:
        return ball_patterns(axes[0], n, R)
    ds = _delta_sets(axes, R, problem)
    maxw = _max_pattern_weight(n, R, problem)
    if maxw > 8:
        raise ValueError("support patterns cap at 8 changed positions")
    pats = [0]
    for w in range(1, maxw + 1):
        for subs in combinations(range(n), w):
            for deltas in product(*[ds[p] for p in subs]):
                if not _admissible(subs, deltas, axes, R, problem):
                    continue
                v = 0
                for k, (p, dl) in enumerate(zip(subs, deltas)):
                    v |= ((p << 4) | dl) << (8 * k)
                pats.append(v)
    return np.array(pats, dtype=np.uint64)


def support_patterns_ox(axes, R, problem="hamming"):
    """Generic ox-format support table."""
    n = len(axes)
    if problem == "hamming" and len(set(axes)) == 1:
        return ball_patterns_ox(axes[0], n, R)
    ds = _delta_sets(axes, R, problem)
    maxw = _max_pattern_weight(n, R, problem)
    masks, offs = [0], [0]
    for w in range(1, maxw + 1):
        for subs in combinations(range(n), w):
            for deltas in product(*[ds[p] for p in subs]):
                if not _admissible(subs, deltas, axes, R, problem):
                    continue
                m, o = 0, 0
                for p, dl in zip(subs, deltas):
                    m |= 1 << p
                    o |= dl << (4 * p)
                masks.append(m)
                offs.append(o)
    return (np.array(masks, dtype=np.uint16), np.array(offs, dtype=np.uint64))


def ball_size(q, n, R):
    """|B_R| = sum_{r<=R} C(n,r) (q-1)^r (homogeneous Hamming)."""
    return sum(comb(n, r) * (q - 1) ** r for r in range(R + 1))


def lee_ball_size(axes, R):
    """|B_R| under the Lee metric: coefficient sum (grades 0..R) of the
    product of per-axis weight enumerators 1 + 2z + ... + 2z^(ceil(a/2)-1)
    (+ 1*z^(a/2) for even a: the antipodal digit has multiplicity 1)."""
    coeff = [1] + [0] * R                  # z^0..z^R, truncated at R
    for a in axes:
        ka = [0] * (R + 1)
        for w in range(min(a // 2, R) + 1):
            ka[w] = 1 if (w == 0 or 2 * w == a) else 2
        coeff = [sum(coeff[i] * ka[d - i] for i in range(d + 1))
                 for d in range(R + 1)]
    return sum(coeff)

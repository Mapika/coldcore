"""Move-trace pins for the search core.

Performance work on the hot loops must not move a single decision. The
pass sequences below are *clock-free* — a fixed number of rounds, and a
polish that runs to feasibility rather than to a budget — so their
traces are a function of the seed alone, and any change in behaviour
shows up as a changed digest.

The digests were taken from the pre-optimisation search and orbit cores
(commit `1710fb7`, 2026-08-24) and are frozen. If a refactor moves them,
that refactor changed the search, whatever else it did.
"""
import hashlib
import platform

import numpy as np
import pytest

from coldcore.reference import RefCoveringBackend, RefLeeBackend
from coldcore.search import Searcher


def _digest(trace):
    h = hashlib.sha256()
    for u, code in trace:
        h.update(f"{u}|{','.join(map(str, code))};".encode())
    return h.hexdigest()[:16]


def _greedy_ruin_trace(mk, seed, rounds=16, notch=2):
    """greedy build -> peel -> a fixed ruin/recreate round sequence,
    exercising every ruin operator and the lazy-greedy repair."""
    s = Searcher(mk(), seed=seed)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    trace = [(int(s.uncov), s.code.tolist())]
    s.peel()
    trace.append((int(s.uncov), s.code.tolist()))
    target = max(1, len(s.code) - notch)
    modes = ["low", "hole", "cluster", "random", "deficit"]
    for i in range(rounds):
        s.ruin(1 + i % 4, modes[i % len(modes)])
        s.greedy_fill(target)
        trace.append((int(s.uncov), s.code.tolist()))
    return trace


def _polish_trace(mk, seed):
    """A shell-neighbourhood LAHC polish that runs *to feasibility*.

    Started from a random point set of a size that is known to admit a
    cover, so the pass terminates on ``uncov == 0`` rather than on the
    clock: the number of moves is then decided by the search alone,
    which is what makes the trace pinnable. (A polish at an infeasible
    size would burn the whole budget and its move count would just be a
    stopwatch reading.)"""
    b = mk()
    s = Searcher(b, seed=seed)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    s.peel()
    M = len(s.code)
    rng = np.random.default_rng(seed * 31 + 7)
    s.load(rng.choice(b.space.size, size=M, replace=False).astype(np.int64))
    s.anneal(M, budget_s=600.0, mode="lahc", seed=seed + 1,
             neighbourhood="shell", log_every=0, pool_refresh=64,
             restart_stale=400)
    assert s.uncov == 0, "pin must exercise a converging polish"
    return [(int(s.uncov), s.code.tolist()),
            (int(s.stats["shift_moves"]), [s.stats["shift_accepts"]])]


def _two_swap_trace(mk, seed):
    """The exact endgame DFS, bounded by nodes rather than by seconds."""
    s = Searcher(mk(), seed=seed)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    s.peel()
    before = s.code.copy()
    s.ruin(1, "low")
    s.two_swap(budget_s=600.0, max_uncov=64, nodes=300, seed=seed + 3)
    assert s.stats.get("two_swap_nodes", 0) > 0, "pin must expand nodes"
    return [(int(s.uncov), s.code.tolist()), (0, before.tolist())]


def _orbit_trace(mk, seed, spec="shift:0", rounds=12):
    """The orbit layer: partition, greedy over orbits, peel, then a
    fixed ruin/recreate round sequence. Every step is scored by an exact
    marginal, so the trace pins those marginals as well as the moves."""
    from coldcore.symmetry import SymmetricSearcher, parse_group
    b = mk()
    s = SymmetricSearcher(b, parse_group(b.space, spec), seed=seed)
    s.greedy_fill(10 ** 9)
    trace = [(int(s.uncov), s.code.tolist())]
    s.peel()
    trace.append((int(s.uncov), s.code.tolist()))
    target = max(1, len(s.code) - int(s.part.sizes[s.allowed].min()))
    modes = ["low", "random", "deficit"]
    for i in range(rounds):
        s.ruin(1 + i % 2, modes[i % len(modes)])
        s.greedy_fill(target)
        trace.append((int(s.uncov), s.code.tolist()))
    trace.append((s.stats["probes"], [s.stats["orbit_adds"],
                                      s.stats["orbit_drops"],
                                      s.stats["transforms"]]))
    return trace


#: (id, trace builder, backend factory, seed, frozen digest)
PINS = [
    ("greedy-K3(4,1)", _greedy_ruin_trace,
     lambda: RefCoveringBackend(3, 4, 1), 4, "7fe8ede9952ee50e"),
    ("greedy-K2(6,1)", _greedy_ruin_trace,
     lambda: RefCoveringBackend(2, 6, 1), 11, "57c1e4d063f33278"),
    ("greedy-lee5x5", _greedy_ruin_trace,
     lambda: RefLeeBackend([5, 5], 1), 3, "3f8e8af3d3acc2ac"),
    ("greedy-mu2", _greedy_ruin_trace,
     lambda: RefCoveringBackend(2, 6, 1, mu=2), 5, "3f4b42018740787c"),
    # The polish trace is deterministic per platform but not across
    # them: some libm/SIMD path in the LAHC shell polish differs by an
    # ULP between architectures (verified: numpy 2.2 on aarch64 still
    # reproduces the aarch64 digest; every numpy/Python combination on
    # x86-64 agrees on the x86-64 digest). Pin both.
    ("polish-K3(4,1)", _polish_trace,
     lambda: RefCoveringBackend(3, 4, 1), 8,
     {"aarch64": "982b6b50ecd86ea7", "x86_64": "507d3f201e7c61a1"}),
    ("polish-K2(6,1)", _polish_trace,
     lambda: RefCoveringBackend(2, 6, 1), 2,
     {"aarch64": "1786de18c1ee1676", "x86_64": "715cacefccefd8d3"}),
    ("twoswap-K3(4,1)", _two_swap_trace,
     lambda: RefCoveringBackend(3, 4, 1), 6, "c4d7192cb844bc5f"),
    ("orbit-K3(4,1)", _orbit_trace,
     lambda: RefCoveringBackend(3, 4, 1), 1, "0aa5be3fd8a0cf3e"),
    ("orbit-K2(6,1)", _orbit_trace,
     lambda: RefCoveringBackend(2, 6, 1), 3, "1139ba7863335ae5"),
    ("orbit-lee5x5", _orbit_trace,
     lambda: RefLeeBackend([5, 5], 1), 2, "116f6cf721452a4e"),
    ("orbit-mu2", _orbit_trace,
     lambda: RefCoveringBackend(2, 6, 1, mu=2), 4, "7630702830292425"),
]


@pytest.mark.parametrize("label,fn,mk,seed,pin", PINS,
                         ids=[p[0] for p in PINS])
def test_move_trace_is_frozen(label, fn, mk, seed, pin):
    if isinstance(pin, dict):                 # per-architecture pin
        arch = platform.machine()
        if arch not in pin:
            pytest.skip(f"no frozen digest for {arch}; "
                        "reproducibility is still checked")
        pin = pin[arch]
    assert _digest(fn(mk, seed)) == pin


@pytest.mark.parametrize("label,fn,mk,seed,pin", PINS,
                         ids=[p[0] for p in PINS])
def test_move_trace_is_reproducible(label, fn, mk, seed, pin):
    """Two runs of the same seed must agree — the property the pin
    depends on, checked separately so a broken pin is distinguishable
    from a broken seed."""
    assert _digest(fn(mk, seed)) == _digest(fn(mk, seed))

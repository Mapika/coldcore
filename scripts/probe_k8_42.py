#!/usr/bin/env python3
"""Long-running probe: how low can orbit-restricted search push K_8(4,2)?

K_8(4,2) is in {22, 23}. Exhaustive settlement of the cell is out of
reach by ~34 orders of magnitude (docs/symmetry.md quantifies it), so
this is the other half of the pincer: a search for a *witness* at M=22
inside the group-invariant subfamily.

Translations act freely on Z_8^4, so every orbit of a translation group
has size |G|, and 22 = 2 x 11 forces |G| = 2 — i.e. exactly the vectors
with 2v = 0: the 15 nonzero vectors over {0,4}. The script sweeps all of
them with a fresh seed each pass, running orbit greedy -> peel -> orbit
descent, and keeps the best code found.

    setsid nohup nice -n 19 python3 scripts/probe_k8_42.py 7200 \
        > scratch_symexact/probe.log 2>&1 &

Measured result (528 passes, 2 h, single core): **M = 32**, against
M = 35 for a free greedy+peel on the same reference backend. Better, and
nowhere near 22 — recorded as a negative result, not a promise.
"""
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coldcore.reference import RefHammingBackend          # noqa: E402
from coldcore.symmetry import SymmetricSearcher, parse_group  # noqa: E402

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 7200.0
end = time.time() + BUDGET
best = (10 ** 9, None, None)
vecs = [v for v in itertools.product((0, 4), repeat=4) if any(v)]
seed = 0
while time.time() < end:
    for v in vecs:
        if time.time() >= end:
            break
        seed += 1
        spec = "translate:" + "x".join(map(str, v))
        b = RefHammingBackend([8] * 4, 2)
        s = SymmetricSearcher(b, parse_group(b.space, spec), seed=seed,
                              deadline=min(end, time.time() + 240))
        s.greedy_fill(10 ** 9)
        if s.uncov:
            continue
        s.peel()
        code = s.descend(floor=22, notch_budget_s=20.0)
        b.recount(code)
        assert b.uncovered() == 0
        if len(code) < best[0]:
            best = (len(code), spec, np.sort(code).tolist())
            out = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "scratch_symexact")
            os.makedirs(out, exist_ok=True)
            np.savetxt(os.path.join(out, "k8_42_best.txt"),
                       b.index_word(np.sort(code)), fmt="%d")
        print(f"[{time.strftime('%H:%M:%S')}] {spec} -> M={len(code)} "
              f"(best {best[0]} via {best[1]})", flush=True)
print(f"DONE best M={best[0]} group={best[1]}", flush=True)

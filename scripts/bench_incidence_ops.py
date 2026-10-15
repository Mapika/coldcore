#!/usr/bin/env python3
"""Ruin-operator comparison on covering designs (docs/incidence.md).

For each cell and operator mix: greedy -> peel -> notch ladder to the
known optimum with that mix only (reject-if-worse, as ``Searcher.lns``),
``hole`` additionally with a shell-neighbourhood LAHC polish every 200
rejected rounds. Reports seconds to a *verified* optimum per seed.

    python scripts/bench_incidence_ops.py [deadline_s] [seeds]
"""
import sys
import time

import numpy as np

from coldcore.problems.covering_design import CoveringDesignBackend
from coldcore.search import Searcher

CELLS = [(13, 4, 2, 13), (10, 4, 3, 30)]
MIXES = [("random",), ("low",), ("cluster",), ("hole",), ("hole+shell",),
         ("deficit",), ("low", "hole", "cluster", "deficit")]


def descend_with(b, opt, modes, seed, deadline_s):
    polish = "hole+shell" in modes
    modes = ["hole" if m == "hole+shell" else m for m in modes]
    s = Searcher(b, seed=seed, deadline=time.time() + deadline_s)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    s.peel()
    t0 = time.time()
    M = len(s.code)
    while M > opt and s.time_left() > 1:
        target = M - 1
        s.ruin(len(s.code) - target, "low")
        r = 0
        while s.uncov > 0 and s.time_left() > 1:
            pre = s.uncov
            s.ruin(int(s.rng.integers(2, max(3, M // 8))), modes[r % len(modes)])
            s.greedy_fill(target)
            if s.uncov > pre and s.rng.random() > 0.1:
                s.load(s.best_code)
            else:
                s._snapshot()
            r += 1
            if polish and r % 200 == 0 and s.uncov > 0:
                s.anneal(target, 2.0, mode="lahc", neighbourhood="shell",
                         log_every=0)
        if s.uncov > 0:
            return None
        assert s.verify_full() == 0 and b.verify(s.code)
        M = len(s.code)
    return time.time() - t0


def main():
    deadline = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print("| cell | optimum | " + " | ".join("+".join(m) for m in MIXES) + " |")
    print("|---|---|" + "---|" * len(MIXES))
    for v, k, t, opt in CELLS:
        b = CoveringDesignBackend(v, k, t)
        cols = []
        for mix in MIXES:
            res = [descend_with(b, opt, list(mix), seed, deadline)
                   for seed in range(1, seeds + 1)]
            ok = [r for r in res if r is not None]
            cols.append(f"{np.median(ok):.1f} s ({len(ok)}/{seeds})" if ok
                        else f"— (0/{seeds})")
            print(f"  C({v},{k},{t}) {'+'.join(mix)}: {res}", file=sys.stderr,
                  flush=True)
        print(f"| C({v},{k},{t}) | {opt} | " + " | ".join(cols) + " |", flush=True)


if __name__ == "__main__":
    main()

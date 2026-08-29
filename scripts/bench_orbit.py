#!/usr/bin/env python3
"""Orbit-layer throughput on the bench_symmetry cells.

`scripts/bench_symmetry.py` answers "does orbit search beat free
search?"; this answers "how fast is the orbit layer itself?", which is
the question a performance change has to move. The workload is
**clock-free** — build the partition, greedy over orbits, peel, then a
fixed number of ruin/recreate rounds — so two runs differ only in
seconds, never in work done, and the comparison is exact.

    python3 scripts/bench_orbit.py
    python3 scripts/bench_orbit.py --json docs/bench_orbit.json
"""
import argparse
import json
import os
import sys
import time


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coldcore.backends import make_backend                    # noqa: E402
from coldcore.symmetry import SymmetricSearcher, parse_group   # noqa: E402

#: (label, problem, axes, R, mu, group spec). The first block is the
#: bench_symmetry cell list; the rest are larger cells with *many*
#: orbits, where the per-orbit bookkeeping — heap build, segment sums,
#: add/drop code splicing — is what is actually being timed rather than
#: the backend underneath it.
CELLS = [
    ("K2(6,1)", "hamming", [2] * 6, 1, 1, "translate:111111"),
    ("K3(4,1)", "hamming", [3] * 4, 1, 1, "translate:1011"),
    ("K3(5,1)", "hamming", [3] * 5, 1, 1, "translate:10111"),
    ("K2^2(6,1)", "hamming", [2] * 6, 1, 2,
     "translate:010100+translate:111011"),
    ("K2^3(8,1)", "hamming", [2] * 8, 1, 3,
     "translate:10000111+translate:01000110"),
    ("K4^2(4,1)", "hamming", [4] * 4, 1, 2,
     "translate:2x0x1x3+translate:1x3x0x3"),
    ("lee5x5R1", "lee", [5, 5], 1, 1, "translate:1x2"),
    ("king9x9", "torus_linf", [9, 9], 1, 1, "translate:3x1"),
    ("K4(4,1)", "hamming", [4] * 4, 1, 1, "translate:0x1x1x3"),
    ("torus11x11", "torus_linf", [11, 11], 1, 1, "translate:10x2"),
    # --- many orbits: the orbit layer's own cost ---
    ("K3(8,2)/shift", "hamming", [3] * 8, 2, 1, "shift:0"),
    ("K3(8,2)/diag", "hamming", [3] * 8, 2, 1, "diag"),
    ("K2(14,3)/shift", "hamming", [2] * 14, 3, 1, "shift:0"),
    ("K5(6,2)/shift", "hamming", [5] * 6, 2, 1, "shift:0"),
    ("torus27x27/tr", "torus_linf", [27, 27], 1, 1, "translate:9x3"),
    ("lee11x11x11/tr", "lee", [11] * 3, 2, 1, "translate:1x3x9"),
]


def workload(kind, problem, axes, R, mu, spec, seed, rounds):
    """Partition + greedy + peel + a fixed ruin/recreate round sequence."""
    b = make_backend(kind, axes, R, problem=problem, mu=mu)
    t0 = time.perf_counter()
    G = parse_group(b.space, spec)
    s = SymmetricSearcher(b, G, seed=seed)
    t_part = time.perf_counter() - t0
    t1 = time.perf_counter()
    s.greedy_fill(10 ** 9)
    t_greedy = time.perf_counter() - t1
    t2 = time.perf_counter()
    s.peel()
    t_peel = time.perf_counter() - t2
    t3 = time.perf_counter()
    target = max(1, len(s.code) - int(s.part.sizes[s.allowed].min()))
    modes = ["low", "random", "deficit"]
    for i in range(rounds):
        s.ruin(1 + i % 2, modes[i % len(modes)])
        s.greedy_fill(target)
    t_rounds = time.perf_counter() - t3
    return {"orbits": len(s.part), "M": len(s.code), "uncov": int(s.uncov),
            "part_s": t_part, "greedy_s": t_greedy, "peel_s": t_peel,
            "rounds_s": t_rounds, "total_s": time.perf_counter() - t0,
            "probes": s.stats["probes"], "adds": s.stats["orbit_adds"],
            "drops": s.stats["orbit_drops"],
            "transforms": s.stats["transforms"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="cpu",
                    choices=["ref", "cpu", "gpu"])
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only")
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    recs = []
    print(f"{'cell':16s} {'orbits':>7s} {'M':>5s} {'part ms':>9s} "
          f"{'greedy ms':>10s} {'peel ms':>9s} {'rounds ms':>10s} "
          f"{'total ms':>10s} {'probes':>7s}")
    print("-" * 92)
    for label, problem, axes, R, mu, spec in CELLS:
        if args.only and args.only not in label:
            continue
        best = None
        for _ in range(args.repeat):
            r = workload(args.backend, problem, axes, R, mu, spec,
                         args.seed, args.rounds)
            if best is None or r["total_s"] < best["total_s"]:
                best = r
        best["cell"] = label
        best["group"] = spec
        best["backend"] = args.backend
        recs.append(best)
        print(f"{label:16s} {best['orbits']:7d} {best['M']:5d} "
              f"{best['part_s'] * 1e3:9.2f} {best['greedy_s'] * 1e3:10.2f} "
              f"{best['peel_s'] * 1e3:9.2f} {best['rounds_s'] * 1e3:10.2f} "
              f"{best['total_s'] * 1e3:10.2f} {best['probes']:7d}",
              flush=True)
    tot = sum(r["total_s"] for r in recs)
    print(f"\nsum of best-of-{args.repeat} totals: {tot * 1e3:.1f} ms "
          f"over {len(recs)} cells, {args.rounds} rounds each")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rounds": args.rounds, "repeat": args.repeat,
                       "seed": args.seed, "backend": args.backend,
                       "records": recs}, fh, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU backend head-to-head: brute-force reference vs the axis-DP backend.

Two measurements, both paired (same cell, same seed, same machine):

1. **transform** — one coverage transform (`recount`) and one gain
   transform (`gain_map`) at a fixed solution size, plus one batched
   `ball_gather`. This is where the asymptotics live: brute force is
   O(|X|^2 n), the axis DP is O(n |X| (R+1)).
2. **end to end** — greedy build -> peel -> notch descent to a floor,
   run to completion on both backends with the same seed. The DP
   backend is a drop-in with byte-identical fields, so the *result* is
   expected to match; only the clock should differ.

Cells above the reference's 2^16 word cap are run on the DP backend
alone and reported as "brute force: impossible" — which is the point.

    python3 scripts/bench_backends.py
    python3 scripts/bench_backends.py --json docs/bench_backends.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coldcore.backends import make_backend            # noqa: E402
from coldcore.search import Searcher                  # noqa: E402

#: (label, problem, axes, R, mu, descent floor). The first block is
#: inside the reference's reach so the comparison is paired; the second
#: is what the reference cannot do at all.
CELLS = [
    ("K2(12,2)", "hamming", [2] * 12, 2, 1, 24),
    ("K3(8,2)", "hamming", [3] * 8, 2, 1, 80),
    ("K2(16,3)", "hamming", [2] * 16, 3, 1, 350),
    ("lee9x9x9R2", "lee", [9] * 3, 2, 1, 36),
    ("king15x15", "torus_linf", [15, 15], 1, 1, 25),
    ("K2^2(12,2)", "hamming", [2] * 12, 2, 2, 48),
    # --- out of brute force's reach ---
    ("K3(12,3)", "hamming", [3] * 12, 3, 1, 900),
    ("K5(8,3)", "hamming", [5] * 8, 3, 1, 900),
    ("K7(8,3)", "hamming", [7] * 8, 3, 1, 4000),
    ("torus64x64x64R2", "torus_linf", [64] * 3, 2, 1, 3000),
]

REF_CAP = 1 << 16


def _time(fn, repeat=3):
    best = None
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        best = dt if best is None else min(best, dt)
    return best


def transform_bench(kind, problem, axes, R, mu, seed=1, repeat=3):
    b = make_backend(kind, axes, R, problem=problem, mu=mu)
    size = b.space.size
    rng = np.random.default_rng(seed)
    m = max(4, size // 200)
    code = np.unique(rng.integers(0, size, size=m).astype(np.int64))
    probe = rng.integers(0, size, size=min(256, size)).astype(np.int64)
    out = {"size": size, "M": len(code)}
    out["recount_s"] = _time(lambda: b.recount(code), repeat)
    b.recount(code)
    out["gain_s"] = _time(b.gain_map, repeat)
    out["gather256_s"] = _time(lambda: b.ball_gather(probe, 0), repeat)
    b.close()
    return out


def solve_bench(kind, problem, axes, R, mu, floor, seed=1, budget=120.0):
    b = make_backend(kind, axes, R, problem=problem, mu=mu)
    # Searcher.time_left() reads time.time(); the deadline must be on the
    # same clock (a perf_counter deadline reads as "already expired" and
    # every pass exits on entry).
    t0 = time.perf_counter()
    s = Searcher(b, seed=seed, deadline=time.time() + budget)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    t_greedy = time.perf_counter() - t0
    # a backend too slow to even finish the greedy build inside the
    # budget is reported as such — that is the measurement, not an error
    done = s.uncov == 0
    m_after_peel = None
    if done:
        s.peel()
        m_after_peel = len(s.code)
        s.descend(floor=floor, notch_budget_s=max(1.0, budget / 10.0))
    dt = time.perf_counter() - t0
    out = {"greedy_s": t_greedy, "total_s": dt, "M_peel": m_after_peel,
           "M": len(s.code), "uncov": int(s.uncov), "feasible": bool(done),
           "transforms": s.stats["transforms"]}
    b.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=60.0,
                    help="wall-clock cap for each end-to-end solve")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--only", help="substring filter on the cell label")
    ap.add_argument("--skip-solve", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    recs = []
    for label, problem, axes, R, mu, floor in CELLS:
        if args.only and args.only not in label:
            continue
        size = int(np.prod(axes))
        kinds = ["cpu"] if size > REF_CAP else ["ref", "cpu"]
        rec = {"cell": label, "problem": problem, "axes": axes, "R": R,
               "mu": mu, "size": size, "floor": floor, "transform": {},
               "solve": {}}
        for kind in kinds:
            rec["transform"][kind] = transform_bench(kind, problem, axes, R,
                                                     mu, repeat=args.repeat)
            if not args.skip_solve:
                rec["solve"][kind] = solve_bench(kind, problem, axes, R, mu,
                                                 floor, budget=args.budget)
        t = rec["transform"]
        sp = (t["ref"]["recount_s"] / t["cpu"]["recount_s"]
              if "ref" in t else None)
        recs.append(rec)
        print(f"{label:16s} |X|={size:>10,}  cpu recount "
              f"{t['cpu']['recount_s'] * 1e3:8.2f} ms  gain "
              f"{t['cpu']['gain_s'] * 1e3:8.2f} ms" +
              (f"  vs ref {t['ref']['recount_s'] * 1e3:9.2f} ms "
               f"({sp:.0f}x)" if sp else "   (brute force: impossible)"),
              flush=True)

    print(f"\n{'cell':16s} {'|X|':>10s} {'recount ref':>13s} "
          f"{'recount cpu':>13s} {'gain ref':>13s} {'gain cpu':>13s} "
          f"{'x':>6s} {'solve ref':>11s} {'solve cpu':>11s} "
          f"{'M ref':>6s} {'M cpu':>6s}")
    print("-" * 130)
    for r in recs:
        t, s = r["transform"], r["solve"]
        def ms(kind, key):
            return (f"{t[kind][key] * 1e3:10.2f} ms" if kind in t
                    else "         --")
        rt, ct = ms("ref", "recount_s"), ms("cpu", "recount_s")
        rg, cg = ms("ref", "gain_s"), ms("cpu", "gain_s")
        sp = (f"{t['ref']['gain_s'] / t['cpu']['gain_s']:5.0f}x"
              if "ref" in t else "     -")
        rs = f"{s['ref']['total_s']:10.2f}s" if "ref" in s else "         --"
        cs = f"{s['cpu']['total_s']:10.2f}s" if "cpu" in s else "         --"
        rm = (f"{s['ref']['M']:6d}" if s.get("ref", {}).get("feasible")
              else "  none")
        cm = (f"{s['cpu']['M']:6d}" if s.get("cpu", {}).get("feasible")
              else "  none")
        print(f"{r['cell']:16s} {r['size']:10,} {rt} {ct} {rg} {cg} {sp} "
              f"{rs} {cs} {rm} {cm}")
    print("\n'--'   = the brute-force reference refuses the cell "
          "(|X| > 2^16).\n'none' = no feasible cover inside the budget "
          "(the greedy build alone did not finish).")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"budget": args.budget, "repeat": args.repeat,
                       "records": recs}, fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

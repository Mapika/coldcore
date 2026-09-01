#!/usr/bin/env python3
"""Head-to-head: free (word-level) search vs orbit-restricted search.

The claim the symmetry layer makes is narrow and testable: *when the
optimum of a cell is invariant under a group G, searching over G-orbits
finds it faster than searching over words — and when it is not, orbit
search is strictly worse.* This script measures both halves on the same
machine, the same backend and the same wall-clock budget.

Both methods run the same three passes — greedy build, peel, notch
descent — over different atoms:

1. **free** — `Searcher`, atoms = words.
2. **orbit** — `SymmetricSearcher`, atoms = G-orbits.

The measured quantity is **time to first verified solution at the target
size M-star*** (each method stops the moment it produces one; a
from-scratch recount confirms it), capped at ``--budget`` seconds.
"Reached the same M eventually" is the *uninteresting* comparison and
mostly comes out a tie — the whole claim is about how long it takes.

    python3 scripts/bench_symmetry.py --budget 60 --seeds 3
    python3 scripts/bench_symmetry.py --json sym.json

Everything is the pure-numpy reference backend, so it runs with no GPU
and no plugin build.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coldcore.reference import (RefHammingBackend, RefLeeBackend,  # noqa: E402
                                RefTorusBackend)
from coldcore.search import Searcher                               # noqa: E402
from coldcore.symmetry import (SymmetricSearcher,                  # noqa: E402
                               parse_group)

#: (label, backend factory, target M*, group spec, note). The groups are
#: the ones a constructor would reach for: translations by a codeword or
#: by a small linear code, i.e. "search over unions of cosets". M* is the
#: best size either method has ever produced on the cell (the literature
#: optimum where one is known).
CELLS = [
    ("K2(6,1)", 12, lambda: RefHammingBackend([2] * 6, 1),
     "translate:111111", "complementation x -> x+1"),
    ("K3(4,1)", 9, lambda: RefHammingBackend([3] * 4, 1),
     "translate:1011", "one generator of the ternary Hamming code"),
    ("K3(5,1)", 27, lambda: RefHammingBackend([3] * 5, 1),
     "translate:10111", "a weight-4 ternary translation"),
    ("K2^2(6,1)", 20, lambda: RefHammingBackend([2] * 6, 1, mu=2),
     "translate:010100+translate:111011", "order-4 translation group"),
    ("K2^3(8,1)", 96, lambda: RefHammingBackend([2] * 8, 1, mu=3),
     "translate:10000111+translate:01000110", "order-4 translation group"),
    ("K2^4(8,1)", 128, lambda: RefHammingBackend([2] * 8, 1, mu=4),
     "translate:10000111+translate:00000110", "order-4 translation group"),
    ("K4^2(4,1)", 48, lambda: RefHammingBackend([4] * 4, 1, mu=2),
     "translate:2x0x1x3+translate:1x3x0x3", "order-16 translation group"),
    ("K2^2(3,1)", 4, lambda: RefHammingBackend([2] * 3, 1, mu=2),
     "translate:111", "cosets of the repetition code"),
    ("lee5x5R1", 5, lambda: RefLeeBackend([5, 5], 1),
     "translate:1x2", "Golomb-Welch line {(i,2i)}"),
    ("lee5x5R1mu2", 10, lambda: RefLeeBackend([5, 5], 1, mu=2),
     "translate:1x2", "two Golomb-Welch lines"),
    ("king9x9", 9, lambda: RefTorusBackend([9, 9], 1),
     "translate:3x1", "perfect 9-point lattice"),
    # --- the honest half: cells whose optimum is NOT group-invariant ---
    ("K4(4,1)", 24, lambda: RefHammingBackend([4] * 4, 1),
     "translate:0x1x1x3", "no invariant optimum: orbit search loses"),
    ("lee7x7R1", 12, lambda: RefLeeBackend([7, 7], 1),
     "translate:1x2", "7 does not divide the optimum: orbit search loses"),
    ("torus11x11", 16, lambda: RefTorusBackend([11, 11], 1),
     "translate:10x2", "11 does not divide the optimum: orbit search loses"),
]


def _stopwatch(deadline_holder, target, t0):
    """on_solved hook: record the first hit at <= target and stop the run."""
    hit = {"t": None, "M": None}

    def on_solved(searcher):
        def cb(M, idx):
            if hit["M"] is None or M < hit["M"]:
                hit["M"] = int(M)
            if M <= target and hit["t"] is None:
                hit["t"] = time.time() - t0
                searcher.deadline = time.time()      # stop the descent now
        return cb
    return hit, on_solved


#: `Searcher.descend` reserves this much of its deadline before starting
#: a notch (it will not begin one it cannot finish), so a free run has to
#: be given the working budget PLUS that reserve or it silently never
#: descends at all. The orbit descent has no such reserve. Getting this
#: wrong is the difference between "free search never reaches 12 on
#: K_2(6,1)" and "free search reaches it in 7.6s" — it was wrong here
#: once, and the tables had to be redone.
DESCEND_RESERVE = 30.0


def free_run(mk, target, budget, seed):
    """Words as atoms: greedy -> peel -> notch descent."""
    b = mk()
    t0 = time.time()
    s = Searcher(b, seed=seed, deadline=t0 + budget + DESCEND_RESERVE)
    hit, mkcb = _stopwatch(None, target, t0)
    s.on_solved = mkcb(s)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    s.peel()
    if len(s.code) <= target:
        s.on_solved(len(s.code), s.code)
    else:
        # notch budget capped at 20s: descend refuses a notch shorter
        # than min(20, notch_budget_s), so a larger value would make the
        # tail of the run unusable
        s.descend(floor=target,
                  notch_budget_s=min(20.0, max(1.0, budget / 8.0)))
    return hit, None


def orbit_run(mk, target, spec, budget, seed):
    """Orbits as atoms: greedy -> orbit peel -> orbit notch descent."""
    b = mk()
    G = parse_group(b.space, spec)
    t0 = time.time()
    s = SymmetricSearcher(b, G, seed=seed, deadline=time.time() + budget)
    hit, mkcb = _stopwatch(None, target, t0)
    s.on_solved = mkcb(s)
    s.greedy_fill(10 ** 9)
    if s.uncov == 0:
        s.peel()
        s.descend(floor=target, notch_budget_s=max(0.5, budget / 8.0))
    return hit, len(s.part)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=30.0,
                    help="wall-clock cap per (cell, method, seed)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--only", help="substring filter on the cell label")
    ap.add_argument("--json", help="write raw records here")
    args = ap.parse_args(argv)

    rows, recs = [], []
    for label, target, mk, spec, note in CELLS:
        if args.only and args.only not in label:
            continue
        ft, ot, fm, om, T = [], [], [], [], None
        for seed in range(1, args.seeds + 1):
            h, _ = free_run(mk, target, args.budget, seed)
            ft.append(h["t"])
            fm.append(h["M"])
            h, T = orbit_run(mk, target, spec, args.budget, seed)
            ot.append(h["t"])
            om.append(h["M"])
        size = mk().space.size

        def agg(ts):
            """Mean time over the seeds that hit the target, and how many
            did. A method that hits on 2 of 5 seeds is reported as such:
            hiding the misses inside an average would be dishonest."""
            ok = [t for t in ts if t is not None]
            return (max(round(sum(ok) / len(ok), 4), 1e-4)
                    if ok else None), len(ok)
        f_t, f_ok = agg(ft)
        o_t, o_ok = agg(ot)
        speed = (round(f_t / o_t, 1)
                 if (f_t is not None and o_t is not None) else None)
        rows.append((label, size, T, target, f_t, f_ok, o_t, o_ok, speed,
                     min(x for x in fm if x), min(x for x in om if x),
                     args.seeds, spec))
        recs.append({"cell": label, "space": size, "orbits": T,
                     "target": target, "group": spec, "note": note,
                     "free_s": ft, "orbit_s": ot,
                     "free_bestM": fm, "orbit_bestM": om})
        fstr = f"{f_t:8.4f}s" if f_t is not None else "  TIMEOUT"
        ostr = f"{o_t:8.4f}s" if o_t is not None else "  TIMEOUT"
        print(f"{label:12s} target={target:4d}  free {fstr} "
              f"({f_ok}/{args.seeds})   orbit {ostr} ({o_ok}/{args.seeds})"
              f"   speedup {speed if speed is not None else '-'}", flush=True)

    print(f"\n{'cell':12s} {'|X|':>6s} {'orb':>5s} {'M*':>5s} "
          f"{'free s':>10s} {'hit':>5s} {'orbit s':>10s} {'hit':>5s} "
          f"{'speedup':>9s}  group")
    print("-" * 108)
    for (label, size, T, target, f_t, f_ok, o_t, o_ok, speed, fbest, obest,
         nseeds, spec) in rows:
        fs = f"{f_t:10.4f}" if f_t is not None else f"{'>' + str(fbest):>10s}"
        os_ = (f"{o_t:10.4f}" if o_t is not None
               else f"{'>' + str(obest):>10s}")
        if speed is not None:
            sp = f"{speed:8.1f}x"
        elif o_t is not None:
            sp = f"  >{args.budget / max(o_t, 1e-4):.0f}x"
        else:
            sp = "     LOSS"
        print(f"{label:12s} {size:6d} {T:5d} {target:5d} {fs} "
              f"{f_ok}/{nseeds:<3d} {os_} {o_ok}/{nseeds:<3d} {sp}  {spec}")
    print("\n'hit' = seeds that produced a verified solution at M* inside "
          "the budget;\n'>M'  = never reached M*, M is the smallest size "
          "that method did verify.")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"budget": args.budget, "seeds": args.seeds,
                       "records": recs}, fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

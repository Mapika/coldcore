#!/usr/bin/env python3
"""Device-side cost of one polish move, as the search issues it today and
through the v0.3 ``swap_eval`` operation.

A polish move drops an incumbent ``w`` and costs a batch of candidates:

  sequence  : ball_gather([w], 1); ball_update([w], -1);
              ball_gather(cand, 0); ball_update([w], +1)      (rejected move)
  swap_eval : swap_eval([w], [cand])                           (read-only)
  accept    : the state change of an accepted move, ball_update([w], -1) +
              ball_update([v], +1), and its v0.3 checked form with
              ball_update_delta (exact deficit change, no scan) vs the
              ``uncovered()`` scan the polish runs after every accept.

The state is a saved cover with ``--notch`` words removed (a polish
state); candidates are random words, ``--batch`` per move (the shell
polish averages ~95-125 on the profiled cells). Both forms return the
same numbers (checked on every iteration).

    python3 scripts/bench_move.py --cell 'K8(9,4)' --load-code s.npy
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coldcore.backends import backend_for_spec        # noqa: E402


def timed(fn, n, cap_s=8.0):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
        if len(ts) >= 5 and sum(ts) > cap_s:
            break
    ts = np.array(ts) * 1e6
    return {"median_us": float(np.median(ts)), "mean_us": float(ts.mean()),
            "n": len(ts)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--load-code", required=True)
    ap.add_argument("--notch", type=int, default=2)
    ap.add_argument("--batch", type=int, default=110)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    b = backend_for_spec("gpu", args.cell)
    rng = np.random.default_rng(args.seed)
    code = np.load(args.load_code)
    code = code[:max(1, len(code) - args.notch)]
    b.recount(code)
    size = b.space.size
    w = np.array([int(code[len(code) // 2])], dtype=np.int64)
    cand = rng.integers(0, size, size=args.batch).astype(np.int64)
    v = cand[:1].copy()
    res = {"cell": args.cell, "M": len(code), "uncov": b.uncovered(),
           "batch": args.batch}

    def seq():
        lw = b.ball_gather(w, 1)
        b.ball_update(w, -1)
        g = b.ball_gather(cand, 0)
        b.ball_update(w, +1)
        return lw, g

    lw0, g0 = seq()
    res["sequence"] = timed(seq, args.reps)
    sw = getattr(b, "swap_eval", None)
    if sw is not None:
        lo, g1 = sw(w, [cand])
        assert int(lo[0]) == int(lw0[0]) and (g1[0] == g0).all()
        res["swap_eval"] = timed(lambda: sw(w, [cand]), args.reps)

    def accept_scan():
        b.ball_update(w, -1)
        b.ball_update(v, +1)
        u = b.uncovered()
        b.ball_update(v, -1)
        b.ball_update(w, +1)
        return u
    res["accept+uncovered (x2: do+undo)"] = timed(accept_scan, args.reps)
    ud = getattr(b, "ball_update_delta", None)
    if ud is not None:
        def accept_delta():
            d = ud(w, -1) + ud(v, +1)
            ud(v, -1)
            ud(w, +1)
            return d
        u0 = b.uncovered()
        assert u0 + accept_delta() == accept_scan()
        res["accept+update_delta (x2: do+undo)"] = timed(accept_delta,
                                                         args.reps)
    res["stats"] = b.gpu_stats() if hasattr(b, "gpu_stats") else {}
    b.close()
    for k, x in res.items():
        if isinstance(x, dict) and "median_us" in x:
            print(f"{args.cell} {k:<36}{x['median_us']:>12.1f} us median "
                  f"(n={x['n']})")
    print(f"{args.cell} M={res['M']} uncov={res['uncov']} "
          f"batch={args.batch} stats={res['stats']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Whole-space transform throughput on one cell.

Times ``recount`` (code -> cnt), ``gain_map`` and ``loss_map`` (median of
``--reps`` calls, after one warm-up) and reports the effective bandwidth
against the v0.2 traffic model: per axis pass the legacy kernel reads
R+1 int32 planes and writes R (one on the last pass), plus R layer fills
and the indicator init. The number is the same model for every build, so
it is a speed ratio, not a claim about the bytes a given build moves.

With ``DCT_TIMING=1`` in the environment the plugin also prints each
kernel's time to stderr.

    python3 scripts/bench_transform.py --cell 'K8(9,4)' --load-code s.npy
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coldcore.backends import backend_for_spec, parse_cell   # noqa: E402


def legacy_bytes(axes, R):
    size = int(np.prod(axes))
    L = R + 1
    n = len(axes)
    planes = R + 1.5            # fills (R) + init (read cnt 0.5, write 1)
    planes += (n - 1) * (L + R) + (L + 1)
    return planes * 4 * size


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--load-code", default=None)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    problem, axes, R, mu = parse_cell(args.cell)
    b = backend_for_spec("gpu", args.cell)
    rng = np.random.default_rng(1)
    if args.load_code:
        code = np.load(args.load_code)
        code = code[:max(1, len(code) - 2)]
    else:
        code = np.sort(rng.choice(b.space.size, size=100, replace=False))
    out = {"cell": args.cell, "size": b.space.size}
    model = legacy_bytes(axes, R)
    for name, fn in (("recount", lambda: b.recount(code)),
                     ("gain_map", b.gain_map), ("loss_map", b.loss_map)):
        fn()
        ts = []
        for _ in range(args.reps):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
            if len(ts) >= 3 and sum(ts) > 20:
                break
        med = float(np.median(ts))
        out[name] = {"median_ms": 1e3 * med, "n": len(ts),
                     "model_GBps": model / med / 1e9}
        print(f"{args.cell} {name:9s} {1e3 * med:10.3f} ms  "
              f"(v0.2-model {model / med / 1e9:7.1f} GB/s, n={len(ts)})",
              flush=True)
    b.close()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

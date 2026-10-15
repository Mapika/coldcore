#!/usr/bin/env python3
"""Per-operation GPU profile of the search inner loop.

Three parts, all on one cell, one seed:

1. **build** — exact lazy greedy from the empty code until covered,
   then ``peel``; wall time and per-operation breakdown.
2. **polish** — the shell-neighbourhood LAHC polish (the campaigns'
   inner loop) for a fixed wall budget after notching ``--notch`` words
   off the built cover; moves/second and the per-operation breakdown.
3. **micro** — per-call latency of every backend operation the search
   uses, at the polished state (median / mean of ``--reps`` calls).

Every backend call is synchronous (the binding returns after the
device has finished and the result is on the host), so wall time
around a call is the call's cost. ``other`` is everything that is not
inside a backend call: the search core's Python and numpy.

    python3 scripts/profile_gpu_ops.py --cell 'K5(8,3)' --polish-s 10
    python3 scripts/profile_gpu_ops.py --cell 'K8(9,4)' --json out.json

Hold the shared-GPU lock around it when the numbers are to be reported.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coldcore.backends import backend_for_spec        # noqa: E402
from coldcore.search import Searcher                  # noqa: E402

OPS = ("recount", "gain_map", "loss_map", "ball_update", "ball_gather",
       "swap_eval", "ball_update_delta",
       "count_eq", "uncovered", "map_max", "map_hist", "map_extract",
       "map_read_at", "read_cnt")


class OpTimer:
    """Wraps the backend's methods; accumulates calls, seconds, items."""

    def __init__(self, b):
        self.b = b
        self.reset()
        for name in OPS:
            orig = getattr(b, name, None)
            if orig is None:
                continue
            setattr(b, name, self._wrap(name, orig))

    def reset(self):
        self.calls = defaultdict(int)
        self.secs = defaultdict(float)
        self.items = defaultdict(int)

    def _wrap(self, name, orig):
        def f(*a, **kw):
            t = time.perf_counter()
            r = orig(*a, **kw)
            self.secs[name] += time.perf_counter() - t
            self.calls[name] += 1
            if a and name in ("ball_update", "ball_gather", "map_read_at"):
                self.items[name] += int(np.size(a[0]))
            return r
        return f

    def table(self, wall):
        rows = []
        inside = sum(self.secs.values())
        for name in sorted(self.secs, key=lambda k: -self.secs[k]):
            c, s = self.calls[name], self.secs[name]
            rows.append({"op": name, "calls": c, "s": s,
                         "us_per_call": 1e6 * s / max(c, 1),
                         "items_per_call": self.items[name] / max(c, 1),
                         "share": s / max(wall, 1e-12)})
        rows.append({"op": "other", "calls": 0, "s": wall - inside,
                     "us_per_call": 0.0, "items_per_call": 0.0,
                     "share": (wall - inside) / max(wall, 1e-12)})
        return rows


def show(title, rows, wall):
    print(f"\n-- {title}: {wall:.3f} s wall")
    print(f"{'op':<13}{'calls':>9}{'total s':>10}{'us/call':>11}"
          f"{'items/call':>12}{'share':>8}")
    for r in rows:
        print(f"{r['op']:<13}{r['calls']:>9}{r['s']:>10.3f}"
              f"{r['us_per_call']:>11.1f}{r['items_per_call']:>12.1f}"
              f"{100 * r['share']:>7.1f}%")


def micro(b, code, reps, rng):
    """Per-call latency of each operation at the current state."""
    out = {}

    def bench(name, fn, n=reps):
        fn()                                  # warm
        ts = []
        for _ in range(n):                    # at most ~3 s per entry
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
            if len(ts) >= 3 and sum(ts) > 3.0:
                break
        ts = np.array(ts) * 1e6
        out[name] = {"median_us": float(np.median(ts)),
                     "mean_us": float(ts.mean()), "n": len(ts)}

    size = b.space.size
    w = np.array([int(code[0])], dtype=np.int64)

    def upd_pair():
        b.ball_update(w, -1)
        b.ball_update(w, +1)

    def upd_pair_sync():                  # + a one-word read that waits
        b.ball_update(w, -1)
        b.ball_update(w, +1)
        b.map_read_at(w)
    # v0.3 updates return before the device finishes: "issue" is the host
    # cost, "+ sync" the completed cost (minus one map_read_at(1 word))
    bench("map_read_at(1 word)", lambda: b.map_read_at(w))
    bench("ball_update x2 (1 word, -1/+1)", upd_pair)
    bench("ball_update x2 + map_read_at(1)", upd_pair_sync)
    bench("ball_gather role1 (1 word)", lambda: b.ball_gather(w, 1))
    # repeated gathers below see an unchanged cnt (v0.3 then reuses its
    # deficient-word list); the polish changes cnt before every gather:
    c64 = rng.integers(0, size, size=64).astype(np.int64)

    def upd_gather():
        upd_pair()
        b.ball_gather(c64, 0)

    def upd_uncov():
        upd_pair()
        b.uncovered()
    bench("ball_update x2 + gather role0 (64)", upd_gather,
          n=max(5, reps // 4))
    bench("ball_update x2 + uncovered", upd_uncov)
    for k in (64, 256, 1024):
        cand = rng.integers(0, size, size=k).astype(np.int64)
        bench(f"ball_gather role0 ({k} words)",
              lambda c=cand: b.ball_gather(c, 0), n=max(5, reps // 4))
    bench("uncovered", b.uncovered)
    nt = max(3, min(reps // 10, 20))
    bench("gain_map (transform)", b.gain_map, n=nt)
    mx = b.map_max()
    bench("map_max", b.map_max)
    if getattr(b, "supports_hist", True) and mx > 0:
        bench("map_hist(1024)", lambda: b.map_hist(1024, mx))
    bench("map_extract(thr=max)", lambda: b.map_extract(max(1, mx)))
    bench("map_extract(thr=max/2, cap 4096)",
          lambda: b.map_extract(max(1, mx // 2), cap=4096))
    bench("map_extract(thr=1, cap 2^18)",
          lambda: b.map_extract(1, cap=1 << 18), n=nt)
    bench("loss_map (transform)", b.loss_map, n=nt)
    bench("map_read_at(code)", lambda: b.map_read_at(code))
    bench("recount (transform)", lambda: b.recount(code), n=nt)
    return out


def gpu_util():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10).stdout.strip()
    except Exception:            # noqa: BLE001
        return "?"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="gpu", choices=["ref", "cpu", "gpu"])
    ap.add_argument("--cell", default="K5(8,3)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--notch", type=int, default=2)
    ap.add_argument("--polish-s", type=float, default=10.0)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--no-micro", action="store_true")
    ap.add_argument("--no-peel", action="store_true")
    ap.add_argument("--build-deadline", type=float, default=None,
                    help="stop the greedy build after about this many "
                         "seconds (the search stops 5 s before a deadline); "
                         "compare placements per second, not M")
    ap.add_argument("--json", default=None)
    ap.add_argument("--load-code", default=None,
                    help="start from a saved code (.npy) instead of building")
    ap.add_argument("--save-code", default=None,
                    help="save the built code (.npy) for later --load-code")
    args = ap.parse_args(argv)

    res = {"cell": args.cell, "backend": args.backend, "seed": args.seed,
           "gpu_before": gpu_util(),
           "plugin": os.environ.get("COLDCORE_PLUGIN_PATH", "default")}
    b = backend_for_spec(args.backend, args.cell)
    T = OpTimer(b)
    s = Searcher(b, seed=args.seed, deadline=(
        time.time() + args.build_deadline if args.build_deadline else None))
    print(f"cell {args.cell} |X|={b.space.size:,} backend={args.backend} "
          f"gpu-util {res['gpu_before']}", flush=True)

    if args.load_code:
        s.load(np.load(args.load_code))
        print(f"loaded {args.load_code}: M={len(s.code)} uncov={s.uncov}")
    else:
        t0 = time.perf_counter()
        s.load(np.zeros(0, dtype=np.int64))
        s.greedy_fill(10 ** 9)
        t_greedy = time.perf_counter() - t0
        m_greedy = len(s.code)
        if not args.no_peel:
            s.peel()
        wall = time.perf_counter() - t0
        res["build"] = {"wall_s": wall, "greedy_s": t_greedy,
                        "M_greedy": m_greedy, "M_peel": len(s.code),
                        "uncov": int(s.uncov), "ops": T.table(wall),
                        "placements": s.stats["placements"]}
        print(f"build: greedy {t_greedy:.3f} s -> M={m_greedy}, "
              f"peel -> M={len(s.code)}, uncov={s.uncov}, "
              f"{s.stats['placements'] / max(t_greedy, 1e-9):.1f} "
              f"placements/s")
        show("build (greedy + peel)", res["build"]["ops"], wall)
        if args.save_code:
            np.save(args.save_code, np.asarray(s.code, dtype=np.int64))

    # the micro benchmark runs on this state (not the post-polish one), so
    # it is the same state for every build given the same --load-code
    code0 = np.asarray(s.code, dtype=np.int64).copy()
    if args.polish_s > 0:
        target = max(1, len(s.code) - args.notch)
        s.ruin(len(s.code) - target, "low")
        T.reset()
        t0 = time.perf_counter()
        s.anneal(target, budget_s=args.polish_s, mode="lahc", seed=args.seed,
                 neighbourhood="shell", log_every=0, pool_refresh=64)
        wall = time.perf_counter() - t0
        mv = s.stats.get("shift_moves", 0)
        res["polish"] = {"wall_s": wall, "budget_s": args.polish_s,
                         "moves": mv, "mv_per_s": mv / max(wall, 1e-9),
                         "accepts": s.stats.get("shift_accepts", 0),
                         "M": target, "uncov": int(s.uncov),
                         "ops": T.table(wall)}
        print(f"\npolish: {mv} moves in {wall:.2f} s = "
              f"{mv / max(wall, 1e-9):.1f} mv/s (M={target}, "
              f"uncov={s.uncov})")
        show("polish", res["polish"]["ops"], wall)

    if not args.no_micro:
        rng = np.random.default_rng(args.seed)
        code = code0
        # drop a few words so the gain map is not identically zero
        code = code[:max(1, len(code) - max(1, args.notch))]
        b.recount(code)
        res["micro_state"] = {"M": len(code), "uncov": b.uncovered()}
        res["micro"] = micro(b, code, args.reps, rng)
        print(f"\n-- micro (per call) at M={len(code)} "
              f"uncov={res['micro_state']['uncov']}")
        for k, v in res["micro"].items():
            print(f"  {k:<36}{v['median_us']:>12.1f} us median"
                  f"{v['mean_us']:>12.1f} us mean  (n={v['n']})")
    res["gpu_after"] = gpu_util()
    b.close()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

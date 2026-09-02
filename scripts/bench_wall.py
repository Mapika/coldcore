#!/usr/bin/env python3
"""Wall-breaking benchmark: which endgame pass mix actually gets through.

A *wall* is the state this framework spends most of its record-hunting
time in: a cover of size M+1 is in hand, a cover of size M is believed
to exist, and every configuration sits at 1-3 uncovered words for hours.
This script measures the passes that claim to break one, on the same
starting incumbent, the same seeds and the same wall-clock budget.

    python scripts/bench_wall.py --cell 5x7x3 --budget 60 --seeds 5
    python scripts/bench_wall.py --cell 6x6x3 --target 60 --configs lns,wall

Protocol
--------
1. Establish an incumbent: greedy + peel + a short default descent
   (fixed seed), cached under --cache so every config starts from
   *identical* bytes. The size attacked is the *hardest* one that
   descent reached, started from the solution one notch above it — so
   the target is known-feasible and known-hard. ``--target M``
   overrides it.
2. For each config x seed: load the incumbent, run the config at fixed M
   for --budget seconds, record the final exact uncovered count (always
   re-verified by a full recount), the time to zero if it got there, and
   the pass throughput counters.
3. Print a table, and dump raw JSON with --json.

Configs (ablations, so a win can be attributed):

  lns           the default notch attempt: a single fixed-mix LNS
  deficit       the same, with the deficit-targeted ruin operator added
  polish        LNS for half the budget, then LAHC polish over the stock
                pool neighbourhood (the documented best before this
                round), fixed pool cadence
  polish_auto   the same with pool_refresh="auto"
  polish_shell  the same with the local (shell) candidate neighbourhood
  wall_nots     break_wall with the exact two-swap endgame disabled
  wall_nodef    break_wall with the deficit-targeted ruin disabled
  wall          break_wall: focused-ruin LNS / local polish / two-swap
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coldcore.search import Searcher                        # noqa: E402


def make_backend(cell, backend):
    q, n, R = (int(x) for x in cell.split("x"))
    if backend == "ref":
        from coldcore.reference import RefCoveringBackend
        return RefCoveringBackend(q, n, R), (q, n, R)
    from coldcore.gpu import GpuBackend
    return GpuBackend([q] * n, R), (q, n, R)


def incumbent(b, cache, secs, seed=1):
    """A start state one solved size ABOVE the hardest size the default
    descent could reach in ``secs``, and that hardest size as the target.

    That makes the target both *provably feasible* (the same descent
    produced a witness at it, so a config that fails is failing at
    something reachable) and genuinely hard (it was the last notch the
    setup budget could manage). Cached as word indices, so every config
    and seed starts from identical bytes."""
    tfile = (cache + ".target") if cache else None
    if cache and os.path.exists(cache) and os.path.exists(tfile):
        idx = np.loadtxt(cache, dtype=np.int64, ndmin=1)
        with open(tfile) as fh:
            return np.asarray(idx, dtype=np.int64), int(fh.read().strip())
    seen = []
    # +35s: descend reserves 30s of the deadline before starting a notch
    s = Searcher(b, seed=seed, deadline=time.time() + secs + 35,
                 on_solved=lambda m, i: seen.append((m, np.sort(i).copy())))
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    s.descend(floor=1, notch_budget_s=max(4.0, secs / 12.0))
    if len(seen) < 2:
        raise SystemExit("setup descent never improved; raise --setup")
    print(f"# setup: descent reached M={seen[-1][0]} "
          f"(sizes {[m for m, _ in seen]}); attacking it from M={seen[-2][0]}")
    idx, target = seen[-2][1], int(seen[-1][0])
    if cache:
        np.savetxt(cache, idx, fmt="%d")
        with open(tfile, "w") as fh:
            fh.write(str(target))
    return idx, target


def run(cfg, b, start, M, budget, seed):
    s = Searcher(b, seed=seed, deadline=time.time() + budget + 60)
    s.load(start)
    if len(s.code) > M:
        s.ruin(len(s.code) - M, "low")
        s.greedy_fill(M)
    t0 = time.time()
    hit = {"t": None}

    def watch(self):
        if self.uncov == 0 and hit["t"] is None:
            hit["t"] = time.time() - t0
    s.on_improve = watch

    if cfg == "lns":
        s.lns(M, round_budget_s=budget, seed=seed)
    elif cfg == "deficit":
        s.lns(M, round_budget_s=budget, seed=seed, deficit=True)
    elif cfg in ("polish", "polish_auto", "polish_shell"):
        s.lns(M, round_budget_s=budget * 0.5, seed=seed)
        if s.uncov:
            s.anneal(M, budget_s=budget * 0.5, mode="lahc", seed=seed,
                     pool_refresh=("auto" if cfg != "polish" else 64),
                     neighbourhood=("shell" if cfg == "polish_shell"
                                    else "pool"))
    elif cfg == "wall":
        s.break_wall(M, budget_s=budget, burst_s=max(2.0, budget / 8.0),
                     seed=seed)
    elif cfg == "wall_nots":
        s.break_wall(M, budget_s=budget, burst_s=max(2.0, budget / 8.0),
                     seed=seed, two_swap_at=0)
    elif cfg == "wall_nodef":
        s.break_wall(M, budget_s=budget, burst_s=max(2.0, budget / 8.0),
                     seed=seed, deficit=False)
    else:
        raise SystemExit(f"unknown config {cfg!r}")

    wall = time.time() - t0
    unc = s.verify_full()                    # exact, from scratch
    st = dict(s.stats)
    st.pop("ops", None)
    st.pop("wall", None)
    return {"cfg": cfg, "seed": seed, "uncov": int(unc), "M": int(len(s.code)),
            "secs": round(wall, 2),
            "t_solved": (round(hit["t"], 2) if unc == 0 and hit["t"]
                         else None),
            "rounds_s": round(st["rounds"] / wall, 1),
            "moves_s": round(st["moves"] / wall, 1),
            "transforms_s": round(st["transforms"] / wall, 1),
            "accept_rate": round(st["accepts"] / max(1, st["moves"]), 4),
            "shift_moves": int(st.get("shift_moves", 0)),
            "shift_accepts": int(st.get("shift_accepts", 0)),
            "hood": st.get("polish_neighbourhood"),
            "two_swap_nodes": int(st.get("two_swap_nodes", 0)),
            "two_swap_hits": int(st.get("two_swap_hits", 0)),
            "deficit_ruins": int(st.get("deficit_ruins", 0))}


DEFAULT_CFGS = ("lns,deficit,polish,polish_auto,polish_shell,"
                "wall_nots,wall")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="5x7x3", help="qxnxR, e.g. 5x7x3")
    ap.add_argument("--backend", choices=["gpu", "ref"], default="gpu")
    ap.add_argument("--budget", type=float, default=60.0,
                    help="seconds per (config, seed)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--target", type=int, default=None,
                    help="M to attack (default: the hardest size the setup "
                         "descent reached)")
    ap.add_argument("--configs", default=DEFAULT_CFGS)
    ap.add_argument("--setup", type=float, default=120.0,
                    help="seconds for the shared incumbent build")
    ap.add_argument("--cache", default=None, help="incumbent cache file")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    b, (q, n, R) = make_backend(a.cell, a.backend)
    cache = a.cache or f"/tmp/coldcore_inc_{a.cell}.txt"
    start, auto_target = incumbent(b, cache, a.setup)
    M = a.target if a.target is not None else auto_target
    print(f"# K_{q}({n},{R})  |X|={b.space.size}  incumbent M={len(start)}  "
          f"attacking M={M}  budget={a.budget}s x {a.seeds} seeds")

    rows = []
    # seeds outer / configs inner: every config sees the same machine
    # conditions, which matters when the GPU is shared
    for k in range(a.seeds):
        for cfg in a.configs.split(","):
            r = run(cfg, b, start, M, a.budget, a.seed0 + k)
            rows.append(r)
            print(f"  {cfg:<13} seed={r['seed']} uncov={r['uncov']} "
                  f"t_solved={r['t_solved']} rounds/s={r['rounds_s']} "
                  f"moves/s={r['moves_s']} acc={r['accept_rate']} "
                  f"tr/s={r['transforms_s']} shift_mv={r['shift_moves']} "
                  f"ts_nodes={r['two_swap_nodes']}", flush=True)
    b.close()

    print(f"\n{'config':<13} {'solved':>7} {'best':>5} {'mean unc':>9} "
          f"{'med t_solve':>11} {'rounds/s':>9} {'moves/s':>9} "
          f"{'acc':>7} {'tr/s':>7}")
    summary = []
    for cfg in a.configs.split(","):
        rs = [r for r in rows if r["cfg"] == cfg]
        ok = [r for r in rs if r["uncov"] == 0]
        ts = sorted(r["t_solved"] for r in ok if r["t_solved"] is not None)
        med = (round(ts[len(ts) // 2], 1) if ts else None)
        line = {"cfg": cfg, "solved": f"{len(ok)}/{len(rs)}",
                "best": min(r["uncov"] for r in rs),
                "mean_uncov": round(float(np.mean([r["uncov"] for r in rs])), 2),
                "med_t_solved": med,
                "rounds_s": round(float(np.mean([r["rounds_s"] for r in rs])), 1),
                "moves_s": round(float(np.mean([r["moves_s"] for r in rs])), 1),
                "accept_rate": round(float(np.mean(
                    [r["accept_rate"] for r in rs])), 4),
                "tr_s": round(float(np.mean([r["transforms_s"] for r in rs])), 1)}
        summary.append(line)
        print(f"{cfg:<13} {line['solved']:>7} {line['best']:>5} "
              f"{line['mean_uncov']:>9} {str(med):>11} "
              f"{line['rounds_s']:>9} {line['moves_s']:>9} "
              f"{line['accept_rate']:>7} {line['tr_s']:>7}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"cell": a.cell, "M": M, "incumbent": len(start),
                       "budget": a.budget, "rows": rows,
                       "summary": summary}, fh, indent=1)


if __name__ == "__main__":
    main()

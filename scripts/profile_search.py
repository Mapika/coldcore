#!/usr/bin/env python3
"""cProfile a representative search run, and report search-core throughput.

The run is the one the campaigns actually spend their time in: a fixed
number of LNS ruin/recreate rounds at a fixed M, then a fixed-duration
shell-neighbourhood LAHC polish. Both are driven with a fixed seed, so
the comparison before/after an optimisation is paired.

    python3 scripts/profile_search.py --backend cpu --cell K7(8,3)
    python3 scripts/profile_search.py --backend gpu --cell K3(8,2) --polish-s 3

Reports: the cProfile top-N by cumulative and by tottime, the number of
LNS rounds and polish moves achieved, and moves/second — which is the
number an optimisation has to move.
"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coldcore.backends import backend_for_spec        # noqa: E402
from coldcore.search import Searcher                  # noqa: E402


def build(backend, cell, seed, notch):
    b = backend_for_spec(backend, cell)
    s = Searcher(b, seed=seed)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    s.peel()
    target = max(1, len(s.code) - int(notch))
    s.ruin(len(s.code) - target, "low")
    return b, s, target


def workload(s, target, rounds, polish_s, seed):
    """A fixed round count (clock-free) plus a fixed-duration polish."""
    for i in range(rounds):
        s.ruin(1 + i % 4, ["low", "hole", "cluster", "random"][i % 4])
        s.greedy_fill(target)
    if polish_s > 0:
        s.anneal(target, budget_s=polish_s, mode="lahc", seed=seed,
                 neighbourhood="shell", log_every=0, pool_refresh=64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="cpu", choices=["ref", "cpu", "gpu"])
    ap.add_argument("--cell", default="K5(8,3)")
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--polish-s", type=float, default=10.0)
    ap.add_argument("--notch", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--top", type=int, default=22)
    ap.add_argument("--no-profile", action="store_true",
                    help="time only (profiler overhead distorts fine-grained "
                         "hot loops; use this for the headline mv/s)")
    args = ap.parse_args(argv)

    b, s, target = build(args.backend, args.cell, args.seed, args.notch)
    print(f"cell {args.cell} |X|={b.space.size:,} backend={args.backend} "
          f"M={len(s.code)} target={target} uncov={s.uncov}", flush=True)

    t0 = time.perf_counter()
    if args.no_profile:
        workload(s, target, args.rounds, args.polish_s, args.seed)
    else:
        pr = cProfile.Profile()
        pr.enable()
        workload(s, target, args.rounds, args.polish_s, args.seed)
        pr.disable()
    dt = time.perf_counter() - t0

    st = s.stats
    mv = st.get("shift_moves", 0)
    print(f"\n{dt:.2f}s wall: rounds={st['rounds'] + args.rounds} "
          f"placements={st['placements']} removals={st['removals']} "
          f"transforms={st['transforms']} shift_moves={mv} "
          f"shift_accepts={st.get('shift_accepts', 0)}")
    if args.polish_s > 0:
        print(f"polish throughput: {mv / max(args.polish_s, 1e-9):.1f} mv/s")
    print(f"final uncov={s.uncov} M={len(s.code)}")

    if not args.no_profile:
        for sort in ("cumulative", "tottime"):
            buf = io.StringIO()
            ps = pstats.Stats(pr, stream=buf).sort_stats(sort)
            ps.print_stats(args.top)
            print(f"\n===== by {sort} =====")
            print("\n".join(buf.getvalue().splitlines()[4:]))
    b.close()


if __name__ == "__main__":
    main()

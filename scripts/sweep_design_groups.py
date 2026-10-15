#!/usr/bin/env python3
"""Group sweep for covering designs (docs/incidence.md, "Group-invariant
search"): for every built-in group of the point set, orbit search on
the quotient backend, then the free release seeded with the orbit
solution.

    python scripts/sweep_design_groups.py v k t SPEC orbit_s release_s \
        [short_orbits 0|1] [device]

Prints one JSON line with the greedy, orbit-best and released-best block
counts and the time to each; logs the trajectory on stderr.
"""
import json
import sys
import time

from coldcore.problems.covering_design import CoveringDesignBackend
from coldcore.problems.design_orbits import (DesignOrbits, orbit_search,
                                             release)


def main():
    v, k, t = (int(x) for x in sys.argv[1:4])
    spec = sys.argv[4]
    orbit_s, release_s = float(sys.argv[5]), float(sys.argv[6])
    short = bool(int(sys.argv[7])) if len(sys.argv) > 7 else False
    device = sys.argv[8] if len(sys.argv) > 8 else "gpu"
    t0 = time.time()
    orb = DesignOrbits(v, k, t, spec)
    t_build = time.time() - t0
    log = lambda m: print(f"  [{time.time() - t0:6.0f}s] {m}", file=sys.stderr,  # noqa: E731
                          flush=True)
    log(orb.summary())
    traj = []

    def on_orbit(n, blocks):
        traj.append((round(time.time() - t0, 1), "orbit", int(n)))
        log(f"orbit solution: {n} blocks")

    blocks, s, q = orbit_search(v, k, t, spec, budget_s=orbit_s, device=device,
                                short_orbits=short, on_log=log,
                                on_solved=on_orbit, orbits=orb,
                                notch_budget_s=max(5.0, orbit_s / 6))
    n_orbit = len(blocks) if len(blocks) else None
    greedy = traj[0][2] if traj else None
    n_rel = None
    if n_orbit:
        full = CoveringDesignBackend(v, k, t)
        assert full.verify(blocks), "orbit solution failed independent verify"

        def on_rel(n, bl):
            traj.append((round(time.time() - t0, 1), "release", int(n)))
            log(f"released solution: {n} blocks")

        rel, s2 = release(v, k, t, blocks, release_s, device=device, on_log=log,
                          on_solved=on_rel, notch_budget_s=max(5.0, release_s / 4),
                          step0=1)
        assert full.verify(rel)
        n_rel = len(rel)
    rec = dict(cell=f"C({v},{k},{t})", group=spec, order=orb.order,
               block_orbits=len(orb.block_part),
               full_orbits=int((orb.block_part.sizes == orb.order).sum()),
               t_orbits=len(orb.univ_part), build_s=round(t_build, 1),
               short_orbits=short, greedy=greedy, orbit_best=n_orbit,
               released_best=n_rel, traj=traj, secs=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()

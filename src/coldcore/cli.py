"""coldcore CLI.

    python -m coldcore solve --backend ref --plugin covering -p q=3 -p n=4 -p R=1
    python -m coldcore solve --backend gpu --plugin path/to/libcoldcore_covering.so \
        -p q=7 -p n=8 -p R=3 --hours 1 --out live.txt --seed-file start.txt

`solve` = greedy build (or seed file) -> peel -> notch descent, writing
every verified improvement to --out (atomic) and logging to stdout.

The notch attempt is configurable but defaults to the proven pass
sequence exactly: `--ruin-select adaptive` re-weights the LNS ruin
operators by measured reward, `--portfolio N` runs N time-sliced LNS
streams around a shared incumbent, `--deficit` adds the
deficit-targeted ruin operator, `--polish anneal|lahc` spends
`--polish-share` of each notch budget on the swap-neighbourhood polish
when the repair walls out, and `--polish-neighbourhood shell|auto`
switches that polish to the local candidate batch — which is what makes
it accept anything at all on a walled cell. See docs/search-passes.md.

`--wall M` replaces the descent with the wall-breaker: one fixed target
size, worked with focused-ruin LNS / polish / exact two-swap bursts and
restart-from-best. That is the pass sequence for the case where a cover
at M+1 is in hand and M has been stuck at 1-3 uncovered for hours:

    python -m coldcore solve --backend gpu -p q=6,n=10,R=5 \
        --seed-file best_483.txt --wall 482 --hours 6 --out wall482.txt
"""
import argparse
import sys
import time

import numpy as np

from .backends import BACKENDS, make_backend
from .io import read_solution, write_solution_atomic
from .search import Searcher


def _params(pairs):
    out = {}
    for p in pairs or []:
        for item in p.split(","):
            if not item:
                continue
            k, _, v = item.partition("=")
            out[k.strip()] = v.strip()
    return out


def _mk_backend(args, prm):
    problem = prm.get("problem", "hamming")
    R = int(prm["R"])
    if "axes" in prm:                       # e.g. -p axes=9x9  or  3x3x4x5
        axes = [int(x) for x in prm["axes"].split("x")]
    else:
        axes = [int(prm["q"])] * int(prm["n"])
    mu = int(prm.get("mu", 1))              # mu-fold covering K^mu (>=1)
    plugin = None if args.plugin in (None, "covering") else args.plugin
    try:
        return make_backend(args.backend, axes, R, problem=problem, mu=mu,
                            plugin_path=plugin)
    except ValueError as e:
        sys.exit(str(e))


def cmd_symmetry(args, b, deadline, on_solved, spec):
    """Orbit-restricted solve: the solution is a union of G-orbits.

    Greedy over orbits -> orbit peel -> orbit notch descent (one notch =
    one orbit's worth of points). With ``--symmetry-polish`` the
    group-invariant solution is then handed to the ordinary word-level
    searcher, which is free to break the symmetry."""
    from .symmetry import SymmetricSearcher, parse_group
    G = parse_group(b.space, spec)
    s = SymmetricSearcher(b, G, seed=args.seed, deadline=deadline,
                          on_log=lambda m: print(f"[coldcore] {m}", flush=True),
                          on_solved=on_solved)
    p = s.part
    print(f"[coldcore] symmetry {spec!r}: {len(p)} orbits of sizes "
          f"{int(p.sizes.min())}..{int(p.sizes.max())} over |X|={b.space.size} "
          f"(search space cut by ~{b.space.size / max(len(p), 1):.1f}x)",
          flush=True)
    s.greedy_fill(10 ** 9)
    if s.uncov != 0:
        sys.exit("[coldcore] no union of orbits covers the space; the group "
                 "is too coarse for this cell")
    s.peel()
    code = s.descend(floor=args.floor, notch_budget_s=args.notch_budget)
    print(f"[coldcore] orbit descent ended at M={len(code)} stats={s.stats}",
          flush=True)
    if not args.symmetry_polish:
        return
    free = s.free_searcher(seed=args.seed)
    free.on_solved = on_solved
    free.on_log = lambda m: print(f"[coldcore] {m}", flush=True)
    print(f"[coldcore] handing M={len(free.code)} to word-level search",
          flush=True)
    M = free.descend(floor=args.floor, notch_budget_s=args.notch_budget,
                     step0=args.step0, select=args.ruin_select or "fixed")
    print(f"[coldcore] free polish ended at M={M} stats={free.stats}",
          flush=True)


def cmd_solve(args):
    prm = _params(args.param)
    if "R" not in prm or not ("axes" in prm or ("q" in prm and "n" in prm)):
        sys.exit("need -p R=... plus either -p axes=AxBx... or -p q=.. -p n=..")
    b = _mk_backend(args, prm)
    deadline = time.time() + args.hours * 3600 if args.hours else None

    def on_solved(M, idx):
        print(f"[coldcore] SOLVED M={M}", flush=True)
        if args.out:
            write_solution_atomic(args.out, b.index_word(idx))

    spec = args.symmetry or prm.get("group")
    if spec:
        try:
            cmd_symmetry(args, b, deadline, on_solved, spec)
        finally:
            b.close()
        return

    s = Searcher(b, seed=args.seed, deadline=deadline,
                 on_log=lambda m: print(f"[coldcore] {m}", flush=True),
                 on_solved=on_solved)
    polish = None if args.polish == "none" else args.polish
    try:
        if args.seed_file:
            s.load(read_solution(args.seed_file, n=b.space.n))
            if s.uncov != 0:
                print(f"[coldcore] seed not feasible (uncov={s.uncov}); "
                      f"repairing at M={len(s.code)}", flush=True)
                s.lns(len(s.code), round_budget_s=args.notch_budget * 4,
                      select=args.ruin_select or "fixed")
        else:
            s.load(np.zeros(0, dtype=np.int64))
            s.greedy_fill(target_m=10 ** 9)
        if args.wall is not None:
            M0 = len(s.code)
            print(f"[coldcore] wall-breaking at M={args.wall} from M={M0} "
                  f"(uncov={s.uncov})", flush=True)
            budget = (args.wall_budget if args.wall_budget else
                      (s.time_left() - 30 if deadline else 3600.0))
            unc = s.break_wall(args.wall, budget_s=budget,
                               burst_s=args.wall_burst,
                               select=args.ruin_select or "adaptive",
                               polish=(polish or "lahc"),
                               polish_share=args.wall_polish_share,
                               polish_neighbourhood=args.polish_hood
                               or "auto")
            if unc == 0 and s.verify_full() == 0:
                on_solved(len(s.code), s.code)
            print(f"[coldcore] wall M={args.wall} ended at uncov={unc} "
                  f"stats={s.stats}", flush=True)
            return
        if s.uncov != 0:
            sys.exit("[coldcore] could not establish a feasible solution")
        M0 = len(s.code)
        print(f"[coldcore] feasible at M={M0}; descending "
              f"(floor={args.floor})", flush=True)
        M = s.descend(floor=args.floor, notch_budget_s=args.notch_budget,
                      step0=args.step0, select=args.ruin_select or "fixed",
                      polish=polish, polish_share=args.polish_share,
                      portfolio=args.portfolio, deficit=args.deficit,
                      polish_neighbourhood=args.polish_hood or "pool")
        print(f"[coldcore] descent ended at M={M} stats={s.stats}",
              flush=True)
    finally:
        b.close()


def cmd_campaign(args):
    """Sweep a list of cells, one worker process each."""
    from .backends import split_cells
    from .campaign import run_campaign, summarise
    if args.cells_file:
        with open(args.cells_file) as fh:
            cells = [ln.strip() for ln in fh
                     if ln.strip() and not ln.startswith("#")]
    else:
        cells = split_cells(args.cells or "")
    if not cells:
        sys.exit("need --cells 'K3(4,1),lee:5x5:1' or --cells-file PATH")
    polish = None if args.polish == "none" else args.polish
    jobs = [{"cell": c} for c in cells]
    recs = run_campaign(
        jobs, outdir=args.outdir, workers=args.workers,
        resume=not args.no_resume, backend=args.backend,
        plugin_path=(None if args.plugin in (None, "covering")
                     else args.plugin),
        budget_s=args.hours * 3600 if args.hours else args.budget,
        seed=args.seed, floor=args.floor, notch_budget_s=args.notch_budget,
        select=args.ruin_select or "fixed", polish=polish,
        polish_share=args.polish_share, portfolio=args.portfolio,
        deficit=args.deficit, polish_neighbourhood=args.polish_hood or "pool",
        on_done=lambda r: print(
            f"[coldcore] done {r['label']}: M={r['M']} uncov={r['uncov']} "
            f"({r['secs']:.0f}s)" + (" ERROR" if r["error"] else ""),
            flush=True))
    print(summarise(recs))
    print(f"\n[coldcore] records in {args.outdir}/campaign.json")


def cmd_info(args):
    from . import __version__
    print(f"coldcore {__version__}")
    try:
        from .gpu import default_plugin_path, load_plugin
        path = args.plugin or default_plugin_path()
        lib = load_plugin(path)
        gen = [s for s in ("dct_init", "dct_init2", "dct_init3")
               if hasattr(lib, s)]
        print(f"plugin: {path}\n  init entry points: {', '.join(gen)}")
    except OSError as e:
        print(f"plugin: not loadable ({e})\n"
              "  build it with `cmake -B build && cmake --build build -j` in the\n"
              "  source tree, or point COLDCORE_PLUGIN_PATH (or --plugin) at\n"
              "  the built libcoldcore_covering.so")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="coldcore")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("solve", help="greedy build + peel + notch descent")
    sp.add_argument("--backend", choices=list(BACKENDS), default="gpu")
    sp.add_argument("--plugin",
                    help="plugin .so path (gpu) or name (ref/cpu); default: "
                         "built-in covering plugin")
    sp.add_argument("-p", "--param", action="append",
                    help="problem parameter key=value (repeatable)")
    sp.add_argument("--hours", type=float, default=None)
    sp.add_argument("--floor", type=int, default=1)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--seed-file", help="start from this solution file")
    sp.add_argument("--out", help="write best solutions here (atomic)")
    sp.add_argument("--notch-budget", type=float, default=300.0)
    sp.add_argument("--step0", type=int, default=4)
    sp.add_argument("--ruin-select", choices=["fixed", "adaptive"],
                    default=None,
                    help="LNS ruin-operator selection: the proven fixed mix, "
                         "or a reward-weighted bandit (default: fixed for the "
                         "descent, adaptive for --wall)")
    sp.add_argument("--polish", choices=["none", "anneal", "lahc"],
                    default="none",
                    help="polish a walled-out notch with a swap "
                         "neighbourhood that accepts regressions: Metropolis "
                         "annealing or late acceptance (default: none)")
    sp.add_argument("--polish-share", type=float, default=0.5,
                    help="fraction of each notch budget reserved for "
                         "--polish (default: 0.5)")
    sp.add_argument("--portfolio", type=int, default=1,
                    help="run N time-sliced LNS streams per notch around a "
                         "shared incumbent (default: 1, a single stream)")
    sp.add_argument("--polish-neighbourhood", dest="polish_hood",
                    choices=["pool", "shell", "auto"], default=None,
                    help="where --polish draws its candidates: the frozen "
                         "global gain pool, the dropped word's local shell, "
                         "or auto (pool until its accepts collapse). The "
                         "descent defaults to pool; --wall defaults to auto")
    sp.add_argument("--wall-polish-share", type=float, default=0.6,
                    help="fraction of each --wall burst given to the polish "
                         "(default: 0.6)")
    sp.add_argument("--deficit", action="store_true",
                    help="add the deficit-targeted ruin operator to the LNS "
                         "mix (ruins the incumbents nearest the words that "
                         "are actually uncovered); off by default")
    sp.add_argument("--wall", type=int, default=None, metavar="M",
                    help="skip the descent and run the wall-breaker at this "
                         "fixed size: focused-ruin LNS / polish / exact "
                         "two-swap bursts with restart-from-best")
    sp.add_argument("--wall-budget", type=float, default=None,
                    help="seconds for --wall (default: all remaining time)")
    sp.add_argument("--wall-burst", type=float, default=None,
                    help="seconds per --wall burst (default: budget/12)")
    sp.add_argument("--symmetry", metavar="SPEC", default=None,
                    help="search over ORBITS of this group instead of over "
                         "words: the solution is a union of orbits (the "
                         "group-invariant construction). Terms joined with "
                         "'+': translate:VEC, diag[:S], shift:AXIS[:S], "
                         "shifts, rot, sym, mono, iso:FAMILY, trivial. "
                         "e.g. --symmetry translate:1011 . Also reachable as "
                         "-p group=SPEC. See docs/symmetry.md")
    sp.add_argument("--symmetry-polish", action="store_true",
                    help="after the orbit descent, hand the group-invariant "
                         "solution to the ordinary word-level descent (which "
                         "may break the symmetry)")
    sp.set_defaults(fn=cmd_solve)

    cp = sub.add_parser("campaign",
                        help="sweep many cells, one worker process each")
    cp.add_argument("--cells", help="comma-separated cell specs, e.g. "
                                    "'K3(8,2),lee:5x5:1,K2(6,1)mu3'")
    cp.add_argument("--cells-file", help="one cell spec per line ('#' = "
                                         "comment)")
    cp.add_argument("--backend", choices=list(BACKENDS), default="cpu")
    cp.add_argument("--plugin")
    cp.add_argument("--outdir", default="runs")
    cp.add_argument("--workers", type=int, default=4)
    cp.add_argument("--budget", type=float, default=600.0,
                    help="seconds per cell (default 600)")
    cp.add_argument("--hours", type=float, default=None,
                    help="hours per cell (overrides --budget)")
    cp.add_argument("--no-resume", action="store_true",
                    help="ignore any existing solution file in --outdir")
    cp.add_argument("--seed", type=int, default=1)
    cp.add_argument("--floor", type=int, default=1)
    cp.add_argument("--notch-budget", type=float, default=None)
    cp.add_argument("--step0", type=int, default=4)
    cp.add_argument("--ruin-select", choices=["fixed", "adaptive"],
                    default=None)
    cp.add_argument("--polish", choices=["none", "anneal", "lahc"],
                    default="none")
    cp.add_argument("--polish-share", type=float, default=0.5)
    cp.add_argument("--polish-neighbourhood", dest="polish_hood",
                    choices=["pool", "shell", "auto"], default=None)
    cp.add_argument("--portfolio", type=int, default=1)
    cp.add_argument("--deficit", action="store_true")
    cp.set_defaults(fn=cmd_campaign)

    si = sub.add_parser("info", help="version + plugin probe")
    si.add_argument("--plugin")
    si.set_defaults(fn=cmd_info)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

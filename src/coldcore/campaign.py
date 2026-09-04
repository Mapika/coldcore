"""Multi-cell campaign runner: N cells, N worker processes, one call.

Every campaign so far has hand-rolled the same shell loop — spawn one
process per cell, redirect a log, remember where the checkpoint went,
notice which cells died. This is that loop, once, as a library.

It is deliberately thin: :mod:`multiprocessing` through
``ProcessPoolExecutor``, one job per process, no daemon, no scheduler,
no state beyond the output directory. A campaign is a list of job
dicts; a job is a backend kind, a cell spec and a budget, plus whatever
:meth:`~coldcore.search.Searcher.descend` / ``break_wall`` knobs the
cell needs.

    from coldcore.campaign import run_campaign
    run_campaign([{"cell": "K3(8,2)", "budget_s": 600},
                  {"cell": "K5(8,3)", "budget_s": 600, "polish": "lahc"}],
                 backend="cpu", workers=8, outdir="runs/sweep")

Per cell the runner writes, into ``outdir``:

``<label>.txt``     the best verified solution (atomic rewrite on every
                    improvement — this is the artefact)
``<label>.log``     human-readable progress
``<label>.json``    the final record (best M, stats, timings, error)

and ``campaign.json`` collects every record. A rerun **resumes**: an
existing ``<label>.txt`` is loaded as the seed, so a campaign can be
stopped and restarted without losing ground.

One process per cell is not just convenient — the GPU plugin ABI is a
singleton per loaded ``.so`` per process, so a GPU campaign *must* fork.
"""
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .backends import backend_for_spec, parse_cell
from .io import read_solution, write_solution_atomic
from .search import Searcher

#: Job keys and their defaults. Anything else in a job dict is rejected,
#: so a typo in a campaign script fails loudly instead of silently
#: running the default search.
JOB_DEFAULTS = {
    "cell": None,                # required: a coldcore.backends cell spec
    "label": None,               # defaults to the cell spec, path-sanitised
    "backend": "cpu",
    "plugin_path": None,
    "budget_s": 600.0,
    "seed": 1,
    "floor": 1,
    "notch_budget_s": None,      # default: budget/10
    "step0": 4,
    "select": "fixed",
    "polish": None,
    "polish_share": 0.5,
    "polish_neighbourhood": "pool",
    "portfolio": 1,
    "deficit": False,
    "wall": None,                # int M: run break_wall instead of descend
    "wall_burst_s": None,
    "seed_file": None,           # explicit start; otherwise resume/greedy
}


def _safe(label):
    return "".join(c if c.isalnum() or c in "-_.=" else "_" for c in label)


def normalise(job):
    """Fill defaults and validate one job dict."""
    unknown = set(job) - set(JOB_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown campaign job keys: {sorted(unknown)}")
    out = dict(JOB_DEFAULTS)
    out.update(job)
    if not out["cell"]:
        raise ValueError("a campaign job needs a 'cell' spec")
    parse_cell(out["cell"])              # fail here, not in the worker
    out["label"] = _safe(out["label"] or out["cell"])
    return out


def run_job(job, outdir=".", resume=True):
    """Run one cell to its budget. Returns the record dict.

    Never raises: a crashed cell is a record with ``error`` set, so one
    bad cell cannot take the campaign down with it."""
    job = normalise(job)
    os.makedirs(outdir, exist_ok=True)
    label = job["label"]
    sol = os.path.join(outdir, f"{label}.txt")
    logp = os.path.join(outdir, f"{label}.log")
    t0 = time.time()
    rec = {"label": label, "cell": job["cell"], "backend": job["backend"],
           "seed": job["seed"], "budget_s": job["budget_s"], "M": None,
           "uncov": None, "solutions": [], "stats": {}, "error": None,
           "solution_file": sol, "log_file": logp}
    log = open(logp, "a", buffering=1)
    b = None
    try:
        log.write(f"\n=== {label} {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"backend={job['backend']} budget={job['budget_s']}s ===\n")
        b = backend_for_spec(job["backend"], job["cell"],
                             plugin_path=job["plugin_path"])
        deadline = time.time() + float(job["budget_s"])

        def on_solved(M, idx):
            rec["M"] = int(M)
            rec["solutions"].append([round(time.time() - t0, 2), int(M)])
            write_solution_atomic(sol, b.index_word(idx))
            log.write(f"[{time.time() - t0:8.1f}s] SOLVED M={M}\n")

        s = Searcher(b, seed=job["seed"], deadline=deadline,
                     on_log=lambda m: log.write(
                         f"[{time.time() - t0:8.1f}s] {m}\n"),
                     on_solved=on_solved)
        start = job["seed_file"] or (sol if resume and os.path.exists(sol)
                                     else None)
        if start:
            s.load(read_solution(start, n=b.space.n))
            log.write(f"resumed from {start} at M={len(s.code)} "
                      f"(uncov={s.uncov})\n")
            if s.uncov != 0:
                s.lns(len(s.code), round_budget_s=float(job["budget_s"]) / 4,
                      select=job["select"])
        else:
            s.load(np.zeros(0, dtype=np.int64))
            s.greedy_fill(10 ** 9)
        notch = job["notch_budget_s"] or max(1.0, float(job["budget_s"]) / 10)
        if job["wall"] is None and float(job["budget_s"]) < 60:
            # descend refuses to start a notch it cannot finish 30 s
            # before the deadline, so a very short budget descends zero
            # notches. Say so rather than reporting a mystery no-op.
            log.write(f"WARNING budget {job['budget_s']}s is below the "
                      f"descent's 30 s end-of-run reserve; expect the greedy "
                      f"build only\n")
        if job["wall"] is not None:
            s.break_wall(int(job["wall"]), budget_s=max(0.0, deadline -
                                                        time.time()),
                         burst_s=job["wall_burst_s"],
                         select=job["select"] or "adaptive",
                         polish=job["polish"] or "lahc",
                         polish_neighbourhood=job["polish_neighbourhood"])
            if s.uncov == 0 and s.verify_full() == 0:
                on_solved(len(s.code), s.code)
        elif s.uncov == 0:
            s.descend(floor=job["floor"], notch_budget_s=notch,
                      step0=job["step0"], select=job["select"],
                      polish=job["polish"],
                      polish_share=job["polish_share"],
                      portfolio=job["portfolio"], deficit=job["deficit"],
                      polish_neighbourhood=job["polish_neighbourhood"])
        else:
            log.write("no feasible cover established inside the budget\n")
        rec["uncov"] = int(s.uncov)
        rec["M"] = rec["M"] if rec["M"] is not None else len(s.code)
        rec["stats"] = {k: v for k, v in s.stats.items()
                        if isinstance(v, (int, float))}
    except Exception:
        rec["error"] = traceback.format_exc()
        log.write(rec["error"])
    finally:
        if b is not None:
            try:
                b.close()
            except Exception:
                pass
        rec["secs"] = round(time.time() - t0, 2)
        log.write(f"=== done M={rec['M']} uncov={rec['uncov']} "
                  f"in {rec['secs']}s ===\n")
        log.close()
    with open(os.path.join(outdir, f"{label}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def _worker(args):
    return run_job(*args)


def run_campaign(jobs, outdir="runs", workers=4, resume=True, on_done=None,
                 **common):
    """Run every job, ``workers`` at a time. Returns the list of records.

    ``common`` supplies defaults for keys a job does not set (e.g.
    ``backend="gpu", budget_s=3600``). Jobs are dispatched in the order
    given and results collected as they finish; ``on_done(record)`` is
    called for each."""
    jobs = [normalise({**common, **j}) for j in jobs]
    labels = [j["label"] for j in jobs]
    if len(set(labels)) != len(labels):
        raise ValueError("campaign job labels must be unique "
                         "(set 'label' when running one cell twice)")
    os.makedirs(outdir, exist_ok=True)
    recs = []
    workers = max(1, min(int(workers), len(jobs)))
    if workers == 1:                      # in-process: easier to debug
        for j in jobs:
            r = run_job(j, outdir, resume)
            recs.append(r)
            if on_done:
                on_done(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, (j, outdir, resume)): j for j in jobs}
            for f in as_completed(futs):
                r = f.result()
                recs.append(r)
                if on_done:
                    on_done(r)
    order = {lab: i for i, lab in enumerate(labels)}
    recs.sort(key=lambda r: order[r["label"]])
    with open(os.path.join(outdir, "campaign.json"), "w") as fh:
        json.dump({"outdir": outdir, "workers": workers, "records": recs},
                  fh, indent=1)
    return recs


def summarise(recs):
    """A one-line-per-cell table of a finished campaign."""
    lines = [f"{'cell':22s} {'backend':>7s} {'M':>7s} {'uncov':>7s} "
             f"{'secs':>8s}  status"]
    lines.append("-" * 70)
    for r in recs:
        status = "ERROR" if r["error"] else (
            "cover" if r["uncov"] == 0 else "infeasible")
        lines.append(f"{r['label']:22s} {r['backend']:>7s} "
                     f"{str(r['M']):>7s} {str(r['uncov']):>7s} "
                     f"{r['secs']:8.1f}  {status}")
    return "\n".join(lines)

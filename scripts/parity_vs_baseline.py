#!/usr/bin/env python3
"""Bit-parity of a candidate plugin build against a baseline build.

Loads both shared objects into one process (each is its own singleton:
separate ``dct_*`` globals) and drives them through the same random
sequence of operations on one cell, comparing every returned value and
the whole ``cnt`` field after each mutation:

    recount -> count_eq / uncovered -> ball_update (single words and
    batches, both signs) -> ball_gather (roles 0/1, batch sizes from 1
    to 4096) -> gain_map / loss_map -> map_max / map_hist / map_extract
    (as sorted sets) / map_read_at.

Maps are compared in full where the extraction of every positive entry
fits ``--full-cap``; otherwise by max, histogram, the extracted top
slice and a random sample of positions.

    python3 scripts/parity_vs_baseline.py --base /path/old.so \\
        --cells 'K5(8,3),K8(9,4),lee:7x7x7x7:2'
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from coldcore.backends import parse_cell, split_cells   # noqa: E402
from coldcore.gpu import GpuBackend, default_plugin_path  # noqa: E402


class Mismatch(AssertionError):
    pass


def same(tag, a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.array_equal(a, b):
        bad = np.flatnonzero(a.ravel() != b.ravel())[:5] if a.shape == b.shape \
            else "shape"
        raise Mismatch(f"{tag}: mismatch (first bad {bad})")


def ext_set(be, thr, cap):
    idx, val, found = be.map_extract(thr, cap=cap)
    o = np.argsort(idx, kind="stable")
    return np.asarray(idx)[o], np.asarray(val)[o], found


def compare_map(tag, A, B, rng, full_cap):
    size = A.space.size
    same(f"{tag} map_max", A.map_max(), B.map_max())
    mx = A.map_max()
    if mx > 0:
        same(f"{tag} hist", A.map_hist(1024, mx), B.map_hist(1024, mx))
        for thr in sorted({1, max(1, mx // 2), mx}):
            cap = min(full_cap, A.extract_cap)
            ia, va, fa = ext_set(A, thr, cap)
            ib, vb, fb = ext_set(B, thr, cap)
            same(f"{tag} extract found thr={thr}", fa, fb)
            if fa <= cap:
                same(f"{tag} extract idx thr={thr}", ia, ib)
                same(f"{tag} extract val thr={thr}", va, vb)
    k = min(size, 1 << 16)
    probe = rng.choice(size, size=k, replace=False).astype(np.int64)
    same(f"{tag} read_at", A.map_read_at(probe), B.map_read_at(probe))
    if size <= (1 << 22):
        every = np.arange(size, dtype=np.int64)
        same(f"{tag} read_at(all)", A.map_read_at(every),
             B.map_read_at(every))


def compare_cnt(tag, A, B):
    same(f"{tag} cnt", A.read_cnt(), B.read_cnt())
    for t in (0, 1, 2):
        same(f"{tag} count_eq({t})", A.count_eq(t), B.count_eq(t))
    same(f"{tag} uncovered", A.uncovered(), B.uncovered())


def run_cell(spec, base, cand, seed, M, steps, full_cap, near=False):
    problem, axes, R, mu = parse_cell(spec)
    rng = np.random.default_rng(seed)
    A = GpuBackend(axes, R, problem=problem, mu=mu, plugin_path=base)
    B = GpuBackend(axes, R, problem=problem, mu=mu, plugin_path=cand)
    size = A.space.size
    try:
        code = rng.choice(size, size=min(M, size // 2), replace=False)
        code = np.sort(code).astype(np.int64)
        if near:
            # a greedy cover minus a few words: tiny deficient set
            from coldcore.search import Searcher
            S = Searcher(A, seed=seed)
            S.load(np.zeros(0, dtype=np.int64))
            S.greedy_fill(10 ** 9)
            cov = np.asarray(S.code, dtype=np.int64)
            code = np.sort(rng.choice(cov, size=max(1, len(cov) - 3),
                                      replace=False))
        A.recount(code)
        B.recount(code)
        compare_cnt("recount", A, B)
        live = list(code)
        n_ops = 0
        for step in range(steps):
            # ---- mutations: single words and batches, both signs
            for nb in (1, 1, 3, 64, 1500):
                if nb > 1 and len(live) < nb + 10:
                    continue
                if near and nb > 3:
                    continue            # keep the deficient set small
                if rng.random() < 0.5 and len(live) > nb:
                    pick = rng.choice(len(live), size=nb, replace=False)
                    w = np.array([live[i] for i in pick], dtype=np.int64)
                    for i in sorted(pick, reverse=True):
                        live.pop(i)
                    A.ball_update(w, -1)
                    B.ball_update(w, -1)
                else:
                    w = rng.integers(0, size, size=nb).astype(np.int64)
                    live.extend(w.tolist())
                    A.ball_update(w, +1)
                    B.ball_update(w, +1)
                n_ops += 1
                # gather right after an (async) update: ordering check
                g = rng.integers(0, size, size=5).astype(np.int64)
                same(f"step{step} gather-after-update",
                     A.ball_gather(g, 0), B.ball_gather(g, 0))
            compare_cnt(f"step{step} updates", A, B)
            # ---- gathers
            for nb in (1, 7, 64, 300, 4096):
                cand_w = rng.integers(0, size, size=nb).astype(np.int64)
                if rng.random() < 0.3 and live:
                    cand_w[:min(nb, len(live))] = rng.choice(
                        np.array(live, dtype=np.int64), size=min(nb, len(live)))
                for role in (0, 1):
                    same(f"step{step} gather role{role} nb={nb}",
                         A.ball_gather(cand_w, role),
                         B.ball_gather(cand_w, role))
                    n_ops += 1
            # ---- fields
            A.gain_map()
            B.gain_map()
            compare_map(f"step{step} gain", A, B, rng, full_cap)
            A.loss_map()
            B.loss_map()
            compare_map(f"step{step} loss", A, B, rng, full_cap)
            # ---- recount from the live multiset (dedup: a set code)
            if step % 2 == 1:
                code = np.unique(np.array(live, dtype=np.int64))
                live = list(code)
                A.recount(code)
                B.recount(code)
                compare_cnt(f"step{step} recount", A, B)
        st = ""
        if hasattr(B._lib, "dct_stats"):
            import ctypes
            buf = (ctypes.c_longlong * 8)()
            B._lib.dct_stats(buf, 8)
            st = (f" [cand: set-side chunks {buf[0]}, walk chunks {buf[1]}, "
                  f"scans {buf[2]}, overflows {buf[3]}]")
        return f"{n_ops} ops{st}"
    finally:
        A.close()
        B.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="baseline plugin .so")
    ap.add_argument("--cand", default=None,
                    help="candidate plugin .so (default: this tree's build)")
    ap.add_argument("--cells", default="K3(8,1),K4(8,2),K5(8,3),K6(8,3)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--full-cap", type=int, default=1 << 22)
    args = ap.parse_args(argv)
    cand = args.cand or default_plugin_path()
    bad = 0
    for spec in split_cells(args.cells):
        for near in (False, True):
            tag = f"{spec} ({'near-cover' if near else 'random'} start)"
            try:
                n = run_cell(spec, args.base, cand, args.seed, args.M,
                             args.steps, args.full_cap, near=near)
                print(f"PASS {tag}: {n}, all identical", flush=True)
            except Mismatch as e:
                bad += 1
                print(f"FAIL {tag}: {e}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

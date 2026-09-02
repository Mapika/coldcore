"""The explicit-incidence backend family.

(a) SparseIncidenceBackend (host) against an independent brute-force
    oracle on random incidences and random solutions, mu in {1, 2};
(b) GPU vs host parity for every query (marked gpu);
(c) the unchanged search core on covering designs, reaching known
    optima on the host backend;
(d) the LJCR file format round-trip and the rank helpers.
"""
import hashlib
import itertools
import os
import time
from math import comb

import numpy as np
import pytest

from coldcore.incidence import IncidenceSpace, SparseIncidenceBackend
from coldcore.problems.covering_design import (CoveringDesignBackend,
                                               all_blocks, binom_table,
                                               build_incidence, rank_subsets,
                                               read_cover, unrank_subsets,
                                               write_cover)
from coldcore.reference import RefCoveringBackend
from coldcore.search import Searcher


# ------------------------------------------------------------ the oracle
class Oracle:
    """Pure-Python, dict-of-lists brute force over an incidence list."""

    def __init__(self, inc, n_univ, mu):
        self.rows = [sorted(int(u) for u in r) for r in inc]
        self.n_univ, self.mu = int(n_univ), int(mu)

    def cnt(self, sol):
        c = [0] * self.n_univ
        for w in sol:
            for u in self.rows[int(w)]:
                c[u] += 1
        return np.array(c, dtype=np.int64)

    def role(self, c, target):
        if target == 0:
            return c < self.mu
        if target == 1:
            return 1 <= c <= self.mu
        return c == target

    def gather(self, sol, cands, target):
        c = self.cnt(sol)
        return np.array([sum(self.role(int(c[u]), target)
                             for u in self.rows[int(w)]) for w in cands])

    def field(self, sol, target):
        return self.gather(sol, range(len(self.rows)), target)


def random_incidence(rng, n_cand, d, n_univ):
    return np.stack([np.sort(rng.choice(n_univ, d, replace=False))
                     for _ in range(n_cand)]).astype(np.int32)


CASES = [(60, 4, 40, 1), (120, 7, 80, 2), (25, 3, 9, 1), (200, 10, 300, 2)]


# ------------------------------------------------------------ (a) host
@pytest.mark.parametrize("n_cand,d,n_univ,mu", CASES)
def test_host_backend_matches_oracle(n_cand, d, n_univ, mu):
    rng = np.random.default_rng(n_cand * 7 + d)
    inc = random_incidence(rng, n_cand, d, n_univ)
    o = Oracle(inc, n_univ, mu)
    b = SparseIncidenceBackend(inc, n_univ, mu=mu, extract_cap=32)
    assert isinstance(b.space, IncidenceSpace) and b.space.size == n_cand
    for _ in range(4):
        M = int(rng.integers(0, n_cand // 2 + 1))
        sol = np.sort(rng.choice(n_cand, M, replace=False))
        b.recount(sol)
        c = o.cnt(sol)
        assert (b.cnt == c).all()
        assert (b.read_cnt() == c.astype(np.uint16)).all()
        assert (b.read_cnt(3, 5) == c[3:8]).all()
        assert b.uncovered() == int(np.maximum(mu - c, 0).sum())
        for t in (0, 1, 2, 3):
            assert b.count_eq(t) == int((c == t).sum())
        b.gain_map()
        assert (b.map == o.field(sol, 0)).all()
        b.loss_map()
        loss = o.field(sol, 1)
        assert (b.map == loss).all()
        assert b.map_max() == int(loss.max(initial=0))
        # map queries on the loss field
        mx = max(1, b.map_max())
        for nbins in (1, 5, 64):
            h = np.zeros(nbins, dtype=np.int64)
            for v in loss[loss > 0]:
                h[min((int(v) - 1) * nbins // mx, nbins - 1)] += 1
            assert (b.map_hist(nbins, mx) == h).all()
        for thr in (0, 1, mx):
            idx, val, found = b.map_extract(thr)
            assert found == int((loss >= thr).sum())
            assert len(idx) == min(found, 32)
            assert (loss[idx] == val).all()
        assert (b.map_read_at(sol) == loss[sol]).all()
        assert (b.map_all() == loss).all()
        # exact gathers (independent of the map)
        probe = rng.choice(n_cand, 11)
        for t in (0, 1, 2):
            assert (b.ball_gather(probe, t) == o.gather(sol, probe, t)).all()
        assert b.ball_gather(probe[:0], 0).shape == (0,)
        # incremental updates, mixed sizes and duplicates
        add = rng.choice(n_cand, 3)
        b.ball_update(add, +1)
        big = rng.choice(n_cand, n_cand // 2)
        b.ball_update(big, +1)
        b.ball_update(big, -1)
        b.ball_update(np.array([add[0]]), -1)
        c2 = o.cnt(list(sol) + list(add[1:]))
        assert (b.cnt == c2).all()
        # verification does not read the fields
        b.cnt[:] = 99
        assert b.verify(sol) == bool((c >= mu).all())
        assert (b.coverage(sol) == c).all()
        assert (b.ball_indices(add[0]) == np.sort(inc[add[0]])).all()


def test_host_backend_validation():
    inc = np.array([[0, 1], [1, 2]])
    with pytest.raises(ValueError):
        SparseIncidenceBackend(inc, 2)            # entry 2 out of range
    with pytest.raises(ValueError):
        SparseIncidenceBackend(inc, 3, mu=0)
    with pytest.raises(ValueError):
        SparseIncidenceBackend(inc.ravel(), 3)
    b = SparseIncidenceBackend(inc, 3)
    assert b.inc.dtype == np.int32 and b.d == 2 and b.n_cand == 2
    sp = b.space
    assert sp.n == 1 and sp.axes == [2] and (sp.pow_q == [1]).all()
    assert (sp.index_word(np.array([1, 0])) == [[1], [0]]).all()
    assert (sp.word_index(sp.index_word([1, 0])) == [1, 0]).all()
    assert sp.all_words().shape == (2, 1)


def test_extract_shuffle_mimics_gpu_order():
    rng = np.random.default_rng(3)
    inc = random_incidence(rng, 50, 3, 30)
    b = SparseIncidenceBackend(inc, 30, extract_cap=8, seed_shuffle=1)
    b.recount(np.arange(5))
    b.gain_map()
    idx, val, found = b.map_extract(1)
    assert found > 8 and len(idx) == 8 and (b.map[idx] == val).all()


# ------------------------------------------------------------ (b) GPU
def _sparse_plugin():
    from coldcore.incidence_gpu import default_sparse_plugin_path
    p = default_sparse_plugin_path()
    if not os.path.exists(p):
        pytest.skip(f"sparse plugin not built: {p}")
    return p


@pytest.mark.gpu
@pytest.mark.parametrize("n_cand,d,n_univ,mu", CASES)
@pytest.mark.parametrize("inc_mode", [0, 1])
def test_gpu_matches_host(gpu_backend_factory, n_cand, d, n_univ, mu,
                          inc_mode):
    from coldcore.incidence_gpu import GpuIncidenceBackend
    _sparse_plugin()
    rng = np.random.default_rng(n_cand + 31 * d + inc_mode)
    inc = random_incidence(rng, n_cand, d, n_univ)
    cap = 16
    c = SparseIncidenceBackend(inc, n_univ, mu=mu, extract_cap=cap)
    with GpuIncidenceBackend(inc, n_univ, mu=mu, extract_cap=cap,
                             inc_mode=inc_mode) as g:
        assert g.inc_mode in (0, 1, 2) and g.space.size == n_cand
        for _ in range(3):
            M = int(rng.integers(0, n_cand // 2 + 1))
            sol = np.sort(rng.choice(n_cand, M, replace=False))
            c.recount(sol)
            g.recount(sol)
            assert (c.read_cnt() == g.read_cnt()).all()
            assert (c.read_cnt(2, 6) == g.read_cnt(2, 6)).all()
            assert (c.read_cnt(n_univ - 2, 9) == g.read_cnt(n_univ - 2, 9)).all()
            assert len(g.read_cnt(n_univ + 3, 2)) == len(c.read_cnt(n_univ + 3, 2)) == 0
            assert c.uncovered() == g.uncovered()
            for t in (0, 1, 2, 3):
                assert c.count_eq(t) == g.count_eq(t)
            every = np.arange(n_cand)
            for fld in ("gain_map", "loss_map"):
                getattr(c, fld)()
                getattr(g, fld)()
                assert (c.map_read_at(every) == g.map_read_at(every)).all()
                assert c.map_max() == g.map_max()
                mx = max(1, c.map_max())
                for nbins in (1, 7, 1024):
                    assert (c.map_hist(nbins, mx) == g.map_hist(nbins, mx)).all()
                for thr in (0, 1, mx // 2, mx):
                    ci, cv, cf = c.map_extract(thr)
                    gi, gv, gf = g.map_extract(thr)
                    assert cf == gf                     # true total
                    assert len(gi) == min(gf, cap)      # capped return
                    assert (c.map[gi] == gv).all()      # right values
                    if gf <= cap:                       # same set, any order
                        assert sorted(ci.tolist()) == sorted(gi.tolist())
            probe = rng.choice(n_cand, 21)
            for t in (0, 1, 2, 5):
                assert (c.ball_gather(probe, t) == g.ball_gather(probe, t)).all()
            assert g.ball_gather(probe[:0], 0).shape == (0,)
            add = rng.choice(n_cand, 7)
            c.ball_update(add, +1)
            g.ball_update(add, +1)
            c.ball_update(add[:3], -1)
            g.ball_update(add[:3], -1)
            assert (c.read_cnt() == g.read_cnt()).all()
            assert g.verify(sol) == c.verify(sol)


@pytest.mark.gpu
def test_gpu_covering_design_search(gpu_backend_factory):
    """End-to-end on the GPU backend: C(13,4,2) = 13."""
    _sparse_plugin()
    with CoveringDesignBackend(13, 4, 2, device="gpu") as b:
        assert b.inc is None                      # host copy released
        s = Searcher(b, seed=1, deadline=time.time() + 90)
        s.load(np.zeros(0, dtype=np.int64))
        assert s.greedy_fill(10 ** 6) == 0
        M = s.descend(floor=13, notch_budget_s=20, step0=1)
        assert M <= 13 and s.verify_full() == 0
        assert b.verify(s.code)


# ------------------------------------------------------------ (c) search
KNOWN = [(7, 3, 2, 7), (9, 3, 2, 12), (8, 4, 3, 14), (13, 4, 2, 13)]


@pytest.mark.parametrize("v,k,t,opt", KNOWN)
def test_search_reaches_known_optimum(v, k, t, opt):
    b = CoveringDesignBackend(v, k, t)
    assert b.n_cand == comb(v, k) and b.n_univ == comb(v, t)
    s = Searcher(b, seed=1, deadline=time.time() + 120)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    assert s.verify_full() == 0
    M = s.descend(floor=opt, notch_budget_s=20, step0=1)
    assert M <= opt and len(s.code) == M
    assert s.verify_full() == 0
    assert b.verify(s.code)
    assert len(np.unique(b.rank_block(s.code), axis=0)) == len(s.code)


def test_mu_fold_design_search():
    """Every pair twice with 3-subsets of a 7-set: 14 blocks suffice
    (two copies of the Fano plane) and the search finds at most that."""
    b = CoveringDesignBackend(7, 3, 2, mu=2)
    s = Searcher(b, seed=2, deadline=time.time() + 90)
    s.load(np.zeros(0, dtype=np.int64))
    assert s.greedy_fill(10 ** 6) == 0
    M = s.descend(floor=14, notch_budget_s=20, step0=1)
    assert M <= 14 and s.verify_full() == 0
    assert b.verify(s.code)
    assert len(np.unique(s.code)) == len(s.code)


# ------------------------------------------------------------ (d) files
def test_rank_helpers_agree_with_itertools():
    for v, k in [(5, 1), (7, 3), (9, 4), (10, 10)]:
        lex = np.array(list(itertools.combinations(range(v), k)))
        r = rank_subsets(lex, C=binom_table(v, k))
        assert sorted(r.tolist()) == list(range(comb(v, k)))
        assert (unrank_subsets(r, v, k) == lex).all()
        blocks = all_blocks(v, k)
        assert (blocks[r] == lex).all()
        # colex: rank increases with the largest element first
        assert (np.diff(blocks[:, -1].astype(int)) >= 0).all()
    # shuffled columns rank like sorted ones
    assert rank_subsets([[4, 0, 2]], v=5)[0] == rank_subsets([[0, 2, 4]], v=5)[0]
    with pytest.raises(ValueError):
        unrank_subsets([comb(7, 3)], 7, 3)


def test_incidence_build_matches_itertools():
    v, k, t = 8, 4, 2
    blocks = all_blocks(v, k)
    inc = build_incidence(blocks, v, t, rows=7)      # several chunks
    Ct = binom_table(v, t)
    for i in [0, 1, 17, len(blocks) - 1]:
        T = np.array(list(itertools.combinations(blocks[i].tolist(), t)))
        assert sorted(inc[i].tolist()) == sorted(rank_subsets(T, C=Ct).tolist())


def test_cover_file_round_trip(tmp_path):
    b = CoveringDesignBackend(9, 3, 2)
    rng = np.random.default_rng(5)
    idx = np.sort(rng.choice(b.n_cand, 12, replace=False))
    p = str(tmp_path / "c.txt")
    b.write_cover(p, idx)
    lines = open(p).read().splitlines()
    assert len(lines) == 12
    for ln in lines:
        vals = [int(x) for x in ln.split()]
        assert len(vals) == 3 and vals == sorted(vals) and min(vals) >= 1
        assert max(vals) <= 9
    assert (b.read_cover(p) == idx).all()
    # module-level functions, blank lines and comments
    with open(p, "a") as fh:
        fh.write("\n# trailing comment\n")
    assert (read_cover(p, 9, 3) == idx).all()
    write_cover(p, b.rank_block(idx))
    assert (read_cover(p, 9, 3) == idx).all()
    # the CLI hooks are the same functions
    assert (b.read_solution_file(p) == idx).all()
    with pytest.raises(ValueError):
        read_cover(p, 9, 4)
    bad = str(tmp_path / "bad.txt")
    open(bad, "w").write("1 2 10\n")
    with pytest.raises(ValueError):
        read_cover(bad, 9, 3)
    open(bad, "w").write("1 2 2\n")
    with pytest.raises(ValueError):
        read_cover(bad, 9, 3)
    open(bad, "w").write("# nothing\n\n")
    assert read_cover(bad, 9, 3).shape == (0,)


def test_covering_design_backend_dispatch():
    b = CoveringDesignBackend(7, 3, 2, device="ref")
    assert isinstance(b, CoveringDesignBackend)
    assert isinstance(b, SparseIncidenceBackend)
    assert b.label == "C(7,3,2)" and b.d == 3
    assert CoveringDesignBackend(7, 3, 2, mu=2).label == "C(7,3,2)mu2"
    with pytest.raises(ValueError):
        CoveringDesignBackend(7, 3, 2, device="tpu")
    with pytest.raises(ValueError):
        CoveringDesignBackend(3, 4, 2)
    with pytest.raises(ValueError):
        b.block_rank([[0, 1, 9]])
    assert b.block_rank([2, 0, 1])[0] == 0


# ------------------------------------------------------------ (e) locality hooks
def test_block_metric_hooks():
    b = CoveringDesignBackend(9, 4, 2)
    v, k, t = b.v, b.k, b.t
    rng = np.random.default_rng(9)
    idx = rng.choice(b.n_cand, 40)
    # distance: symmetric-difference/2, zero on the diagonal, symmetric
    D = b.distance(idx[:, None], idx[None, :])
    assert D.shape == (40, 40) and (D == D.T).all() and (np.diag(D) == 0).all()
    A = b.blocks[idx].astype(int)
    for i in range(40):
        for j in range(40):
            assert D[i, j] == len(set(A[i]) - set(A[j]))
    assert (b.distance(idx, idx[0]) == D[:, 0]).all()
    # neighbours at radius 1: all k(v-k) single swaps, all at distance 1
    n1 = b.neighbours(int(idx[0]), 1)
    assert len(n1) == k * (v - k) == len(np.unique(n1))
    assert (b.distance(n1, idx[0]) == 1).all() and int(idx[0]) not in n1
    assert len(b.neighbours(int(idx[0]), 1, m=7, rng=rng)) == 7
    n2 = b.neighbours(int(idx[0]), 2, m=50, rng=rng)
    assert 0 < len(n2) <= 50 and (b.distance(n2, idx[0]) == 2).all()
    assert len(b.neighbours(int(idx[0]), 99)) == 0
    # universe distance is 0 exactly on the incidence
    u = rng.choice(b.n_univ, 6)
    ud = b.universe_distance(idx, u)
    assert ud.shape == (40, 6) and ud.min() >= 0 and ud.max() <= t
    for i in range(40):
        for j in range(6):
            assert (ud[i, j] == 0) == (u[j] in b.inc[idx[i]])
    # candidates_covering agrees with the incidence matrix
    for uu in u[:3]:
        cc = b.candidates_covering(int(uu))
        assert len(cc) == comb(v - t, k - t)
        assert (cc == np.flatnonzero((b.inc == uu).any(axis=1))).all()
    # independence threshold: farther than k-t -> no shared t-subset
    assert b.indep_dist == k - t
    for i in range(40):
        for j in range(40):
            shared = bool(set(b.inc[idx[i]]) & set(b.inc[idx[j]]))
            if D[i, j] > b.indep_dist:
                assert not shared
    # deficient_universe reads the field exactly (4 blocks leave most
    # of the 36 pairs deficient)
    sol = rng.choice(b.n_cand, 4, replace=False)
    b.recount(sol)
    want = np.flatnonzero(b.cnt < b.mu)
    assert (b.deficient_universe(10 ** 6) == want).all()
    assert (b.deficient_universe(5) == want[:5]).all()
    sub = b.deficient_universe(5, rng)
    assert len(sub) == 5 and np.isin(sub, want).all()


@pytest.mark.gpu
def test_gpu_hooks_match_host(gpu_backend_factory):
    _sparse_plugin()
    c = CoveringDesignBackend(9, 4, 2)
    with CoveringDesignBackend(9, 4, 2, device="gpu") as g:
        rng = np.random.default_rng(2)
        sol = rng.choice(c.n_cand, 15, replace=False)
        c.recount(sol)
        g.recount(sol)
        assert (c.deficient_universe(10 ** 6) == g.deficient_universe(10 ** 6)).all()
        assert (c.distance(sol, sol[0]) == g.distance(sol, sol[0])).all()
        assert g.indep_dist == c.indep_dist == 2


def _locality_trace(mk, seed, rounds=12):
    """greedy -> peel -> cluster/hole/deficit ruins -> a shell polish to
    feasibility: every code path the locality hooks touch."""
    b = mk()
    s = Searcher(b, seed=seed)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    s.peel()
    tr = [(int(s.uncov), s.code.tolist())]
    target = max(1, len(s.code) - 2)
    modes = ["cluster", "hole", "deficit"]
    for i in range(rounds):
        s.ruin(1 + i % 3, modes[i % 3])
        s.greedy_fill(target)
        tr.append((int(s.uncov), s.code.tolist()))
    M = len(s.code) + 2
    rng = np.random.default_rng(seed + 5)
    s.load(rng.choice(b.space.size, size=M, replace=False).astype(np.int64))
    s.anneal(M, budget_s=600.0, mode="lahc", seed=seed + 1,
             neighbourhood="shell", log_every=0, pool_refresh=64,
             restart_stale=400)
    assert s.uncov == 0
    tr.append((int(s.stats["shift_moves"]), s.code.tolist()))
    return tr


def _digest(tr):
    h = hashlib.sha256()
    for u, code in tr:
        h.update(f"{u}|{','.join(map(str, code))};".encode())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("q,n,R,seed,pin", [
    (3, 4, 1, 4, "1db1c1a48386dfff"),
    (2, 6, 1, 11, "25fb925641a8454a"),
])
def test_product_space_unchanged_without_hooks(q, n, R, seed, pin):
    """Digests taken before the locality hooks existed (commit 21c70a5):
    a product-space backend defines none, so the trajectory is
    bit-identical."""
    b = RefCoveringBackend(q, n, R)
    assert b.distance is None and b.neighbours is None
    assert _digest(_locality_trace(lambda: b, seed)) == pin


def _descend_with(b, opt, modes, polish, seed, deadline_s):
    """greedy -> notch ladder to ``opt`` with a fixed ruin operator mix
    (and optionally a shell polish on a walled notch). Returns the
    seconds to the optimum, or None."""
    s = Searcher(b, seed=seed, deadline=time.time() + deadline_s)
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    s.peel()
    t0 = time.time()
    M = len(s.code)
    while M > opt and s.time_left() > 1:
        target = M - 1
        s.ruin(len(s.code) - target, "low")
        r = 0
        while s.uncov > 0 and s.time_left() > 1:
            pre = s.uncov
            s.ruin(int(s.rng.integers(2, max(3, M // 8))), modes[r % len(modes)])
            s.greedy_fill(target)
            if s.uncov > pre and s.rng.random() > 0.1:
                s.load(s.best_code)
            else:
                s._snapshot()
            r += 1
            if polish and r % 200 == 0 and s.uncov > 0:
                s.anneal(target, 2.0, mode="lahc", neighbourhood="shell",
                         log_every=0)
        if s.uncov > 0:
            return None
        assert s.verify_full() == 0
        M = len(s.code)
    return time.time() - t0 if M <= opt else None


HOOK_OPS = [("deficit",), ("low", "hole", "cluster", "deficit")]


@pytest.mark.parametrize("v,k,t,opt", [(13, 4, 2, 13), (10, 4, 3, 30)])
def test_locality_ruins_reach_optimum(v, k, t, opt):
    """With the block metric the exact deficit ruin (and the standard
    mix that contains it) reach the optimum no later than random ruin
    does — on C(10,4,3) random ruin does not get there within the cap
    at all, the deficit ruin in well under a second."""
    b = CoveringDesignBackend(v, k, t)
    cap = 30.0
    t_rand = _descend_with(b, opt, ["random"], polish=False, seed=3,
                           deadline_s=cap)
    for modes in HOOK_OPS:
        secs = _descend_with(b, opt, list(modes), polish=False, seed=3,
                             deadline_s=cap)
        assert secs is not None, modes
        if t_rand is not None:
            assert secs <= max(t_rand, 1.0), (modes, secs, t_rand)
    # cluster / hole ruins and the shell polish run and keep feasibility
    for modes in (["cluster"], ["hole"]):
        assert _descend_with(CoveringDesignBackend(13, 4, 2), 13, modes,
                             polish=True, seed=3, deadline_s=cap) is not None

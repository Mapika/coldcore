"""Search core: exact lazy greedy + ruin-and-recreate LNS + notch descent.

Problem-agnostic port of the prototype engine that set the
covering-code records reported in the companion paper.
Runs against any coldcore.protocol.Backend — GPU plugin or numpy
reference alike.

The pass library, in the order a descent uses it:

  greedy_fill  exact lazy greedy build / repair
  ruin         remove k incumbents (low / hole / cluster / random /
               deficit — the last one targeted at the actual uncovered
               words)
  lns          ruin-and-recreate rounds at fixed M, fixed or adaptive
               (bandit) ruin-operator selection
  anneal       Metropolis or late-acceptance polish at fixed M — accepts
               coverage-degrading swaps, so it can walk off the
               "uncovered = 1..3" walls that pure repair stalls on;
               neighbourhood="shell" costs a whole *local* candidate
               batch per move, which is what actually moves on a wall
  two_swap     exact, budget-capped depth-2 (add, drop) search for the
               endgame, where no single swap improves
  portfolio    time-sliced restarts at fixed M with a shared incumbent
  peel         drop zero-loss incumbents in independent batches
  break_wall   the endgame driver: focused-ruin LNS / polish / two-swap
               bursts at one fixed M, with restart-from-best
  descend      the notch ladder that drives all of the above

Every pass is exact: coverage is always read back from the backend's
exact fields (`ball_gather`, `count_eq`), never approximated.

Callbacks (all optional):
  on_log(str)                    — progress lines
  on_improve(searcher)           — new best-by-uncovered snapshot
  on_solved(M, idx_array)        — a feasible solution at size M was
                                   verified by full recount; hook record
                                   gates / checkpoint writers here
  yield_check()                  — called periodically; long-running GPU
                                   cooperation (pause/yield) goes here
"""
import contextlib
import heapq
import time

import numpy as np

#: The four proven ruin operators, and the relative frequencies the
#: record runs used for them (``hole`` twice as often as the rest).
#: Frozen: this tuple is what the record runs used, and what the
#: portfolio seeds its streams from.
RUIN_MODES = ("low", "hole", "cluster", "random")
RUIN_MIX = {"low": 1.0, "hole": 2.0, "cluster": 1.0, "random": 1.0}

#: The proven four plus ``deficit`` — the wall-breaking operator that
#: ruins the incumbents nearest the *actual* uncovered words. Opt-in
#: (``lns(..., deficit=True)``); never used unless asked for.
RUIN_MODES_EXT = RUIN_MODES + ("deficit",)
RUIN_MIX_EXT = dict(RUIN_MIX, deficit=2.0)

#: Fixed-selection draw lists: the proven one, and the deficit-enabled
#: one (deficit weighted like ``hole``).
_FIXED_MODES = ["low", "hole", "hole", "cluster", "random"]
_FIXED_MODES_DEFICIT = _FIXED_MODES + ["deficit", "deficit"]

#: Cap on the repair-candidate pool the polish pass extracts per refresh.
_POOL_CAP = 1 << 14

#: How many uncovered words the ``deficit`` operator aims at, and how
#: many words a single scan of ``cnt`` reads per pass.
_DEFICIT_WORDS = 4
_SCAN_CHUNK = 1 << 24

#: Local-neighbourhood polish defaults (``anneal(neighbourhood="shell")``).
#: The candidate batch per move is the moved word's full distance-1
#: shell, ``_SHELL2`` sampled distance-2 words, ``_SHELL_TOP`` draws from
#: the top of the gain map and ``_SHELL_RAND`` uniform draws; the word to
#: move is drawn from the ``_SHELL_TOPK`` cheapest incumbents.
_SHELL2 = 128
_SHELL_TOP = 48
_SHELL_RAND = 8
_SHELL_TOPK = 24
_SHELL_POOL_CAP = 4096


class OperatorBandit:
    """Reward-proportional (roulette-wheel) ruin-operator selection.

    The classic adaptive-LNS scheme: every operator carries a score that
    is an exponential moving average of the reward its rounds earned,
    and the next operator is drawn proportionally to that score, mixed
    with a uniform ``floor`` so a temporarily unlucky operator is never
    starved. Scores start at :data:`RUIN_MIX` (normalised to mean 1), so
    the selector *begins* at the hand-tuned mix the records were set
    with and only moves away from it when the instance says otherwise.

    Rewards are tiered by what a round achieved (new global best >
    improved on the round's start > sideways > worse).
    """

    REWARD_BEST = 3.0
    REWARD_BETTER = 1.0
    REWARD_EQUAL = 0.3
    REWARD_WORSE = 0.0

    def __init__(self, rng, ops=RUIN_MODES, floor=0.15, decay=0.9,
                 prior=RUIN_MIX):
        self.ops = list(ops)
        self.rng = rng
        self.floor = float(floor)
        self.decay = float(decay)
        w = np.array([float(prior.get(o, 1.0)) for o in self.ops])
        self.score = w * (len(w) / w.sum())          # mean 1, reward scale
        self.uses = np.zeros(len(self.ops), dtype=np.int64)
        self.reward = np.zeros(len(self.ops), dtype=np.float64)

    def probabilities(self, boost=None):
        """Selection distribution: floor-mixed, score-proportional.

        ``boost`` is an optional ``{op: factor}`` *hint* applied to the
        scores before normalising — the driver's way of saying "the
        state we are in is the one this operator was built for" without
        overwriting what the rewards have learned. ``boost=None`` (the
        default everywhere) is the plain reward-proportional rule."""
        w = np.maximum(self.score, 0.0)
        k = len(self.ops)
        if boost:
            w = w.copy()
            for op, f in boost.items():
                if op in self.ops:
                    i = self.ops.index(op)
                    # a boosted operator with a crushed score must still
                    # come back, so lift off the floor before scaling
                    w[i] = max(w[i], self.floor) * float(f)
        if w.sum() <= 0:
            return np.full(k, 1.0 / k)
        return self.floor / k + (1.0 - self.floor) * w / w.sum()

    def select(self, boost=None):
        i = int(self.rng.choice(len(self.ops), p=self.probabilities(boost)))
        self.uses[i] += 1
        return self.ops[i]

    def update(self, op, reward):
        i = self.ops.index(op)
        self.reward[i] += reward
        d = self.decay
        self.score[i] = d * self.score[i] + (1.0 - d) * reward

    def summary(self):
        """Per-operator {uses, mean reward, selection weight} — logged and
        parked in ``Searcher.stats['ops']`` so runs are auditable."""
        p = self.probabilities()
        return {o: {"uses": int(self.uses[i]),
                    "reward": float(self.reward[i] / max(1, self.uses[i])),
                    "p": float(p[i])}
                for i, o in enumerate(self.ops)}


class Searcher:
    """LNS driver for one problem instance on one backend."""

    def __init__(self, backend, seed=1, deadline=None, on_log=None,
                 on_improve=None, on_solved=None, yield_check=None):
        self.b = backend
        self.rng = np.random.default_rng(seed * 1000003 + 12345)
        self.deadline = deadline
        self.on_log = on_log or (lambda s: None)
        self.on_improve = on_improve or (lambda s: None)
        self.on_solved = on_solved or (lambda m, idx: None)
        self.yield_check = yield_check or (lambda: None)
        self.code = np.zeros(0, dtype=np.int64)   # current solution (indices)
        self.uncov = None
        self.best_code = None
        self.best_uncov = None
        self.stats = {"transforms": 0, "placements": 0, "removals": 0,
                      "rounds": 0, "recounts": 0, "moves": 0, "accepts": 0,
                      "restarts": 0}
        # deficit-word scan: rotating start offset + one-entry cache
        # keyed on the solution that produced it (cnt is a function of
        # the solution, so the key is exact).
        self._scan_off = 0
        self._deficit_cache = (None, None)
        self._sh1 = None            # per-axis shell-1 displacement tables

    # ------------------------------------------------ time / state
    def time_left(self):
        return np.inf if self.deadline is None else self.deadline - time.time()

    @contextlib.contextmanager
    def _rng_scope(self, seed):
        """Run a pass on its own random stream when given an explicit seed.

        ``seed=None`` (the default everywhere) keeps the searcher's own
        stream, so pass sequences stay bit-identical to a plain run;
        a seed makes that one pass reproducible on its own, including
        the sub-passes it calls."""
        if seed is None:
            yield self.rng
            return
        saved = self.rng
        self.rng = np.random.default_rng(seed * 1000003 + 12345)
        try:
            yield self.rng
        finally:
            self.rng = saved

    def load(self, code):
        """Accepts 1-D point indices or a 2-D (M, n) digit array."""
        arr = np.asarray(code)
        if arr.ndim == 2:
            arr = self.b.word_index(arr.astype(np.int64))
        self.code = (np.unique(arr.astype(np.int64)) if arr.size
                     else np.zeros(0, dtype=np.int64))
        self.b.recount(self.code)
        self.stats["transforms"] += 1
        self.uncov = self.b.uncovered()
        self._snapshot()

    def _snapshot(self):
        if self.best_uncov is None or self.uncov < self.best_uncov or (
                self.uncov == self.best_uncov and
                (self.best_code is None or len(self.code) < len(self.best_code))):
            self.best_uncov = self.uncov
            self.best_code = self.code.copy()
            self.on_improve(self)

    def verify_full(self):
        """From-scratch recount; returns exact uncovered count."""
        self.b.recount(self.code)
        self.stats["transforms"] += 1
        self.stats["recounts"] += 1
        self.uncov = self.b.uncovered()
        return self.uncov

    # ------------------------------------------------ recreate (lazy greedy)
    def greedy_fill(self, target_m, batch=64, refresh_every=512):
        """Exact lazy greedy until covered / target_m / deadline.

        Threshold slice of the exact gain map feeds a stale-bound
        max-heap; candidates are re-checked exactly (ball_gather) and a
        placement is accepted only when its exact gain beats the next
        stale bound. Returns final uncovered."""
        b = self.b
        codeset = set(self.code.tolist())
        placed_since = refresh_every          # force initial refresh
        heap = []                             # (-stale_gain, idx)
        thr = 1
        while self.uncov > 0 and len(codeset) < target_m:
            if self.time_left() < 5:
                break
            self.yield_check()
            if placed_since >= refresh_every or not heap:
                b.gain_map()
                self.stats["transforms"] += 1
                mx = b.map_max()
                if mx <= 0:
                    break
                # threshold: extraction bounded, but deep enough to
                # survive many placements before a refresh
                cap = min(b.extract_cap, 1 << 18)
                thr = 1
                if mx > 1 and getattr(b, "supports_hist", True):
                    hist = b.map_hist(1024, mx)
                    csum = np.cumsum(hist[::-1])[::-1]
                    good = np.flatnonzero(csum <= cap * 0.9)
                    bin0 = int(good[0]) if len(good) else 1023
                    thr = max(1, bin0 * mx // 1024 + 1)
                idx, val, found = b.map_extract(thr, cap=cap)
                while found > cap and thr < mx:
                    thr = min(mx, thr + max(1, (mx - thr) // 2))
                    idx, val, found = b.map_extract(thr, cap=cap)
                order = np.argsort(-val, kind="stable")
                # ``.tolist()`` once instead of per-element numpy scalar
                # indexing: same tuples, same order, same heap — but the
                # extraction is up to 2^18 entries, and doing it the
                # scalar way was the single hottest line of the build.
                nval = (-np.asarray(val, dtype=np.int64)[order]).tolist()
                nidx = np.asarray(idx, dtype=np.int64)[order].tolist()
                heap = [t for t in zip(nval, nidx) if t[1] not in codeset]
                heapq.heapify(heap)
                placed_since = 0
                if not heap:
                    if thr > 1:
                        placed_since = refresh_every   # retry at thr=1
                        thr = 1
                        continue
                    break
            # pop a batch of best stale bounds, evaluate exactly
            cand = []
            while heap and len(cand) < batch:
                v, i = heapq.heappop(heap)
                if i not in codeset:
                    cand.append((-v, i))
            if not cand:
                placed_since = refresh_every
                continue
            widx = np.fromiter((c[1] for c in cand), dtype=np.int64,
                               count=len(cand))
            exact = b.ball_gather(widx, 0)
            bi = int(np.argmax(exact))
            bg = int(exact[bi])
            ex = exact.tolist()               # Python ints once, not per pop
            next_bound = -heap[0][0] if heap else 0
            if bg < 1:
                placed_since = refresh_every
                continue
            if bg >= next_bound and bg >= thr:
                w = int(widx[bi])
                b.ball_update(np.array([w], dtype=np.int64), +1)
                codeset.add(w)
                self.uncov -= bg
                self.stats["placements"] += 1
                placed_since += 1
                for k, (v, i) in enumerate(cand):
                    if i != w and ex[k] > 0:
                        heapq.heappush(heap, (-ex[k], i))
            else:
                for k, (v, i) in enumerate(cand):
                    if ex[k] > 0:
                        heapq.heappush(heap, (-ex[k], i))
                if bg < thr:
                    placed_since = refresh_every   # refresh to stay greedy
        self.code = np.array(sorted(codeset), dtype=np.int64)
        return self.uncov

    # ------------------------------------------------ the deficit itself
    def supports_deficit(self):
        """True when the backend can hand back the ``cnt`` field, which
        is what turns "something is uncovered" into "*these words* are
        uncovered". ``read_cnt`` is an optional backend capability (both
        shipped backends have it); without it the deficit-targeted
        operators degrade to their untargeted cousins."""
        return hasattr(self.b, "read_cnt")

    def uncovered_words(self, limit=_DEFICIT_WORDS, chunk=_SCAN_CHUNK,
                        scan_cap=None):
        """Exact indices of up to ``limit`` currently deficient words.

        Scans the exact ``cnt`` field in chunks, from a rotating offset
        (so repeated calls at the same state see different members of a
        large deficit set), stopping as soon as ``limit`` are found. The
        result is cached against the solution that produced it: on a
        wall — where round after round is rejected back to the same
        incumbent — the scan runs once and every later call is free.

        ``scan_cap`` bounds the words read per call; ``None`` means "as
        many as it takes", which is a full-field read in the worst case.
        Returns a possibly-empty int64 array; empty means "not found
        within the cap", never "nothing is uncovered"."""
        if not self.supports_deficit() or self.uncov == 0:
            return np.zeros(0, dtype=np.int64)
        exact = getattr(self.b, "deficient_universe", None)
        if exact is not None:               # locality hook (protocol.py)
            return np.asarray(exact(limit, self.rng), dtype=np.int64)
        key = (len(self.code), self.uncov,
               hash(np.ascontiguousarray(self.code).tobytes()))
        ckey, cval = self._deficit_cache
        if ckey == key and cval is not None and len(cval) >= min(
                limit, self.uncov):
            return cval[:limit]
        size = self.b.space.size
        mu = getattr(self.b, "mu", 1)
        chunk = max(1, int(chunk))
        budget = size if scan_cap is None else min(size, int(scan_cap))
        found = []
        off = int(self._scan_off) % size
        read = 0
        while read < budget and len(found) < limit:
            take = min(chunk, size - off, budget - read)
            block = np.asarray(self.b.read_cnt(off, take))
            hit = np.flatnonzero(block < mu)
            if len(hit):
                take_n = hit[:limit - len(found)] + off
                found.extend(int(v) for v in take_n)
                self._scan_off = (int(take_n[-1]) + 1) % size
            read += take
            off = (off + take) % size
        if len(found) < limit:
            self._scan_off = 0          # exhausted: restart the sweep
        out = np.array(sorted(found), dtype=np.int64)
        if len(out):
            self._deficit_cache = (key, out)
        return out

    # ------------------------------------------------ ruin
    def losses(self):
        self.b.loss_map()
        self.stats["transforms"] += 1
        return self.b.map_read_at(self.code)

    def ruin(self, k, mode="low", deficit_pool=3):
        """Remove k solution points; returns the removed indices.

        Modes: ``low`` (cheapest by exact loss), ``cluster`` (near a
        random incumbent), ``hole`` (near a random high-gain point),
        ``random``, and ``deficit`` — k incumbents drawn from the
        ``deficit_pool``*k nearest to the words that are *actually*
        uncovered. ``hole`` aims at a random point of the deficit
        neighbourhood, which is up to R away from the deficit in an
        arbitrary direction; ``deficit`` aims at the deficient words
        themselves. It falls back to ``hole`` when the backend cannot
        report them."""
        m = len(self.code)
        k = min(k, m - 1)
        if k <= 0:
            return np.zeros(0, dtype=np.int64)
        if mode == "deficit":
            tgt = self.uncovered_words()
            if not len(tgt):
                return self.ruin(k, "hole")
            self.stats["deficit_ruins"] = self.stats.get("deficit_ruins", 0) + 1
            udist = getattr(self.b, "universe_distance", None)
            if udist is not None:           # locality hook (protocol.py)
                d = np.asarray(udist(self.code, tgt), dtype=np.int64).min(axis=1)
            else:
                digs = self.b.index_word(self.code).astype(np.int16)
                tw = self.b.index_word(tgt).astype(np.int16)
                # distance from every incumbent to the NEAREST deficient word
                d = np.full(m, np.iinfo(np.int32).max, dtype=np.int64)
                for t in tw:
                    np.minimum(d, (digs != t[None, :]).sum(axis=1), out=d)
            # Focus, then diversify. Taking the strict k nearest is what
            # the operator is "supposed" to do and it measurably *loses*
            # to `hole`: on a wall the deficient words are the same ones
            # round after round, so the strict rule ruins the same k
            # incumbents every time and the greedy repair rebuilds the
            # same solution — a no-op round. Drawing k out of the nearest
            # ~3k keeps the aim and restores the variety that makes a
            # ruin-and-recreate round worth running.
            order = np.argsort(d + self.rng.random(m))
            near = order[:min(m, max(deficit_pool * k, k + 8))]
            pick = self.rng.choice(near, size=k, replace=False)
        elif mode == "random":
            pick = self.rng.choice(m, size=k, replace=False)
        elif mode == "cluster":
            dist = getattr(self.b, "distance", None)
            if dist is not None:            # locality hook (protocol.py)
                d = np.asarray(dist(self.code, self.code[self.rng.integers(m)]))
            else:
                digs = self.b.index_word(self.code).astype(np.int16)
                c = digs[self.rng.integers(m)]
                d = (digs != c[None, :]).sum(axis=1)
            pick = np.argsort(d + self.rng.random(m))[:k]
        elif mode == "hole":
            # cluster the ruin near missing coverage: a top-gain point
            # sits inside/near the uncovered region
            self.b.gain_map()
            self.stats["transforms"] += 1
            mx = self.b.map_max()
            if mx <= 0:
                return self.ruin(k, "low")
            oi, ov, found = self.b.map_extract(max(1, mx // 2), cap=4096)
            if not len(oi):
                return self.ruin(k, "low")
            anchor = int(oi[self.rng.integers(len(oi))])
            dist = getattr(self.b, "distance", None)
            if dist is not None:            # locality hook (protocol.py)
                d = np.asarray(dist(self.code, anchor))
            else:
                c = self.b.index_word(np.array([anchor]))[0].astype(np.int16)
                digs = self.b.index_word(self.code).astype(np.int16)
                d = (digs != c[None, :]).sum(axis=1)
            pick = np.argsort(d + self.rng.random(m))[:k]
        else:  # "low": exact loss with exponential jitter
            lo = self.losses().astype(np.float64)
            jit = self.rng.exponential(scale=max(1.0, lo.mean() * 0.15), size=m)
            pick = np.argsort(lo + jit)[:k]
        removed = self.code[pick]
        self.b.ball_update(removed, -1)
        self.code = np.delete(self.code, pick)
        self.uncov = self.b.uncovered()
        self.stats["removals"] += int(k)
        return removed

    # ------------------------------------------------ peel
    def peel(self):
        """Remove zero-loss points (feasibility preserved) to fixpoint.

        Batched: one loss transform yields candidates, exact re-check
        filters them, and a pairwise-independent subset (digit distance
        > backend.indep_dist) is removed in one go."""
        b = self.b
        total = 0
        while True:
            if self.time_left() < 5:
                break
            lo = self.losses()
            zeros = np.flatnonzero(lo == 0)
            if not len(zeros):
                break
            exact = b.ball_gather(self.code[zeros], 1)
            zeros = zeros[exact == 0]
            if not len(zeros):
                break
            self.rng.shuffle(zeros)
            sep = b.indep_dist
            dist = getattr(b, "distance", None)
            if sep is not None and (dist is not None or sep < self.b.space.n):
                cand = self.code[zeros]
                digs = (None if dist is not None
                        else b.index_word(cand).astype(np.int16))
                sel = []
                for i in range(len(zeros)):
                    if not sel:
                        sel.append(i)
                        continue
                    if dist is not None:    # locality hook (protocol.py)
                        d = np.asarray(dist(cand[np.array(sel)], cand[i]))
                    else:
                        d = (digs[np.array(sel)] != digs[i][None, :]).sum(axis=1)
                    if (d > sep).all():
                        sel.append(i)
                pick = zeros[np.array(sel)]
            else:
                pick = zeros[:1]
            b.ball_update(self.code[pick], -1)
            mask = np.ones(len(self.code), dtype=bool)
            mask[pick] = False
            self.code = self.code[mask]
            total += len(pick)
            self.stats["removals"] += len(pick)
        if total:
            self.uncov = self.b.uncovered()
        return total

    # ------------------------------------------------ LNS at fixed M
    def lns(self, M, round_budget_s, kmin=8, kmax=None, log_every=20,
            select="fixed", seed=None, deficit=False, deficit_hint=3,
            deficit_boost=4.0):
        """Ruin/recreate rounds at fixed M until feasible or budget out.

        ``select="fixed"`` draws the ruin operator from the hand-tuned
        mix the records were set with (bit-identical to the original
        pass); ``select="adaptive"`` hands the choice to an
        :class:`OperatorBandit` that re-weights the operators by the
        rewards their rounds actually earn on *this* instance.

        ``deficit=True`` adds the deficit-targeted ruin operator to
        whichever selector is in use (weighted like ``hole`` in the
        fixed mix, given the ``hole`` prior in the bandit), and hints
        the selector towards it while the state is within
        ``deficit_hint`` of feasible — which is exactly the wall the
        operator exists for. ``deficit=False`` (the default) leaves both
        selectors bit-identical to before.

        Returns final uncovered (0 = solved)."""
        if select not in ("fixed", "adaptive"):
            raise ValueError(f"unknown ruin selection: {select!r}")
        with self._rng_scope(seed):
            return self._lns(M, round_budget_s, kmin, kmax, log_every, select,
                             bool(deficit), int(deficit_hint),
                             float(deficit_boost))

    def _lns(self, M, round_budget_s, kmin, kmax, log_every, select,
             deficit=False, deficit_hint=3, deficit_boost=4.0):
        t0 = time.time()
        kmax = kmax or max(kmin + 1, M // 20)
        self.best_uncov = None
        self.best_code = None
        if len(self.code) < M:
            self.greedy_fill(M)
        self._snapshot()
        modes = _FIXED_MODES_DEFICIT if deficit else _FIXED_MODES
        if select == "adaptive":
            bandit = (OperatorBandit(self.rng, ops=RUIN_MODES_EXT,
                                     prior=RUIN_MIX_EXT) if deficit
                      else OperatorBandit(self.rng))
        else:
            bandit = None
        # the hint: while the deficit is small the blocking structure is
        # local, and the targeted operator is the one built for it
        hint = ({"deficit": deficit_boost} if deficit else None)
        r = 0
        while self.uncov > 0:
            if time.time() - t0 > round_budget_s or self.time_left() < 10:
                break
            self.yield_check()
            pre = self.uncov
            pre_best = self.best_uncov
            k = int(self.rng.integers(kmin, kmax + 1))
            near = deficit and self.uncov <= deficit_hint
            if bandit is None:
                if near and self.rng.random() < 0.5:
                    mode = "deficit"
                else:
                    mode = modes[int(self.rng.integers(len(modes)))]
            else:
                mode = bandit.select(hint if near else None)
            self.ruin(k, mode)
            self.greedy_fill(M)
            self.stats["rounds"] += 1
            r += 1
            if self.uncov > pre and self.rng.random() > 0.10:
                # reject: restore best snapshot
                self.code = self.best_code.copy()
                self.b.recount(self.code)
                self.stats["transforms"] += 1
                self.uncov = self.b.uncovered()
            else:
                self._snapshot()
            if bandit is not None:
                if self.best_uncov < pre_best:
                    reward = bandit.REWARD_BEST
                elif self.uncov < pre:
                    reward = bandit.REWARD_BETTER
                elif self.uncov == pre:
                    reward = bandit.REWARD_EQUAL
                else:
                    reward = bandit.REWARD_WORSE
                bandit.update(mode, reward)
                self.stats["ops"] = bandit.summary()
            if log_every and r % log_every == 0:
                self.on_log(f"lns M={M} round={r} uncov={self.uncov} "
                            f"best={self.best_uncov} k={k} "
                            f"({time.time() - t0:.0f}s)")
            if r % 50 == 0:                    # periodic drift check
                u_inc = self.uncov
                u_full = self.verify_full()
                if u_inc != u_full:
                    self.on_log(f"WARNING drift: incremental {u_inc} != "
                                f"recount {u_full}; adopting recount")
        if self.uncov == 0:
            self._snapshot()
        elif self.best_code is not None and self.best_uncov < self.uncov:
            self.code = self.best_code.copy()
            self.b.recount(self.code)
            self.uncov = self.b.uncovered()
        return self.uncov

    # ------------------------------------------------ polish at fixed M
    def _gain_pool(self):
        """Repair candidates: every point with positive exact gain.

        One gain transform; a point has gain > 0 exactly when it covers
        something currently uncovered, so the pool is precisely the set
        of useful additions (and is disjoint from the incumbent)."""
        self.b.gain_map()
        self.stats["transforms"] += 1
        if self.b.map_max() <= 0:
            return np.zeros(0, dtype=np.int64)
        cap = min(_POOL_CAP, self.b.extract_cap)
        idx, _val, _found = self.b.map_extract(1, cap=cap)
        return np.asarray(idx, dtype=np.int64)

    @staticmethod
    def _auto_refresh(tr_s, move_s, share, lo=1, hi=1 << 16):
        """Refresh cadence that keeps pool refreshes to ``share`` of wall
        time: with a transform costing ``tr_s`` and an accepted move
        costing ``move_s``, refreshing every P accepted moves spends
        tr/(tr + P*move) of the clock on transforms, so

            P >= tr * (1 - share) / (share * move).

        Both inputs are measured, so this tracks the real machine: on a
        small GPU cell a transform is about the cost of one move and the
        cadence collapses to single digits (a much fresher pool than the
        fixed 64); on a cell where a transform is seconds it climbs into
        the thousands."""
        share = min(max(float(share), 1e-3), 0.999)
        move_s = max(float(move_s), 1e-9)
        p = tr_s * (1.0 - share) / (share * move_s)
        return int(min(max(round(p), lo), hi))

    # -------------------------------------- local (shell) neighbourhood
    def _shell_tables(self):
        """Per-axis displacement tables for :meth:`_shell1`, built once.

        ``tbl[p][d]`` lists the index displacements that move digit ``d``
        of axis ``p`` to every *other* value, in ascending value order —
        precisely the array the per-move code used to rebuild from
        ``np.arange`` on every call."""
        if self._sh1 is None:
            sp = self.b.space
            self._sh1 = []
            for p in range(sp.n):
                a, pw = int(sp.axes[p]), int(sp.pow_q[p])
                v = np.arange(a, dtype=np.int64)
                rows = np.empty((a, a - 1), dtype=np.int64)
                for d in range(a):
                    rows[d] = (v[v != d] - d) * pw
                self._sh1.append(rows)
        return self._sh1

    def _shell1(self, idx):
        """Every word at digit-distance 1 from ``idx`` — one coordinate
        changed to every other value it can take. Problem-blind: it is
        built from the product space alone, and for every shipped family
        it contains the moves that matter (a Lee/Chebyshev ±1 step is a
        digit change on one axis)."""
        nb = getattr(self.b, "neighbours", None)
        if nb is not None:                  # locality hook (protocol.py)
            return np.asarray(nb(int(idx), 1), dtype=np.int64)
        sp = self.b.space
        tbl = self._shell_tables()
        d = (idx // sp.pow_q) % np.asarray(sp.axes, dtype=np.int64)
        out = [idx + tbl[p][d[p]] for p in range(sp.n)]
        return (np.concatenate(out) if out else np.zeros(0, dtype=np.int64))

    def _shell_k(self, idx, k, m, rng):
        """``m`` sampled words at digit-distance <= k from ``idx``."""
        nb = getattr(self.b, "neighbours", None)
        if nb is not None:                  # locality hook (protocol.py)
            return np.asarray(nb(int(idx), k, m, rng), dtype=np.int64)
        sp = self.b.space
        if m <= 0 or k <= 0 or k > sp.n:
            return np.zeros(0, dtype=np.int64)
        ax = np.asarray(sp.axes, dtype=np.int64)
        pw = np.asarray(sp.pow_q, dtype=np.int64)
        movable = np.flatnonzero(ax > 1)      # a size-1 axis has no other digit
        if not len(movable):
            return np.zeros(0, dtype=np.int64)
        out = np.full(m, idx, dtype=np.int64)
        for _ in range(k):
            p = movable[rng.integers(0, len(movable), size=m)]
            a = ax[p]
            v = rng.integers(0, a - 1)
            cur = (out // pw[p]) % a
            v = v + (v >= cur)                    # skip the current digit
            out = out + (v - cur) * pw[p]
        return out

    def _top_gain_pool(self, work, cap=_SHELL_POOL_CAP):
        """The *top* of the gain map, not all of it.

        :meth:`_gain_pool` extracts at ``thr=1``: on a walled cell that
        is tens of millions of points with a median gain of about 5, and
        a uniform draw from it is worthless next to an incumbent whose
        private coverage is in the hundreds. This takes the same
        histogram-guided high threshold ``greedy_fill`` uses, so the
        pool is the strongest few thousand additions instead."""
        b = self.b
        b.gain_map()
        self.stats["transforms"] += 1
        mx = b.map_max()
        if mx <= 0:
            return np.zeros(0, dtype=np.int64)
        thr = 1
        if mx > 1 and getattr(b, "supports_hist", True):
            hist = b.map_hist(1024, mx)
            csum = np.cumsum(hist[::-1])[::-1]
            good = np.flatnonzero(csum <= cap)
            if len(good):
                thr = max(1, int(good[0]) * mx // 1024 + 1)
        idx, _val, _found = b.map_extract(thr, cap=min(cap, b.extract_cap))
        idx = np.asarray(idx, dtype=np.int64)
        return idx[~np.isin(idx, work)] if len(idx) else idx

    def _shift_anneal(self, M, budget_s, mode, t0, t_end, lahc_len,
                      pool_refresh, log_every, rng, pool_share, n_shell2,
                      n_top, n_rand, w_topk, w_random_p, greedy_p,
                      loss_refresh, restart_stale, verify_every):
        """LAHC / Metropolis polish over the *local* swap neighbourhood.

        Same exact cost model as :meth:`_anneal`, same acceptance rules,
        different neighbourhood — see :meth:`anneal` for why that is the
        whole story on a walled cell."""
        b = self.b
        if len(self.code) > M:
            self.ruin(len(self.code) - M, "low")
        if len(self.code) < M and self.uncov > 0:
            self.greedy_fill(M)
        self.best_uncov = None
        self.best_code = None
        self._snapshot()
        if self.uncov == 0 or not len(self.code):
            return self.uncov

        work = np.sort(np.asarray(self.code, dtype=np.int64)).copy()
        M = len(work)
        self.code = work
        uncov = int(self.uncov)
        best_u, best_code = uncov, work.copy()
        losses = self.losses().astype(np.int64)
        pool = self._top_gain_pool(work)
        auto = pool_refresh == "auto"
        period = 1 if auto else int(pool_refresh)
        tr_s = move_s = None
        t_start = time.time()
        t_mark, acc_mark = t_start, 0
        since_pool = since_loss = 0
        hist = [float(uncov)] * max(1, int(lahc_len))
        i = moves = accepts = drift = stale = reheats = 0
        last_slot = -1

        while uncov > 0:
            elapsed = time.time() - t_start
            if elapsed > budget_s or self.time_left() < 10:
                break
            self.yield_check()

            # ---- the word to move: cheapest by exact loss, sometimes
            # uniform so the pass cannot lock onto one corner
            if rng.random() < w_random_p or len(losses) != M:
                slot = int(rng.integers(M))
            else:
                k = min(int(w_topk), M)
                cheap = np.argpartition(losses, k - 1)[:k]
                slot = int(cheap[rng.integers(k)])
            if slot == last_slot and M > 1:
                slot = int(rng.integers(M))
            w = int(work[slot])
            wa = np.array([w], dtype=np.int64)
            lw = int(b.ball_gather(wa, 1)[0])       # exact loss of the drop
            b.ball_update(wa, -1)

            # ---- candidates that can plug the hole w just made
            parts = [self._shell1(w)]
            if n_shell2:
                parts.append(self._shell_k(w, 2, int(n_shell2), rng))
            if n_top and len(pool):
                parts.append(pool[rng.integers(0, len(pool),
                                               size=min(int(n_top),
                                                        len(pool)))])
            if n_rand:
                parts.append(rng.integers(0, b.space.size, size=int(n_rand))
                             .astype(np.int64))
            cand = np.unique(np.concatenate(parts))
            # w itself must go: re-adding it is a null move costing
            # exactly `uncov`, which LAHC would accept forever
            cand = cand[(cand != w) & ~np.isin(cand, work)]
            if not len(cand):
                b.ball_update(wa, +1)
                i += 1
                continue

            # ---- one batched exact gather costs the WHOLE batch
            gains = b.ball_gather(cand, 0).astype(np.int64)
            costs = uncov + lw - gains
            moves += 1

            bound = hist[i % len(hist)]
            if mode == "lahc":
                ok = np.flatnonzero(costs <= max(bound, uncov))
            else:
                temp = t0 * (t_end / t0) ** min(
                    1.0, elapsed / max(budget_s, 1e-9))
                dl = (costs - uncov).astype(np.float64)
                ok = np.flatnonzero(
                    (dl <= 0) | (rng.random(len(dl)) < np.exp(
                        -np.maximum(dl, 0.0) / max(temp, 1e-9))))
            if len(ok):
                if rng.random() < greedy_p:
                    sub = ok[costs[ok] == costs[ok].min()]
                    pick = int(sub[rng.integers(len(sub))])
                else:
                    pick = int(ok[rng.integers(len(ok))])
                v = int(cand[pick])
                b.ball_update(np.array([v], dtype=np.int64), +1)
                work[slot] = v
                uncov = int(costs[pick])
                accepts += 1
                acc_mark += 1
                since_pool += 1
                since_loss += 1
                last_slot = slot
                if verify_every and accepts % verify_every == 0:
                    exact = b.uncovered()
                    if exact != uncov:
                        drift += 1
                        if drift <= 5:
                            self.on_log(f"WARNING drift: incremental {uncov} "
                                        f"!= field {exact}; adopting field")
                        uncov = exact
                if uncov < best_u:
                    best_u, best_code = uncov, work.copy()
                    stale = 0
                    self.code, self.uncov = work, uncov
                    self._snapshot()
                    if uncov == 0:
                        break
            else:
                b.ball_update(wa, +1)
                last_slot = -1

            if mode == "lahc":
                hist[i % len(hist)] = float(uncov)
            i += 1
            stale += 1
            if restart_stale and stale >= restart_stale:
                # intensify: fall back to the best state seen and restart
                # the acceptance history there
                work = best_code.copy()
                b.recount(work)
                self.stats["transforms"] += 1
                uncov = b.uncovered()
                hist = [float(uncov)] * len(hist)
                losses = self.losses().astype(np.int64)
                pool = self._top_gain_pool(work)
                since_pool = since_loss = stale = 0
                last_slot = -1
                reheats += 1
                self.on_log(f"shift M={M} reheat #{reheats} -> best "
                            f"uncov={uncov} (move {i})")
                continue
            if since_pool >= period:
                t_ref = time.time()
                if auto and acc_mark:
                    m_new = max(t_ref - t_mark, 0.0) / acc_mark
                    move_s = m_new if move_s is None else 0.5 * (move_s + m_new)
                pool = self._top_gain_pool(work)
                dt_tr = time.time() - t_ref
                if auto:
                    tr_s = dt_tr if tr_s is None else 0.5 * (tr_s + dt_tr)
                    if move_s is not None:
                        period = self._auto_refresh(tr_s, move_s, pool_share)
                    self.stats["pool_refresh"] = period
                    t_mark, acc_mark = time.time(), 0
                since_pool = 0
            if since_loss >= loss_refresh:
                self.code = work
                losses = self.losses().astype(np.int64)
                since_loss = 0
            if log_every and i % log_every == 0:
                self.on_log(f"shift M={M} move={i} acc={accepts} "
                            f"uncov={uncov} best={best_u} "
                            f"({elapsed:.0f}s)")

        self.stats["moves"] += moves
        self.stats["accepts"] += accepts
        self.stats["shift_moves"] = self.stats.get("shift_moves", 0) + moves
        self.stats["shift_accepts"] = (self.stats.get("shift_accepts", 0)
                                       + accepts)
        self.code = np.sort(best_code if best_u < uncov else work)
        b.recount(self.code)
        self.stats["transforms"] += 1
        self.uncov = b.uncovered()
        self._snapshot()
        self.on_log(f"shift M={M} done moves={moves} accepts={accepts} "
                    f"({100.0 * accepts / max(1, moves):.1f}%) "
                    f"uncov={self.uncov} drift={drift} reheats={reheats}")
        return self.uncov

    def anneal(self, M, budget_s, mode="lahc", t0=1.0, t_end=0.05,
               lahc_len=50, pool_refresh=64, sample=8, seed=None,
               log_every=2000, pool_share=0.10, neighbourhood="pool",
               n_shell2=_SHELL2, n_top=_SHELL_TOP, n_rand=_SHELL_RAND,
               w_topk=_SHELL_TOPK, w_random_p=0.25, greedy_p=0.5,
               loss_refresh=48, restart_stale=15000, verify_every=1,
               auto_probe=0.15, auto_accept_rate=0.01):
        """Swap-neighbourhood polish at fixed M that accepts regressions.

        LNS only ever walks downhill in uncovered count, so it stalls on
        the "best uncovered = 1..3" plateaus the notch ladder runs into:
        every single repair makes things worse before it makes them
        better. This pass walks that plateau instead. One move is a
        **swap** — drop one incumbent, add one candidate — costed
        exactly and incrementally:

          uncov' = uncov + ball_gather(w, 1) - ball_gather(v, 0)

        (points privately covered by ``w`` become uncovered; points that
        ``v`` newly covers stop being). Both terms are exact backend
        queries, so no move is ever accepted on an approximation, and a
        rejected move is undone by the inverse single-word
        ``ball_update`` — cnt returns to precisely its prior state.

        ``mode="anneal"`` accepts a worsening move with probability
        exp(-delta/T) under geometric cooling from ``t0`` to ``t_end``
        across the budget; ``mode="lahc"`` is late-acceptance hill
        climbing (Burke & Bykov) — accept when the new cost beats the
        cost ``lahc_len`` moves ago — which needs no temperature
        schedule and is the better default on integer-valued costs.

        Each move exact-checks ``sample`` random incumbents for the drop
        and ``sample`` pool candidates for the add; the pool is
        refreshed from the gain map every ``pool_refresh`` accepted
        moves (which is also where the incremental count is re-checked
        against the exact field).

        ``pool_refresh="auto"`` sets that cadence from measurement
        instead: the first refresh times the gain transform, the moves
        between refreshes time a move, and the cadence is re-solved from
        :meth:`_auto_refresh` so transforms stay under ``pool_share`` of
        wall time. It is the right setting whenever the transform cost
        is not known in advance (i.e. on any real cell), and it is what
        :meth:`break_wall` uses — but it reads the clock, so a run under
        it is *not* bit-reproducible from its seed. The integer default
        is kept so the plain ``lahc`` path stays reproducible.

        ``neighbourhood`` decides *where candidates come from*, and on a
        walled cell that — not the cost model, not the acceptance rule —
        is the whole story:

        - ``"pool"`` (default, the frozen behaviour): drop one of
          ``sample`` random incumbents, add one of ``sample`` uniform
          draws from ``map_extract(1)``. Measured on the live wall cells
          this is **inert**: at K_8(8,5), M = 78, an incumbent's private
          coverage is 200-odd words while a random positive-gain point
          has gain about 5, so every sampled swap costs +hundreds and
          LAHC rejects all of them — 243,000 moves, *zero* accepts.
        - ``"shell"``: drop one of the ``w_topk`` cheapest incumbents by
          exact loss (from a cached loss map, refreshed every
          ``loss_refresh`` accepts), and take the candidates from where
          the hole actually is — the dropped word's full distance-1
          shell, ``n_shell2`` sampled distance-2 words, ``n_top`` draws
          from the *top* of the gain map (a high threshold, not
          ``thr=1``) and ``n_rand`` uniform draws. The whole batch is
          costed in **one** ``ball_gather`` and the acceptance rule is
          applied to all of it, so a plateau move is taken whenever one
          exists rather than whenever a handful of blind draws happen to
          contain one. On those same cells: 11-19% accepts, and it walks
          K_8(8,5) M = 78 from 62 uncovered to 13 where LNS had stalled
          at 62 for good. After ``restart_stale`` moves without a new
          best it falls back to the best state seen and restarts the
          acceptance history there.
        - ``"auto"``: spend the budget in ``"pool"`` slices of
          ``auto_probe`` of it, and hand the rest to ``"shell"`` the
          first time a slice's accept rate falls below
          ``auto_accept_rate``. Sliced rather than probed once, because
          the pool neighbourhood is not uniformly bad — it is bad *on
          the plateau*, and a probe taken during the easy descent into
          the plateau reports that everything is fine. This is what the
          wall driver uses; it needs no knowledge of the cell.

        Both neighbourhoods use the same exact incremental cost, the
        same ``ball_update`` undo, and re-check the incremental count
        against ``count_eq(0)`` (``"shell"`` every ``verify_every``
        accepts, ``"pool"`` at every refresh).

        Returns final uncovered."""
        if mode not in ("anneal", "lahc"):
            raise ValueError(f"unknown polish mode: {mode!r}")
        if neighbourhood not in ("pool", "shell", "auto"):
            raise ValueError(f"unknown neighbourhood: {neighbourhood!r}")
        if pool_refresh != "auto":
            if int(pool_refresh) < 1:
                raise ValueError("pool_refresh must be >= 1 or 'auto'")
            pool_refresh = int(pool_refresh)
        shell = dict(n_shell2=n_shell2, n_top=n_top, n_rand=n_rand,
                     w_topk=w_topk, w_random_p=w_random_p, greedy_p=greedy_p,
                     loss_refresh=loss_refresh, restart_stale=restart_stale,
                     verify_every=verify_every)
        with self._rng_scope(seed) as rng:
            self.stats["polish_neighbourhood"] = neighbourhood
            if neighbourhood == "shell":
                return self._shift_anneal(M, budget_s, mode, t0, t_end,
                                          lahc_len, pool_refresh, log_every,
                                          rng, pool_share, **shell)
            if neighbourhood == "auto":
                # The pool neighbourhood is not uniformly bad — it is bad
                # *on the plateau*. A single probe at the start measures
                # the easy descent into the plateau and says "fine", so
                # the budget is spent in pool slices and the switch is
                # made by the first slice whose accept rate collapses.
                t_slice = max(1e-3, budget_s * float(auto_probe))
                t_used = 0.0
                unc = self.uncov
                while t_used < budget_s:
                    span = min(t_slice, budget_s - t_used)
                    mv, ac = self.stats["moves"], self.stats["accepts"]
                    t_at = time.time()
                    unc = self._anneal(M, span, mode, t0, t_end, lahc_len,
                                       pool_refresh, sample, log_every, rng,
                                       pool_share)
                    t_used += time.time() - t_at
                    if unc == 0:
                        self.stats["polish_neighbourhood"] = "pool"
                        return unc
                    dm = self.stats["moves"] - mv
                    rate = (self.stats["accepts"] - ac) / max(1, dm)
                    if dm and rate < auto_accept_rate:
                        self.stats["polish_neighbourhood"] = "shell"
                        self.on_log(
                            f"polish auto: pool accepted {100 * rate:.2f}% of "
                            f"{dm} moves -> switching to the shell "
                            f"neighbourhood for the rest")
                        return self._shift_anneal(
                            M, max(0.0, budget_s - t_used), mode, t0, t_end,
                            lahc_len, pool_refresh, log_every, rng,
                            pool_share, **shell)
                    if self.time_left() < 10:
                        break
                self.stats["polish_neighbourhood"] = "pool"
                return unc
            return self._anneal(M, budget_s, mode, t0, t_end, lahc_len,
                                pool_refresh, sample, log_every, rng,
                                pool_share)

    def _anneal(self, M, budget_s, mode, t0, t_end, lahc_len, pool_refresh,
                sample, log_every, rng, pool_share=0.10):
        b = self.b
        if len(self.code) > M:
            self.ruin(len(self.code) - M, "low")
        if len(self.code) < M and self.uncov > 0:
            self.greedy_fill(M)
        self.best_uncov = None
        self.best_code = None
        self._snapshot()
        if self.uncov == 0 or not len(self.code):
            return self.uncov

        work = np.asarray(self.code, dtype=np.int64).copy()
        codeset = set(work.tolist())
        pool = np.zeros(0, dtype=np.int64)
        auto = pool_refresh == "auto"
        period = 1 if auto else pool_refresh
        since_refresh = period                    # force the first refresh
        starved = 0
        hist = [float(self.uncov)] * max(1, int(lahc_len))
        t_start = time.time()
        # auto-cadence measurement state: cost of one transform, cost of
        # one accepted move, and where the last refresh interval began
        tr_s = move_s = None
        t_mark, acc_mark = t_start, 0
        i = moves = accepts = 0
        while self.uncov > 0:
            elapsed = time.time() - t_start
            if elapsed > budget_s or self.time_left() < 10:
                break
            self.yield_check()
            if since_refresh >= period or not len(pool):
                self.code = work
                t_ref = time.time()
                if auto and acc_mark:             # cost of an accepted move
                    m_new = max(t_ref - t_mark, 0.0) / acc_mark
                    move_s = m_new if move_s is None else 0.5 * (move_s + m_new)
                pool = self._gain_pool()
                exact = b.uncovered()
                if exact != self.uncov:
                    self.on_log(f"WARNING drift: incremental {self.uncov} != "
                                f"field {exact}; adopting field")
                    self.uncov = exact
                dt_tr = time.time() - t_ref
                if auto:
                    tr_s = dt_tr if tr_s is None else 0.5 * (tr_s + dt_tr)
                    if move_s is not None:
                        period = self._auto_refresh(tr_s, move_s, pool_share)
                    self.stats["pool_refresh"] = period
                    t_mark, acc_mark = time.time(), 0
                since_refresh = 0
                if not len(pool):
                    break

            # ---- drop: cheapest of `sample` random incumbents
            ns = min(sample, len(work))
            ridx = rng.choice(len(work), size=ns, replace=False)
            loss = b.ball_gather(work[ridx], 1).astype(np.float64)
            j = int(np.argmin(loss + rng.random(ns)))
            slot = int(ridx[j])
            w = int(work[slot])
            wa = np.array([w], dtype=np.int64)
            b.ball_update(wa, -1)
            codeset.discard(w)

            # ---- add: best of `sample` pool candidates (exact gains)
            ns = min(sample, len(pool))
            draw = pool[rng.integers(0, len(pool), size=ns)]
            cand = np.array([c for c in draw.tolist()
                             if c != w and c not in codeset], dtype=np.int64)
            if not len(cand):                     # stale pool exhausted
                b.ball_update(wa, +1)
                codeset.add(w)
                since_refresh = period
                pool = np.zeros(0, dtype=np.int64)
                starved += 1
                if starved > 2:
                    break
                continue
            starved = 0
            gain = b.ball_gather(cand, 0).astype(np.float64)
            g = int(np.argmax(gain + rng.random(len(gain))))
            v = int(cand[g])
            b.ball_update(np.array([v], dtype=np.int64), +1)
            cost = self.uncov + int(loss[j]) - int(gain[g])

            # ---- acceptance
            moves += 1
            if mode == "lahc":
                ok = cost <= hist[i % len(hist)] or cost <= self.uncov
            else:
                delta = cost - self.uncov
                temp = t0 * (t_end / t0) ** min(
                    1.0, elapsed / max(budget_s, 1e-9))
                ok = delta <= 0 or rng.random() < np.exp(
                    -delta / max(temp, 1e-9))
            if ok:
                work[slot] = v
                codeset.add(v)
                self.uncov = cost
                accepts += 1
                acc_mark += 1
                since_refresh += 1
                self.code = work
                self._snapshot()
            else:
                b.ball_update(np.array([v], dtype=np.int64), -1)
                b.ball_update(wa, +1)
                codeset.add(w)
            if mode == "lahc":
                hist[i % len(hist)] = float(self.uncov)
            i += 1
            if log_every and i % log_every == 0:
                self.on_log(f"{mode} M={M} move={i} acc={accepts} "
                            f"uncov={self.uncov} best={self.best_uncov} "
                            f"({elapsed:.0f}s)")

        self.stats["moves"] += moves
        self.stats["accepts"] += accepts
        self.code = np.sort(work)
        if self.uncov == 0:
            self._snapshot()
        elif self.best_code is not None and self.best_uncov < self.uncov:
            self.code = np.sort(self.best_code)
            b.recount(self.code)
            self.stats["transforms"] += 1
            self.uncov = b.uncovered()
        self.on_log(f"{mode} M={M} done moves={moves} accepts={accepts} "
                    f"uncov={self.uncov}")
        return self.uncov

    # ------------------------------------------------ exact endgame search
    def _coverers(self, thr, cap, rng):
        """Points whose ball meets at least ``thr`` deficient words.

        One gain transform plus one extract: at ``thr=1`` this is the
        whole deficit neighbourhood (every addition that helps at all),
        at ``thr=uncov`` it is exactly the additions that would close the
        deficit on their own.

        Ordering is gain-descending, ties broken by a draw from ``rng``.
        Both halves of that matter: the extract is unordered and its GPU
        slot order is not reproducible, so *some* deterministic key is
        needed; and at the wall every candidate has gain 1, so a key of
        "index ascending" would make the driver re-try the same handful
        of numerically-smallest coverers on every burst forever."""
        self.b.gain_map()
        self.stats["transforms"] += 1
        if self.b.map_max() < thr:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        idx, val, _found = self.b.map_extract(int(thr),
                                              cap=min(cap, self.b.extract_cap))
        idx = np.asarray(idx, dtype=np.int64)
        val = np.asarray(val, dtype=np.int64)
        order = np.argsort(-val + rng.random(len(val)), kind="stable")
        return idx[order], val[order]

    def two_swap(self, M=None, budget_s=30.0, max_uncov=2, branch=6,
                 drop_branch=6, slack=1, cand_cap=4096, nodes=2000,
                 depth=2, seed=None):
        """Exact bounded (add, drop) search for the endgame plateau.

        The documented wall is "no single swap improves": every drop
        that frees a slot costs more than any single addition can pay
        back. The way off such a plateau is a *pair* of swaps, and at
        uncovered <= 2 the pair can be searched exactly rather than
        sampled, because the useful additions are not arbitrary points —
        they are exactly the ball around the deficit, which
        :meth:`_coverers` reads straight out of the gain field.

        The search is a depth-``depth`` DFS over (add x, drop c) steps,
        M-preserving at every step:

          * additions come from :meth:`_coverers`; at the *last* level
            only additions that close the whole remaining deficit are
            considered (``thr = uncov``), since a drop can never lower
            the count;
          * drops are ranked by their exact loss **after** the addition
            (`ball_gather(code, 1)`, one batched exact query), so the
            cost of the pair is never estimated;
          * a step is expanded only if it leaves the count within
            ``slack`` of where the level started — worsening by one is
            the escape, worsening by five is a different search;
          * every step is undone by the inverse ``ball_update``, so a
            failed search leaves ``cnt`` bit-identical to the state it
            started from.

        Exponential in ``depth`` in principle; ``branch``,
        ``drop_branch``, ``nodes`` and ``budget_s`` bound it in practice
        (default: at most a few thousand exact pair evaluations). Runs
        only when uncovered is in 1..``max_uncov`` — outside that range
        the deficit neighbourhood is not small and this is the wrong
        tool. Returns the final uncovered count; the improving sequence
        is applied if one was found, otherwise the state is restored."""
        if self.uncov is None or not (0 < self.uncov <= max_uncov):
            return self.uncov
        if M is not None and len(self.code) != M:
            raise ValueError(f"two_swap needs |code| == M ({len(self.code)} "
                             f"!= {M})")
        if int(depth) < 1:
            raise ValueError("two_swap depth must be >= 1")
        with self._rng_scope(seed) as rng:
            return self._two_swap(budget_s, branch, drop_branch, slack,
                                  cand_cap, nodes, int(depth), rng)

    def _two_swap(self, budget_s, branch, drop_branch, slack, cand_cap,
                  nodes, depth, rng):
        start_code = np.sort(np.asarray(self.code, dtype=np.int64)).copy()
        start_u = int(self.uncov)
        self.code = start_code.copy()
        state = {"nodes": int(nodes), "end": time.time() + budget_s,
                 "best_u": start_u, "best_code": None}
        self._ts_step(depth, start_u, slack, branch, drop_branch, cand_cap,
                      state, rng)
        self.stats["two_swap_nodes"] = (
            self.stats.get("two_swap_nodes", 0) + int(nodes) - state["nodes"])
        if state["best_code"] is not None:
            self.code = np.sort(state["best_code"])
            self.b.recount(self.code)
            self.stats["transforms"] += 1
            self.stats["recounts"] += 1
            self.uncov = self.b.uncovered()
            self.stats["two_swap_hits"] = self.stats.get("two_swap_hits", 0) + 1
            self.on_log(f"two_swap M={len(self.code)} {start_u} -> "
                        f"{self.uncov}")
            self._snapshot()
        else:
            self.code = start_code
            self.b.recount(self.code)
            self.stats["transforms"] += 1
            self.stats["recounts"] += 1
            self.uncov = self.b.uncovered()
        return self.uncov

    def _ts_step(self, depth, level_u, slack, branch, drop_branch, cand_cap,
                 state, rng):
        """One (add, drop) level of the endgame DFS. Returns True as soon
        as a state with uncovered == 0 has been recorded."""
        b = self.b
        if self.uncov == 0:
            state["best_u"] = 0
            state["best_code"] = np.sort(self.code).copy()
            return True
        if depth <= 0 or state["nodes"] <= 0 or time.time() > state["end"]:
            return False
        # last level: only an addition that closes the whole remaining
        # deficit can reach zero, since every drop costs >= 0
        thr = int(self.uncov) if depth == 1 else 1
        cand, gains = self._coverers(thr, cand_cap, rng)
        if not len(cand):
            return False
        code = np.asarray(self.code, dtype=np.int64)
        codeset = set(code.tolist())
        taken = 0
        for x, g in zip(cand.tolist(), gains.tolist()):
            if taken >= branch or state["nodes"] <= 0:
                break
            if x in codeset or time.time() > state["end"]:
                continue
            taken += 1
            xa = np.array([x], dtype=np.int64)
            b.ball_update(xa, +1)
            u_add = int(self.uncov) - int(g)
            # exact loss of every incumbent, *given* x is in
            loss = b.ball_gather(code, 1).astype(np.int64)
            order = np.argsort(loss + rng.random(len(loss)))
            drops = 0
            for j in order.tolist():
                if drops >= drop_branch or state["nodes"] <= 0:
                    break
                c = int(code[j])
                new_u = u_add + int(loss[j])
                if new_u > level_u + slack and depth > 1:
                    break                       # sorted: the rest are worse
                if depth == 1 and new_u > 0:
                    break
                drops += 1
                state["nodes"] -= 1
                ca = np.array([c], dtype=np.int64)
                b.ball_update(ca, -1)
                keep = self.uncov
                self.code = np.append(code[code != c], x)
                self.uncov = new_u
                if new_u < state["best_u"]:
                    state["best_u"] = new_u
                    state["best_code"] = np.sort(self.code).copy()
                hit = (new_u == 0 or
                       self._ts_step(depth - 1, level_u, slack, branch,
                                     drop_branch, cand_cap, state, rng))
                if hit:
                    return True
                b.ball_update(ca, +1)
                self.code = code
                self.uncov = keep
            b.ball_update(xa, -1)
        return False

    # ------------------------------------------------ restart portfolio
    def portfolio(self, M, budget_s, streams=4, slice_s=None, kmin=8,
                  kmax=None, ruin_frac=0.15, select="fixed", seed=None,
                  deficit=False):
        """Time-sliced restart portfolio at fixed M with a shared incumbent.

        A single LNS run at a hard notch either gets there or burns the
        whole budget in one basin. This pass keeps ``streams`` distinct
        solutions — the current one plus perturbations of it, each
        seeded by a different ruin operator — and gives each a
        ``slice_s`` slice of :meth:`lns` per round, round-robin. The
        best solution seen is the shared incumbent; after every round
        the *worst* stream is discarded and re-seeded from a fresh
        perturbation of the incumbent, which is what makes this a
        portfolio rather than ``streams`` unrelated runs. Stops as soon
        as any stream reaches uncovered = 0. Returns the incumbent's
        uncovered count, with the incumbent loaded."""
        with self._rng_scope(seed):
            return self._portfolio(M, budget_s, streams, slice_s, kmin, kmax,
                                   ruin_frac, select, bool(deficit))

    def _portfolio(self, M, budget_s, streams, slice_s, kmin, kmax, ruin_frac,
                   select, deficit=False):
        t0 = time.time()
        streams = max(1, int(streams))
        slice_s = slice_s or max(1.0, budget_s / (3.0 * streams))
        if len(self.code) > M:
            self.ruin(len(self.code) - M, "low")
        if len(self.code) < M and self.uncov > 0:
            self.greedy_fill(M)
        kick = max(1, int(ruin_frac * M))
        base = self.code.copy()

        def perturb(code, mode):
            """A fresh stream: kick the incumbent, then greedily refill."""
            self.load(code)
            self.ruin(kick, mode)
            self.greedy_fill(M)
            return self.code.copy(), self.uncov

        pop = [(base, self.uncov)]
        if self.uncov > 0:
            for s in range(1, streams):
                pop.append(perturb(base, RUIN_MODES[s % len(RUIN_MODES)]))
        inc, inc_u = min(pop, key=lambda cu: cu[1])

        while inc_u > 0:
            if time.time() - t0 > budget_s or self.time_left() < 10:
                break
            for s in range(len(pop)):
                left = budget_s - (time.time() - t0)
                if left <= 0 or self.time_left() < 10:
                    break
                self.load(pop[s][0])
                unc = self.lns(M, round_budget_s=min(slice_s, left),
                               kmin=kmin, kmax=kmax, select=select,
                               deficit=deficit)
                pop[s] = (self.code.copy(), unc)
                if unc < inc_u:
                    inc, inc_u = pop[s]
                if unc == 0:
                    break
            if inc_u == 0 or time.time() - t0 > budget_s:
                break
            worst = int(np.argmax([u for _, u in pop]))
            self.stats["restarts"] += 1
            mode = RUIN_MODES[self.stats["restarts"] % len(RUIN_MODES)]
            pop[worst] = perturb(inc, mode)
            if pop[worst][1] < inc_u:
                inc, inc_u = pop[worst]
            self.on_log(f"portfolio M={M} streams={len(pop)} best={inc_u} "
                        f"restart=s{worst}/{mode} "
                        f"({time.time() - t0:.0f}s)")

        self.best_uncov = None
        self.best_code = None
        self.load(inc)
        return self.uncov

    # ------------------------------------------------ wall breaker
    def break_wall(self, M_target, budget_s, burst_s=None, kmin=8, kmax=None,
                   select="adaptive", deficit=True, polish="lahc",
                   polish_share=0.6, polish_neighbourhood="auto",
                   two_swap_at=2, two_swap_s=None, stagnation=3,
                   restart_frac=0.05, seed=None, on_wall_log=None):
        """The endgame playbook for one hard M, as a single call.

        A "wall" is a state where a cover of size M+1 is in hand, a cover
        of size M is believed to exist, and the search sits at
        best-uncovered 1..3 for hours. What works there is not more of
        any one pass; it is a *rotation*, because each pass is blind to a
        different thing:

          1. a **focused-ruin LNS burst** — ruin/recreate with the
             ``deficit`` operator in the mix and hinted while the count
             is low, which moves the incumbents that actually block the
             deficit rather than a random neighbourhood;
          2. a **LAHC polish burst** at the same M over the *local*
             swap neighbourhood (``polish_neighbourhood``, default
             ``"auto"``; ``pool_refresh`` auto-tuned), which walks the
             plateau by swaps instead of rebuilding and accepts
             sideways/worse states. This is the workhorse and gets the
             larger share of every burst: on the live wall cells the
             stock pool neighbourhood accepts nothing at all, while the
             local one runs at 11-19% accepts and moves counts LNS
             cannot;
          3. the **exact two-swap endgame** once the count is down to
             ``two_swap_at``, which is the only pass here that can prove
             "no pair of swaps closes this" rather than sample it.

        Between bursts the best state seen is kept; after ``stagnation``
        bursts without improvement the search is restored to that best
        state and kicked with a ``restart_frac``·M deficit ruin, so a
        burst chain never wanders away from the wall for good.

        Everything is at fixed M — the size never changes — so the only
        outcomes are 0 (broken) or an honest final count. Returns that
        count, with the best state loaded. The per-burst trajectory is
        left in ``stats["wall"]``."""
        if polish not in (None, "anneal", "lahc"):
            raise ValueError(f"unknown polish mode: {polish!r}")
        if polish_neighbourhood not in ("pool", "shell", "auto"):
            raise ValueError(
                f"unknown neighbourhood: {polish_neighbourhood!r}")
        with self._rng_scope(seed):
            return self._break_wall(int(M_target), budget_s, burst_s, kmin,
                                    kmax, select, bool(deficit), polish,
                                    polish_share, polish_neighbourhood,
                                    int(two_swap_at), two_swap_s,
                                    int(stagnation), restart_frac,
                                    on_wall_log)

    def _break_wall(self, M, budget_s, burst_s, kmin, kmax, select, deficit,
                    polish, polish_share, hood, two_swap_at, two_swap_s,
                    stagnation, restart_frac, on_wall_log):
        t0 = time.time()
        log = on_wall_log or self.on_log
        burst_s = burst_s or max(5.0, budget_s / 12.0)
        two_swap_s = two_swap_s if two_swap_s is not None else burst_s * 0.25
        # seed at the incumbent, then drop to the target
        if len(self.code) > M:
            self.ruin(len(self.code) - M, "low")
        if len(self.code) < M:
            self.greedy_fill(M)
        best_code, best_u = self.code.copy(), int(self.uncov)
        traj = [(0.0, best_u)]
        kick = max(1, int(restart_frac * M))
        stall = burst = 0
        share = float(polish_share) if polish else 0.0
        while best_u > 0:
            left = budget_s - (time.time() - t0)
            if left <= 0 or self.time_left() < 10:
                break
            burst += 1
            span = min(burst_s, left)
            # 1. focused-ruin LNS
            self._lns(M, span * (1.0 - share), kmin, kmax, 0, select,
                      deficit, 3, 4.0)
            # 2. plateau polish over the local neighbourhood
            if self.uncov > 0 and share > 0:
                self.anneal(M, span * share, mode=polish, pool_refresh="auto",
                            log_every=0, neighbourhood=hood)
            # 3. exact endgame
            if 0 < self.uncov and len(self.code) < M:
                self.greedy_fill(M)            # two_swap is M-preserving
            if 0 < self.uncov <= two_swap_at and len(self.code) == M:
                self.two_swap(M, budget_s=min(
                    two_swap_s, max(0.0, budget_s - (time.time() - t0))),
                    max_uncov=two_swap_at)
            cur = int(self.uncov)
            if cur < best_u:
                best_u, best_code = cur, self.code.copy()
                stall = 0
            else:
                stall += 1
            traj.append((round(time.time() - t0, 1), cur))
            log(f"wall M={M} burst={burst} uncov={cur} best={best_u} "
                f"stall={stall} ({time.time() - t0:.0f}s)")
            if best_u == 0:
                break
            if stall >= stagnation:            # restart from the best state
                self.load(best_code)
                self.ruin(kick, "deficit" if deficit else "hole")
                self.greedy_fill(M)
                self.stats["restarts"] += 1
                stall = 0
        if self.uncov != best_u or len(self.code) != len(best_code):
            self.load(best_code)
        self.best_code, self.best_uncov = best_code.copy(), best_u
        self.stats["wall"] = {"M": M, "bursts": burst, "best": best_u,
                              "traj": traj,
                              "secs": round(time.time() - t0, 1)}
        return self.uncov

    # ------------------------------------------------ notch descent
    def _notch_attempt(self, M, budget_s, kmin, kmax, select, polish,
                       polish_share, portfolio, deficit=False, hood="pool"):
        """One notch attempt: LNS (or a portfolio of LNS streams), then —
        if that walls out with a few points uncovered — the annealing /
        LAHC polish on the wall, inside the *same* notch budget."""
        share = float(polish_share) if polish else 0.0
        unc = (self.portfolio(M, budget_s * (1.0 - share), streams=portfolio,
                              kmin=kmin, kmax=kmax, select=select,
                              deficit=deficit)
               if portfolio > 1 else
               self.lns(M, round_budget_s=budget_s * (1.0 - share),
                        kmin=kmin, kmax=kmax, select=select,
                        deficit=deficit))
        if unc > 0 and share > 0:
            self.on_log(f"polish {polish} M={M} from uncov={unc}")
            unc = self.anneal(M, budget_s * share, mode=polish,
                              neighbourhood=hood)
        return unc

    def descend(self, floor=1, notch_budget_s=300.0, step0=4, step_cap=1024,
                ruin_extra=0, kmin=8, kmax=None, select="fixed", polish=None,
                polish_share=0.5, portfolio=1, seed=None, deficit=False,
                polish_neighbourhood="pool"):
        """From the current (feasible) solution, descend M by notches:
        try M-step, halve step on failure, double on fast success. Every
        verified solution fires on_solved(M, code). The proven record
        driver, ported from the prototype engine.

        The notch attempt itself is configurable, and defaults to
        exactly the proven one: ``select`` picks fixed or adaptive ruin
        selection, ``portfolio`` > 1 runs that many time-sliced LNS
        streams instead of one, and ``polish`` in {"anneal", "lahc"}
        spends the last ``polish_share`` of each notch budget on
        :meth:`anneal` when the repair walls out."""
        if self.uncov is None or self.uncov != 0:
            raise RuntimeError("descend requires a feasible starting solution")
        if polish not in (None, "anneal", "lahc"):
            raise ValueError(f"unknown polish mode: {polish!r}")
        if polish_neighbourhood not in ("pool", "shell", "auto"):
            raise ValueError(
                f"unknown neighbourhood: {polish_neighbourhood!r}")
        with self._rng_scope(seed):
            return self._descend(floor, notch_budget_s, step0, step_cap,
                                 ruin_extra, kmin, kmax, select, polish,
                                 polish_share, portfolio, bool(deficit),
                                 polish_neighbourhood)

    def _descend(self, floor, notch_budget_s, step0, step_cap, ruin_extra,
                 kmin, kmax, select, polish, polish_share, portfolio,
                 deficit=False, hood="pool"):
        self.peel()
        assert self.verify_full() == 0
        solved = self.code.copy()
        M = len(self.code)
        self.on_solved(M, solved)
        step, fails_at_1 = step0, 0
        while self.time_left() > 0 and M > floor and fails_at_1 < 3:
            target = M - step
            t0 = time.time()
            budget = min(notch_budget_s * (2 ** min(fails_at_1, 2)),
                         max(0.0, self.time_left() - 30))
            if budget < min(20.0, notch_budget_s):
                break                          # before ruin: code stays feasible
            # ruin past the deficit so the first repair has real slots
            self.ruin(len(self.code) - target + ruin_extra,
                      "cluster" if ruin_extra else "low")
            unc = self._notch_attempt(target, budget, kmin, kmax, select,
                                      polish, polish_share, portfolio,
                                      deficit, hood)
            if unc == 0 and self.verify_full() == 0:
                self.peel()
                if self.verify_full() != 0:
                    self.on_log("PEEL BROKE FEASIBILITY (bug!) — restoring")
                    self.load(solved)
                    break
                solved = self.code.copy()
                M = len(self.code)
                dt = time.time() - t0
                self.on_log(f"SOLVED M={M} in {dt:.0f}s (step={step})")
                self.on_solved(M, solved)
                fails_at_1 = 0
                if dt < budget * 0.25:
                    step = min(step * 2, step_cap)
            else:
                self.on_log(f"notch M={target} FAILED (best uncov="
                            f"{self.best_uncov}, step={step}, "
                            f"budget={budget:.0f}s)")
                if step == 1:
                    fails_at_1 += 1
                step = max(1, step // 2)
                self.load(solved)              # restore last feasible
        return M

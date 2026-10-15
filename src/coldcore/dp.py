"""Fast CPU backend: the plugin's axis-DP transforms, in numpy.

The reference backends (``coldcore.reference``) are O(|X|^2) brute force
capped at 2^16 words — they are *oracles*, not engines, and every CPU
campaign that runs on them pays a quadratic price for a job the GPU
plugin does in O(n |X|).

This module closes that gap without a GPU. It is the same mathematics
the CUDA plugin implements (``src/plugins/covering/dctcore_*.cuh``),
vectorised over fibers with numpy instead of over threads with CUDA:

* **hamming / lee** — grade-truncated distance-count layers ``A_0..A_R``.
  ``A_d(x)`` after axes ``0..j`` is the number of solution elements that
  match ``x`` on axes ``> j`` at distance exactly ``d`` on axes ``<= j``.
  One in-place pass per axis over the layer stack; after all ``n`` axes
  ``sum_d A_d`` is the exact field. Hamming's per-fiber recurrence is
  ``newA_d[a] = A_d[a] + T_{d-1} - A_{d-1}[a]`` (``T_d`` = fiber sum);
  Lee's is the circulant ``newA_d[a] = A_d[a] + sum_w (A_{d-w}[a-w] +
  A_{d-w}[a+w])``, the two terms counted once when they coincide.
* **torus_linf / grid_linf** — a single plane and a per-axis window sum,
  circular for the torus and clamped at the boundary for the open grid.

Incremental ``ball_update`` / ``ball_gather`` walk the *same* support
set the plugin does (``coldcore.patterns``'s enumeration, held as
(position, delta) columns rather than nibble-packed uint64s): a ball is
``w + offsets(digits(w))``, a handful of vectorised gathers per centre.
``grid_linf`` also carries the plugin's boundary rejection, so edge
balls shrink.

Exactness
---------
Every field this backend produces is bit-identical to the brute-force
reference's (``tests/test_dp_backend.py`` parity-gates it against all
four ``Ref*Backend`` classes on randomised configurations, mu included),
and so are the ``map_*``/``count_*`` queries — including ``map_extract``
ordering. A :class:`~coldcore.search.Searcher` run therefore produces
the *same* move sequence on ``cpu`` as on ``ref`` for the same seed.

Cost
----
Transforms are ``O(n |X| (R+1))`` time and ``O(|X| (R+1))`` memory in
``dtype`` (int32 by default, matching the plugin's own int32 layer
planes and its overflow envelope; pass ``dtype=np.int64`` for headroom).
Usable to |X| of order 10^7-10^8 on a workstation.
"""
from itertools import combinations

import numpy as np

from .patterns import _delta_sets, _max_pattern_weight
from .protocol import DenseFieldBackend
from .space import ProductSpace

#: Elements per batched ball gather; caps the (centres x ball) temporary.
_GATHER_ELEMS = 1 << 23

#: ``ball_gather`` builds the whole-space role mask only when the batch
#: is wide enough to amortise the O(|X|) pass; below that it gathers cnt
#: values and applies the predicate to those. Pure cost policy — the
#: answer is identical either way. Measured break-even on this machine.
_MASK_BREAKEVEN = 2

PROBLEMS = ("hamming", "torus_linf", "lee", "grid_linf")


class NumpyDPBackend(DenseFieldBackend):
    """One problem cell held in host memory, transformed by axis DP.

    Parameters
    ----------
    axes : per-coordinate alphabet sizes (mixed radix allowed)
    R : covering radius
    problem : one of :data:`PROBLEMS`
    mu : covering multiplicity (``K^mu``; see docs/multi-coverage.md)
    dtype : layer/field integer type (int32 default, int64 for headroom)
    seed_shuffle : shuffle ``map_extract`` output with this seed
    """

    def __init__(self, axes, R, problem="hamming", extract_cap=1 << 22,
                 mu=1, seed_shuffle=None, dtype=np.int32):
        if problem not in PROBLEMS:
            raise ValueError(f"unknown problem {problem!r}")
        self.problem = problem
        self.R = R = int(R)
        self.mu = int(mu)
        if self.mu < 1:
            raise ValueError("mu must be >= 1")
        self.space = ProductSpace(axes)
        self.axes = list(self.space.axes)
        self.n = self.space.n
        size = self.space.size
        if problem == "torus_linf" and any(2 * R + 1 > a for a in self.axes):
            raise ValueError("need 2R+1 <= ax_j on every axis")
        self.extract_cap = extract_cap
        self.dtype = np.dtype(dtype)
        # sum metrics dominate digit distance -> batched peel stays valid
        self.indep_dist = 2 * R if problem in ("hamming", "lee") else None
        self.supports_hist = True
        self._rng = (np.random.default_rng(seed_shuffle)
                     if seed_shuffle is not None else None)

        self.nlayers = 1 if problem in ("torus_linf", "grid_linf") else R + 1
        self.cnt = np.zeros(size, dtype=self.dtype)
        self.map = np.zeros(size, dtype=self.dtype)
        self._A = np.zeros((self.nlayers, size), dtype=self.dtype)
        # per-axis fiber geometry: flat index = h*(Q*S) + a*S + s
        self._fibers = [(int(size // (self.space.pow_q[j] * self.axes[j])),
                         int(self.axes[j]), int(self.space.pow_q[j]))
                        for j in range(self.n)]
        self._pow = np.asarray(self.space.pow_q, dtype=np.int64)
        self._span = max(self.axes)                 # contribution-table stride
        self._build_support()

    # ================================================== support table
    def _build_support(self):
        """The support ("ball") table as (position, delta) columns.

        Same enumeration as :func:`coldcore.patterns.support_patterns` —
        per-position admissible circular shifts, subsets up to the
        family's max pattern weight, Lee's total-weight coupling — but
        built straight into columns instead of nibble-packed uint64s.
        That drops the packed format's ``ax_j <= 16`` and ``weight <= 8``
        caps, which are the plugin's ABI, not the mathematics; the
        enumerated ball is asserted equal to the packed one wherever both
        apply (``tests/test_dp_backend.py``).

        ``_key`` indexes the per-centre contribution table; unused slots
        of a short pattern point at a trailing zero entry, which is what
        makes the walk branch-free.
        """
        n, R, prob = self.n, self.R, self.problem
        ds = _delta_sets(self.axes, R, prob)
        maxw = min(_max_pattern_weight(n, R, prob), n)
        span = self._span
        pad = n * span                              # the zero slot
        blocks = [np.full((1, maxw), pad, dtype=np.int64)]   # the centre
        for w in range(1, maxw + 1):
            for subs in combinations(range(n), w):
                sets = [np.asarray(ds[p], dtype=np.int64) for p in subs]
                if any(not len(s) for s in sets):
                    continue
                grid = np.meshgrid(*sets, indexing="ij")
                d = np.stack([g.ravel() for g in grid], axis=1)  # (K, w)
                if prob == "lee":       # the one family that couples axes
                    wt = np.zeros(len(d), dtype=np.int64)
                    for k, p in enumerate(subs):
                        wt += np.minimum(d[:, k], self.axes[p] - d[:, k])
                    d = d[wt <= R]
                if not len(d):
                    continue
                blk = np.full((len(d), maxw), pad, dtype=np.int64)
                for k, p in enumerate(subs):
                    blk[:, k] = p * span + (d[:, k] - 1)
                blocks.append(blk)
        key = np.concatenate(blocks) if blocks else np.zeros((0, maxw),
                                                             dtype=np.int64)
        self.max_weight = maxw
        self.ball_len = len(key)
        self._key = np.ascontiguousarray(key)       # (L, maxw) int64

    def _contrib(self, digs):
        """Per-centre index displacement of every (position, delta).

        ``digs`` is (m, n); the result is (m, n*span + 1), the last
        column the zero pad. Entry ``p*span + d-1`` is
        ``(((dig_p + d) mod a_p) - dig_p) * pow_p`` — exactly the
        plugin's ``build_contrib``.
        """
        m = len(digs)
        out = np.zeros((m, self.n * self._span + 1), dtype=np.int64)
        for p in range(self.n):
            a = self.axes[p]
            if a < 2:
                continue
            d = np.arange(1, a, dtype=np.int64)
            base = digs[:, p:p + 1]
            nd = (base + d[None, :]) % a
            out[:, p * self._span:p * self._span + a - 1] = (nd - base) * int(
                self._pow[p])
        return out

    def _inbox(self, digs):
        """``grid_linf`` boundary mask, shaped like :meth:`_contrib`.

        A circular delta that had to wrap past a board edge lands outside
        the L-inf ball; the plugin rejects such patterns per centre, and
        so does this."""
        m = len(digs)
        out = np.zeros((m, self.n * self._span + 1), dtype=bool)
        out[:, -1] = True                            # the pad is always fine
        for p in range(self.n):
            a = self.axes[p]
            if a < 2:
                continue
            d = np.arange(1, a, dtype=np.int64)
            base = digs[:, p:p + 1]
            sh = ((base + d[None, :]) % a) - base
            out[:, p * self._span:p * self._span + a - 1] = np.abs(sh) <= self.R
        return out

    def _balls(self, idx):
        """(m, L) word indices covered by each centre, and a validity
        mask (``None`` when every pattern is always in the ball)."""
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if self.max_weight == 0:                     # R=0: the ball is {w}
            return idx[:, None].copy(), None
        digs = self.space.index_word(idx).astype(np.int64)
        c = self._contrib(digs)
        key = self._key
        off = c[:, key[:, 0]]
        for k in range(1, self.max_weight):
            off += c[:, key[:, k]]
        off += idx[:, None]
        if self.problem != "grid_linf":
            return off, None
        ok = self._inbox(digs)
        good = ok[:, key[:, 0]]
        for k in range(1, self.max_weight):
            good &= ok[:, key[:, k]]
        return off, good

    def _chunk(self):
        """Centres per batch: bounds both the (centres x ball) gather and
        the per-centre contribution table."""
        return max(1, _GATHER_ELEMS // max(self.ball_len,
                                          self.n * self._span))

    def ball_indices(self, w):
        """Indices of all points covered by a move at word index w."""
        off, good = self._balls(np.array([int(w)], dtype=np.int64))
        out = off[0] if good is None else off[0][good[0]]
        return np.sort(out)

    # ================================================== axis DP passes
    def _view(self, j):
        h, q, s = self._fibers[j]
        return self._A.reshape(self.nlayers, h, q, s)

    def _axis_hamming(self, j):
        v = self._view(j)
        t = v.sum(axis=2)                            # (nl, H, S) fiber sums
        for d in range(self.nlayers - 1, 0, -1):
            v[d] += t[d - 1][:, None, :] - v[d - 1]

    def _axis_lee(self, j):
        v = self._view(j)
        q = self._fibers[j][1]
        w_max = q // 2
        for d in range(self.nlayers - 1, 0, -1):
            for w in range(1, min(d, w_max) + 1):
                src = v[d - w]
                v[d] += np.roll(src, w, axis=1)      # A_{d-w}[a-w]
                if not (q % 2 == 0 and w == w_max):
                    v[d] += np.roll(src, -w, axis=1)  # A_{d-w}[a+w]

    def _axis_torus(self, j):
        v = self._view(j)
        r = self.R
        acc = np.zeros_like(v[0])
        for e in range(-r, r + 1):
            acc += np.roll(v[0], -e, axis=1)         # A_0[(a+e) mod Q]
        v[0] = acc

    def _axis_grid(self, j):
        v = self._view(j)
        h, q, s = self._fibers[j]
        r = self.R
        cs = np.zeros((h, q + 1, s), dtype=self._A.dtype)
        np.cumsum(v[0], axis=1, out=cs[:, 1:, :])
        a = np.arange(q)
        lo = np.maximum(a - r, 0)
        hi = np.minimum(a + r, q - 1) + 1
        v[0] = cs[:, hi, :] - cs[:, lo, :]

    def _transform(self, out):
        """Run all n axis passes and reduce the layer stack into ``out``."""
        step = {"hamming": self._axis_hamming, "lee": self._axis_lee,
                "torus_linf": self._axis_torus,
                "grid_linf": self._axis_grid}[self.problem]
        for j in range(self.n):
            step(j)
        if self.nlayers == 1:
            out[:] = self._A[0]
        else:
            np.sum(self._A, axis=0, out=out)

    def _seed_layers(self, a0):
        self._A[0] = a0
        if self.nlayers > 1:
            self._A[1:] = 0

    # ================================================== fields
    def recount(self, idx):
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        self._A[0] = 0
        if idx.size:
            np.add.at(self._A[0], idx, 1)
        if self.nlayers > 1:
            self._A[1:] = 0
        self._transform(self.cnt)

    def gain_map(self):
        self._seed_layers(self._deficient())
        self._transform(self.map)

    def loss_map(self):
        self._seed_layers(self._critical())
        self._transform(self.map)

    # ================================================== incremental
    def ball_update(self, idx, delta):
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        step = self._chunk()
        for lo in range(0, len(idx), step):
            off, good = self._balls(idx[lo:lo + step])
            for k in range(len(off)):
                row = off[k] if good is None else off[k][good[k]]
                self.cnt[row] += delta

    def _role_of(self, values, target):
        """The coverage-role predicate applied to gathered cnt values.

        Identical to indexing :meth:`_role_mask` at those positions —
        the difference is only *where* the predicate runs, which is a
        cost decision (see :meth:`ball_gather`)."""
        if target == 0:
            return values < self.mu
        if target == 1:
            return (values >= 1) & (values <= self.mu)
        return values == target

    def ball_gather(self, idx, target):
        """target is a coverage ROLE: 0 = deficient (cnt < mu), 1 =
        critical (1 <= cnt <= mu), anything else = literal cnt ==
        target. At mu=1 all three are the literal count.

        Two ways to read the role off the field, picked by cost, never
        by semantics:

        * build the whole-space role mask once and gather **bytes** from
          it — one O(|X|) pass, then a cheap 1-byte-per-element gather;
        * gather the ``cnt`` **values** at the ball and apply the
          predicate to those — no O(|X|) pass at all, but the gather
          moves ``dtype``-sized elements instead of bytes.

        Measured on this machine the byte gather runs at about a third
        the cost per element of the value gather, so the mask pays for
        itself once the batch touches more than about half the space;
        below that — and a single-word loss query touches one ball —
        scanning |X| to answer a question about a few thousand words is
        pure waste. That single-word query is the polish's inner loop.
        """
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        out = np.empty(len(idx), dtype=np.int32)
        if not len(idx):
            return out
        wide = len(idx) * self.ball_len * _MASK_BREAKEVEN >= self.space.size
        mask = self._role_mask(target) if wide else None
        step = self._chunk()
        for lo in range(0, len(idx), step):
            off, good = self._balls(idx[lo:lo + step])
            hit = (mask[off] if wide
                   else self._role_of(self.cnt[off], target))
            if good is not None:
                hit &= good
            out[lo:lo + step] = hit.sum(axis=1)
        return out

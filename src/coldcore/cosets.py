"""Coset start for the Hamming problem (union of cosets plus a patch).

A code that is a union of cosets of a subgroup ``H`` of the alphabet
group ``A^n`` is described by its set ``S`` of syndromes in the quotient
``Q = A^n / H``. Translations by ``H`` are Hamming isometries, so the set
of covered words is ``H``-invariant, and a word ``x`` is covered exactly
when ``s(x)`` lies in ``S + D``, where ``D`` is the set of syndromes of
the words of weight at most ``R`` (the matrix method; Östergård 1991,
Carnielli 1985). Covering ``Q`` by translates of ``D`` is a covering
problem on ``|Q| = q^n / |H|`` points instead of ``q^n``.

This module builds such a start for ``hamming`` cells:

1. the alphabet ``Z_q`` is identified (as a set) with the product of
   elementary abelian groups ``F_p^e`` over the prime powers ``p^e`` in
   ``q``; any bijection of the alphabet is a Hamming isometry, so this
   only chooses which subgroups are available (for prime-power ``q``
   they include the ``GF(q)``-linear codes);
2. ``H`` is the kernel of a random full-rank syndrome map
   ``s = (P_p x_p)_p``;
3. ``S`` is built by exact greedy over ``Q``: the gain of every
   translate is read from one cyclic cross-correlation (FFT over ``Q``),
   so every placement takes an exact maximum;
4. the greedy is stopped at a chosen prefix, the cosets are expanded to
   words, and the remaining uncovered words are covered by the ordinary
   exact word-level greedy on the backend (the patch), followed by the
   peel.

The result is an ordinary code; it is checked by the backend's exact
recount like any other solution, so a bad choice of ``H`` can only give
a large code, never an invalid one.
"""
import time

import numpy as np

from .protocol import DenseFieldBackend
from .space import ProductSpace

#: Elements per index temporary in QuotientBackend's scatter / gather.
_CHUNK = 1 << 22


def prime_powers(q):
    """``q = prod p^e`` as a list of ``(p, e)``, primes ascending."""
    out, x, p = [], int(q), 2
    while p * p <= x:
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            out.append((p, e))
        p += 1
    if x > 1:
        out.append((x, 1))
    return out


def _rref_mod(M, p):
    """Row-reduced echelon form over F_p; returns (R, pivot columns)."""
    A = np.array(M, dtype=np.int64) % p
    rows, cols = A.shape
    piv = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.flatnonzero(A[r:, c])
        if not len(nz):
            continue
        k = r + int(nz[0])
        A[[r, k]] = A[[k, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] = (A[i] - A[i, c] * A[r]) % p
        piv.append(c)
        r += 1
    return A[:r], piv


class CosetModel:
    """A subgroup ``H`` of the alphabet group ``A^n`` given by a syndrome
    map, with the quotient ``Q`` and the syndrome set ``D`` of the ball.

    ``m`` maps each prime ``p`` of ``q`` to the number of ``F_p``
    syndrome digits; ``|Q| = prod p^m_p`` and ``|H| = q^n / |Q|``."""

    def __init__(self, q, n, R, m, rng):
        self.q, self.n, self.R = int(q), int(n), int(R)
        self.parts = prime_powers(q)
        self.P = {}
        for p, e in self.parts:
            mp = int(m.get(p, 0))
            cols = n * e
            if not 0 <= mp <= cols:
                raise ValueError(f"m[{p}]={mp} outside 0..{cols}")
            while True:
                Pm = rng.integers(0, p, size=(mp, cols))
                red, piv = _rref_mod(Pm, p)
                if len(piv) == mp:
                    break
            self.P[p] = (red, piv)
        # quotient radices, little-endian: prime blocks in ascending order
        self.rad = np.array([p for p, _ in self.parts
                             for _ in range(len(self.P[p][1]))],
                            dtype=np.int64)
        self.m = len(self.rad)
        self.size = int(np.prod(self.rad)) if self.m else 1
        self.h_size = q ** n // self.size
        pw, acc = [], 1
        for r in self.rad:
            pw.append(acc)
            acc *= int(r)
        self.pow = np.array(pw, dtype=np.int64)
        self._digit_vec = self._alphabet_vectors()
        # per-coordinate syndrome digits of the word with digit a at j
        self.T = np.zeros((n, q, self.m), dtype=np.int64)
        for j in range(n):
            for a in range(q):
                self.T[j, a] = self._syn_digits_single(j, a)
        self.D = self._ball_syndromes()

    # ---------------------------------------------------- alphabet maps
    def _alphabet_vectors(self):
        """digit a -> {p: e-vector over F_p} (CRT, then base p)."""
        out = []
        for a in range(self.q):
            v = {}
            for p, e in self.parts:
                c = a % (p ** e)
                v[p] = np.array([(c // p ** i) % p for i in range(e)],
                                dtype=np.int64)
            out.append(v)
        return out

    def _digit_from_vectors(self, comps):
        """Inverse of :meth:`_alphabet_vectors` for arrays of vectors:
        ``comps[p]`` has shape (k, e); returns (k,) digits."""
        k = len(next(iter(comps.values())))
        res = np.zeros(k, dtype=np.int64)
        mod = 1
        for p, e in self.parts:
            c = (comps[p] * (p ** np.arange(e))).sum(axis=1)
            pe = p ** e
            # CRT: find res' = res (mod mod), res' = c (mod pe)
            inv = pow(mod, -1, pe) if mod > 1 else 1
            k_ = ((c - res) % pe) * inv % pe
            res = res + mod * k_
            mod *= pe
        return res % self.q

    def _syn_digits_single(self, j, a):
        out = []
        for p, e in self.parts:
            red, piv = self.P[p]
            if not len(piv):
                continue
            vec = self._digit_vec[a][p]
            cols = red[:, j * e:(j + 1) * e]
            out.append((cols @ vec) % p)
        return (np.concatenate(out) if out
                else np.zeros(0, dtype=np.int64))

    # ---------------------------------------------------- syndromes
    def syndrome_digits(self, words):
        """(k, n) digit words -> (k, m) syndrome digits."""
        w = np.asarray(words, dtype=np.int64)
        s = np.zeros((len(w), self.m), dtype=np.int64)
        for j in range(self.n):
            s += self.T[j, w[:, j]]
        return s % self.rad

    def index(self, digs):
        return (np.asarray(digs, dtype=np.int64) % self.rad) @ self.pow

    def digits(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        return (idx[:, None] // self.pow[None, :]) % self.rad[None, :]

    def _nd(self, flat):
        """Flat little-endian Q array -> m-dim array, axis i = digit i."""
        return flat.reshape(tuple(self.rad[::-1])).transpose(
            tuple(range(self.m - 1, -1, -1))) if self.m else flat

    def _flat(self, nd):
        return (nd.transpose(tuple(range(self.m - 1, -1, -1))).ravel()
                if self.m else nd)

    def _ball_syndromes(self):
        """Indicator of D over Q.

        Small balls: the syndrome of every ball word, read off the
        packed support patterns (one row per word of weight <= R).
        Large balls: a DP over coordinates and weight on Q itself."""
        if not self.m:
            return np.ones(1, dtype=bool)
        from .patterns import ball_size
        nb = ball_size(self.q, self.n, self.R)
        if nb <= 4 * self.size or nb <= 1 << 22:
            return self._ball_syndromes_patterns()
        return self._ball_syndromes_dp()

    def _ball_syndromes_patterns(self, chunk=1 << 20):
        from .patterns import ball_patterns
        pats = ball_patterns(self.q, self.n, self.R)
        out = np.zeros(self.size, dtype=bool)
        for lo in range(0, len(pats), chunk):
            by = pats[lo:lo + chunk].view(np.uint8).reshape(-1, 8)
            acc = np.zeros((len(by), self.m), dtype=np.int64)
            for k in range(min(8, self.R)):
                pos = (by[:, k] >> 4).astype(np.int64)
                dl = (by[:, k] & 15).astype(np.int64)
                live = by[:, k] != 0
                acc[live] += self.T[pos[live], dl[live]]
            out[self.index(acc)] = True
        return out

    def _ball_syndromes_dp(self):
        lay = [np.zeros(self.size, dtype=bool) for _ in range(self.R + 1)]
        lay[0][0] = True
        for j in range(self.n):
            shifts = [self.T[j, a] for a in range(1, self.q)]
            for r in range(min(self.R, j + 1), 0, -1):
                src = self._nd(lay[r - 1])
                acc = self._nd(lay[r].copy())
                for t in shifts:
                    acc |= np.roll(src, shift=tuple(int(x) for x in t),
                                   axis=tuple(range(self.m)))
                lay[r] = self._flat(acc)
        out = lay[0]
        for r in range(1, self.R + 1):
            out = out | lay[r]
        return out

    # ---------------------------------------------------- coset words
    def coset_words(self, syn_idx):
        """All words of the cosets with these syndrome indices,
        as a (len(syn_idx) * |H|, n) digit array."""
        syn_idx = np.asarray(syn_idx, dtype=np.int64)
        if not len(syn_idx):
            return np.zeros((0, self.n), dtype=np.int64)
        sd = self.digits(syn_idx)
        comps = {}
        col = 0
        for p, e in self.parts:
            red, piv = self.P[p]
            cols = self.n * e
            mp = len(piv)
            # representative: pivot coordinates carry the syndrome
            rep = np.zeros((len(syn_idx), cols), dtype=np.int64)
            if mp:
                rep[:, piv] = sd[:, col:col + mp]
            col += mp
            # kernel basis of red over F_p
            free = [c for c in range(cols) if c not in piv]
            basis = np.zeros((len(free), cols), dtype=np.int64)
            for i, f in enumerate(free):
                basis[i, f] = 1
                for r, pc in enumerate(piv):
                    basis[i, pc] = (-red[r, f]) % p
            k = len(free)
            coef = np.stack(np.meshgrid(*[np.arange(p)] * k, indexing="ij"),
                            -1).reshape(-1, k) if k else np.zeros(
                                (1, 0), dtype=np.int64)
            hv = (coef @ basis) % p                    # (|H_p|, cols)
            comps[p] = (rep, hv)
        # combine: words = rep + h over all h in H = prod_p H_p
        hsizes = [len(comps[p][1]) for p, _ in self.parts]
        grids = np.stack(np.meshgrid(*[np.arange(s) for s in hsizes],
                                     indexing="ij"), -1).reshape(
                                         -1, len(hsizes))
        out = []
        for c in range(len(syn_idx)):
            vecs = {}
            for i, (p, e) in enumerate(self.parts):
                rep, hv = comps[p]
                full = (rep[c][None, :] + hv[grids[:, i]]) % p  # (|H|, n*e)
                vecs[p] = full.reshape(len(grids), self.n, e)
            digs = np.zeros((len(grids), self.n), dtype=np.int64)
            for j in range(self.n):
                digs[:, j] = self._digit_from_vectors(
                    {p: vecs[p][:, j, :] for p, _ in self.parts})
            out.append(digs)
        return np.concatenate(out)

    def incidence(self):
        """(|Q|, |D|) int array: row ``s`` lists the syndromes ``s + D``."""
        dig_D = self.digits(np.flatnonzero(self.D))
        allq = self.digits(np.arange(self.size, dtype=np.int64))
        out = np.empty((self.size, len(dig_D)), dtype=np.int64)
        for k in range(len(dig_D)):
            out[:, k] = self.index(allq + dig_D[k][None, :])
        return out

    # ---------------------------------------------------- quotient greedy
    def greedy(self, rng, stop_uncov=0, max_picks=None):
        """Exact greedy cover of Q by translates of D.

        Returns ``(picks, uncov_after)``: the syndromes in pick order and
        the number of uncovered syndromes after each pick. The gain of
        every translate ``s`` is ``|(s + D) & U|``, the cyclic
        cross-correlation of the uncovered indicator ``U`` with ``D``,
        computed for all ``s`` by one FFT pair over ``Q``."""
        shape = tuple(int(r) for r in self.rad)
        Dnd = self._nd(self.D.astype(np.float64))
        ax = tuple(range(self.m))
        FD = np.conj(np.fft.fftn(Dnd, s=shape, axes=ax)) if self.m else None
        U = np.ones(self.size, dtype=bool)
        uncov = self.size
        picks, trace = [], []
        dig_D = self.digits(np.flatnonzero(self.D))
        while uncov > stop_uncov:
            if max_picks is not None and len(picks) >= max_picks:
                break
            if self.m:
                g = np.fft.ifftn(np.fft.fftn(self._nd(U.astype(np.float64)),
                                             axes=ax) * FD, axes=ax).real
                g = np.rint(self._flat(g)).astype(np.int64)
            else:
                g = np.array([int(U[0])])
            mx = int(g.max())
            if mx <= 0:
                break
            cand = np.flatnonzero(g == mx)
            s = int(cand[rng.integers(len(cand))])
            cov = self.index(self.digits(np.array([s]))[0][None, :] + dig_D)
            U[cov] = False
            uncov -= mx
            picks.append(s)
            trace.append(uncov)
        return np.array(picks, dtype=np.int64), np.array(trace,
                                                         dtype=np.int64)


def choose_m(q, n, R, q_max=1 << 17):
    """Candidate syndrome-digit counts ``{p: m_p}`` with ``|Q| <= q_max``,
    from the largest quotient down (smaller ``H`` first)."""
    parts = prime_powers(q)
    out = []

    def rec(i, cur, size):
        if i == len(parts):
            if size > 1:
                out.append((size, dict(cur)))
            return
        p, e = parts[i]
        for mp in range(0, n * e + 1):
            if size * p ** mp > q_max:
                break
            cur[p] = mp
            rec(i + 1, cur, size * p ** mp)
        cur.pop(p, None)

    rec(0, {}, 1)
    out.sort(key=lambda t: -t[0])
    return [m for _, m in out]


class QuotientBackend(DenseFieldBackend):
    """The covering problem on ``Q`` by translates of ``D``, as a backend.

    Candidates and covered points are both the elements of ``Q`` (a
    product of cyclic groups of prime order, so ``space`` is an ordinary
    :class:`~coldcore.space.ProductSpace` with those radices); candidate
    ``s`` covers ``s + D``. The whole-space fields are cyclic
    correlations with the indicator of ``D``, computed by FFT over
    ``Q`` and rounded (the values are integers bounded by ``|D|``, far
    inside float64 precision); ``ball_update`` and ``ball_gather`` are
    exact index gathers over ``s + D``. The search core runs on it
    unchanged. ``indep_dist`` is ``None``: digit distance in ``Q`` says
    nothing about overlap of translates, so the peel removes one
    candidate at a time."""

    indep_dist = None
    supports_hist = True
    #: a field is one FFT over Q, cheaper than scoring a batch of
    #: candidates exactly (|D| reads each), so the greedy takes every
    #: placement straight off a fresh gain map (Searcher._field_fill)
    field_greedy = True

    def __init__(self, model, extract_cap=4096):
        self.model = model
        self.space = ProductSpace([int(r) for r in model.rad])
        self.mu = 1
        self.extract_cap = extract_cap
        # the flat little-endian field reshaped in C order: numpy axis j
        # is digit m-1-j, so the radix-2 digits (lowest, primes ascending)
        # are the last numpy axes. Axis order is irrelevant to a cyclic
        # convolution, so no transpose is needed.
        self._shape = tuple(int(r) for r in model.rad[::-1]) or (1,)
        self._m2 = int((model.rad == 2).sum())
        self._oth = tuple(range(len(model.rad) - self._m2))
        self._dig_D = model.digits(np.flatnonzero(model.D))
        self._FD = self._fwd(model.D)
        self.cnt = np.zeros(model.size, dtype=np.int32)
        self.map = np.zeros(model.size, dtype=np.int32)

    @staticmethod
    def _wht_last(a, k):
        """Unnormalised Walsh-Hadamard transform over the last ``k`` axes
        (each of length 2) of a C-contiguous array."""
        shp = a.shape
        flat = np.ascontiguousarray(a).reshape(-1)
        h = 1
        for _ in range(k):
            v = flat.reshape(-1, 2, h)
            flat = np.concatenate((v[:, :1] + v[:, 1:], v[:, :1] - v[:, 1:]),
                                  axis=1).reshape(-1)
            h *= 2
        return flat.reshape(shp)

    def _fwd(self, field):
        """Character transform of a real field on Q: Walsh-Hadamard on the
        radix-2 digits, real FFT on the others."""
        a = self._wht_last(np.asarray(field, dtype=np.float64).reshape(
            self._shape), self._m2)
        if self._oth:
            a = np.fft.rfftn(a, axes=self._oth)
        return a

    def _inv(self, F):
        if self._oth:
            F = np.fft.irfftn(F, s=[self._shape[i] for i in self._oth],
                              axes=self._oth)
        return (self._wht_last(F, self._m2) / float(1 << self._m2)).reshape(-1)

    def _conv(self, mask, corr):
        """Cyclic convolution (``corr=False``) or correlation with D."""
        F = self._fwd(mask)
        F *= np.conj(self._FD) if corr else self._FD
        return np.rint(self._inv(F)).astype(np.int32)

    def _translates(self, idx):
        """Index array (k, |D|) of ``s + D`` for each s in idx."""
        m = self.model
        dig = m.digits(np.atleast_1d(np.asarray(idx, dtype=np.int64)))
        out = np.zeros((len(dig), len(self._dig_D)), dtype=np.int64)
        for i in range(m.m):               # digit-wise, no (k, |D|, m) temp
            r = int(m.rad[i])
            t = dig[:, i][:, None] + self._dig_D[None, :, i]
            t[t >= r] -= r
            out += t * int(m.pow[i])
        return out

    def _chunks(self, idx):
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64)).ravel()
        step = max(1, _CHUNK // max(1, len(self._dig_D)))
        for lo in range(0, len(idx), step):
            yield lo, idx[lo:lo + step]

    def recount(self, idx):
        ind = np.zeros(self.model.size, dtype=np.float64)
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        np.add.at(ind, idx, 1.0)
        self.cnt = self._conv(ind, corr=False)

    def gain_map(self):
        self.map = self._conv(self._deficient(), corr=True)

    def loss_map(self):
        self.map = self._conv(self._critical(), corr=True)

    def ball_update(self, idx, delta):
        for _lo, part in self._chunks(idx):
            np.add.at(self.cnt, self._translates(part).ravel(), int(delta))

    def ball_gather(self, idx, target):
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64)).ravel()
        out = np.empty(len(idx), dtype=np.int32)
        mask = self._role_mask(int(target))
        for lo, part in self._chunks(idx):
            out[lo:lo + len(part)] = mask[self._translates(part)].sum(
                axis=1, dtype=np.int32)
        return out


def _quotient_search(model, budget_s, seed, greedy_only=False):
    """Greedy + peel (+ notch descent unless ``greedy_only``) on the
    quotient problem within ``budget_s`` seconds; returns the smallest
    syndrome set found that covers ``Q``, or ``None``."""
    from .search import Searcher
    qb = QuotientBackend(model)
    best = {}

    def keep(M, idx):
        if "M" not in best or M < best["M"]:
            best["M"], best["S"] = M, np.asarray(idx, dtype=np.int64).copy()

    s = Searcher(qb, seed=seed, deadline=time.time() + budget_s,
                 on_solved=keep)
    s.loop_reserve_s = s.build_reserve_s = 0.0
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    if s.uncov != 0:
        return None
    s.peel()
    keep(len(s.code), s.code)
    if not greedy_only and s.time_left() > 0.5:
        s.descend(floor=1, notch_budget_s=max(0.5, budget_s / 8),
                  step0=1, kmin=1, salvage=True, persist=True,
                  reserve_s=0.0, min_notch_s=0.2)
    return best.get("S")


def h_sizes(q, n, m_ref, q_max=1 << 21, lo=2, hi=400):
    """Subgroup orders ``|H|`` for which a code of about ``m_ref`` words
    is a union of ``lo``..``hi`` cosets and ``|Q| <= q_max``; each with
    its syndrome-digit counts ``{p: m_p}``. Ordered by ``|H|``."""
    out = {}
    for m in choose_m(q, n, 0, q_max):
        N = int(np.prod([p ** v for p, v in m.items()]))
        h = q ** n // N
        if h >= 2 and lo <= m_ref / h <= hi:
            out.setdefault(h, m)
    return sorted(out.items())


class CosetSearch:
    """The coset start as a resumable search over subgroups.

    Each :meth:`run` call spends its budget in two phases. Screening
    (``screen_share`` of the call): round-robin over the subgroup orders
    of :func:`h_sizes` (largest ``|H|``, i.e. cheapest quotient, first),
    one subgroup per visit (the best of ``draws`` random syndrome maps by
    ``|D|``), scored by the exact greedy on the quotient, expanded to
    words and peeled. Search (the rest): quotient notch descent on the
    screened subgroups not yet searched, best score first, each for
    ``max(min_search_s, search_factor * its screening cost)`` seconds;
    a subgroup order whose best screened code after two visits is more
    than twice the reference size is not screened again;
    only subgroups whose screening score is within ``search_slack`` of
    the reference size (the smaller of ``m_ref`` and the best coset
    code) are searched, and a call whose screening finds none returns
    early. A quotient cover larger than ``1 + expand_slack`` times the
    reference size is scored by its size and not expanded. State
    (scores, costs, best code) carries over between calls.

    Every candidate is expanded, loaded on the word-level backend,
    peeled and verified there with an exact recount."""

    def __init__(self, searcher, q, n, R, m_ref, draws=4, q_max=1 << 21,
                 min_search_s=3.0, search_factor=5.0, search_slack=0.05,
                 expand_slack=0.25, on_log=None):
        self.s = searcher
        self.q, self.n, self.R = int(q), int(n), int(R)
        self.m_ref = int(m_ref)
        self.draws = int(draws)
        self.min_search_s = float(min_search_s)
        self.search_factor = float(search_factor)
        self.search_slack = float(search_slack)
        self.expand_slack = float(expand_slack)
        self.log = on_log or (lambda s_: None)
        self.sizes = h_sizes(q, n, m_ref, q_max)[::-1]
        self.cost = {}                        # |H| -> seconds per screen
        self.visits = {}                      # |H| -> screening visits
        self.order_best = {}                  # |H| -> best screened size
        self.scored = []                      # [M, trial, cost, model]
        self.trial = 0
        self.best_m = None
        self.best_code = None
        self.t0 = time.time()

    @property
    def productive(self):
        """True when some coset code is smaller than the reference."""
        return self.best_m is not None and self.best_m < self.m_ref

    def _word_size(self, model, S, tag):
        raw = len(S) * model.h_size
        ref = min(self.m_ref, self.best_m or self.m_ref)
        if raw > ref * (1.0 + self.expand_slack):
            # the word-level peel removes a few words, not a quarter of
            # the code; expanding and peeling a code this large is only
            # cost (on a 1e9-word space, minutes)
            self.log(f"coset {tag}: |H|={model.h_size} |Q|={model.size} "
                     f"|D|={int(model.D.sum())} cosets={len(S)} -> "
                     f"{raw} words before peel, not expanded "
                     f"({time.time() - self.t0:.0f}s)")
            return raw
        s = self.s
        s.load(s.b.word_index(model.coset_words(S)))
        if s.uncov != 0:                    # cannot happen; checked anyway
            self.log(f"coset {tag}: expansion is not a cover (bug?)")
            return None
        s.peel()
        if s.verify_full() != 0:
            return None
        M = len(s.code)
        self.log(f"coset {tag}: |H|={model.h_size} |Q|={model.size} "
                 f"|D|={int(model.D.sum())} cosets={len(S)} -> M={M} "
                 f"({time.time() - self.t0:.0f}s)")
        if self.best_m is None or M < self.best_m:
            self.best_m, self.best_code = M, s.code.copy()
        return M

    def _screen(self, until):
        rng = self.s.rng
        while time.time() < until:
            left = until - time.time()
            ref = min(self.m_ref, self.best_m or self.m_ref)
            live = [(h, m) for h, m in self.sizes
                    if self.cost.get(h, 0.0) <= left
                    and not (self.visits.get(h, 0) >= 2
                             and self.order_best.get(h, np.inf) > 2 * ref)]
            if not live:
                return
            h, m = live[self.trial % len(live)]
            self.trial += 1
            t1 = time.time()
            model = max((CosetModel(self.q, self.n, self.R, m, rng)
                         for _ in range(self.draws)),
                        key=lambda md: int(md.D.sum()))
            left = until - time.time()          # after building the models
            if left <= 0:
                return
            S = _quotient_search(model, left, int(rng.integers(1 << 30)),
                                 greedy_only=True)
            self.cost[h] = time.time() - t1
            self.visits[h] = self.visits.get(h, 0) + 1
            if S is None:
                continue
            M = self._word_size(model, S, f"screen {self.trial}")
            if M is not None:
                self.order_best[h] = min(self.order_best.get(h, M), M)
            if M is not None:
                self.scored.append([M, self.trial, self.cost[h], model])

    def _worth(self):
        ref = min(self.m_ref, self.best_m or self.m_ref)
        lim = ref * (1.0 + self.search_slack)
        return sorted((x for x in self.scored if x[0] <= lim),
                      key=lambda x: (x[0], x[1]))

    def run(self, budget_s, screen_share=0.5):
        """Spend ``budget_s`` seconds; returns the best verified coset
        code so far (sorted indices) or ``None``. The searcher is left
        loaded with that code when there is one."""
        if not self.sizes:
            return None
        end = time.time() + budget_s
        self._screen(time.time() + budget_s * float(screen_share))
        rng = self.s.rng
        while time.time() < end - 1.0:
            todo = self._worth()
            if not todo:
                break
            x = todo[0]
            self.scored.remove(x)          # searched once, then dropped
            _M, tr, c, model = x
            span = min(end - time.time(),
                       max(self.min_search_s, self.search_factor * c))
            S = _quotient_search(model, span, int(rng.integers(1 << 30)))
            if S is not None:
                self._word_size(model, S, f"search {tr}")
            if not self._worth() and time.time() < end - 1.0:
                self._screen(min(end, time.time() + max(
                    5.0, 0.25 * (end - time.time()))))
        if self.best_code is not None:
            self.s.load(self.best_code)
        return self.best_code


def coset_stage(searcher, q, n, R, budget_s, m_ref, on_log=None, **kw):
    """One-shot :class:`CosetSearch` run of ``budget_s`` seconds."""
    return CosetSearch(searcher, q, n, R, m_ref, on_log=on_log,
                       **kw).run(budget_s)

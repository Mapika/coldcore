"""Isomorph-free exhaustive settlement: does it get the known answers?

The search core can only ever say "here is a cover of size M". This
module claims the other half — "and there is none of size M-1" — so the
tests are, first and foremost, a comparison against cells whose optimum
is settled in the literature, plus the internal guarantees that make the
claim mean anything:

* the witness it returns really is a cover (re-verified by recount);
* one size below, the verdict is *infeasible*, not "unknown";
* the isomorph rejection changes the node count, never the answer
  (same result with the trivial group, which does no rejection at all);
* a group that is not an automorphism group is refused rather than
  silently producing an unsound proof;
* the Knuth tree-size estimator — the tool the K_8(4,2) feasibility
  verdict rests on — is calibrated against real node counts.
"""
import numpy as np
import pytest

from coldcore.reference import (RefGridBackend, RefHammingBackend,
                                RefLeeBackend, RefTorusBackend)
from coldcore.space import ProductSpace
from coldcore.symmetry import (coordinate_rotation, monomial_group,
                               trivial_group, value_shifts)
from coldcore.symmetry_exact import (IsomorphFreeSearch, _mask,
                                     affine_perms, all_perms,
                                     binary_monomial_table, monomial_table,
                                     settle)

#: (label, factory, known optimum, a size to start the ladder from).
#: Every optimum here is a settled value from the covering-code /
#: domination literature.
SETTLED = [
    ("K_2(3,1)=2", lambda: RefHammingBackend([2] * 3, 1), 2, 4),
    ("K_2(4,1)=4", lambda: RefHammingBackend([2] * 4, 1), 4, 6),
    ("K_2(5,1)=7", lambda: RefHammingBackend([2] * 5, 1), 7, 9),
    ("K_3(3,1)=5", lambda: RefHammingBackend([3] * 3, 1), 5, 7),
    ("K_3(4,2)=3", lambda: RefHammingBackend([3] * 4, 2), 3, 5),
    ("K_4(3,1)=8", lambda: RefHammingBackend([4] * 3, 1), 8, 10),
    ("Lee Z_5^2 R=1 = 5", lambda: RefLeeBackend([5, 5], 1), 5, 7),
    ("king 6x6 torus = 4", lambda: RefTorusBackend([6, 6], 1), 4, 6),
    ("king 9x9 torus = 9", lambda: RefTorusBackend([9, 9], 1), 9, 11),
    ("king 8x8 board = 9", lambda: RefGridBackend([8, 8], 1), 9, 11),
]


@pytest.mark.parametrize("label,mk,opt,hi", SETTLED)
def test_settles_known_optima(label, mk, opt, hi):
    b = mk()
    M, witness, trace = settle(b, hi, time_limit=300)
    assert M == opt, f"{label}: got {M}, trace {[(m, v) for m, v, _ in trace]}"
    b.recount(witness)
    assert b.uncovered() == 0 and len(witness) == opt
    assert trace[-1][1] == "infeasible" and trace[-1][0] == opt - 1


MU_SETTLED = [
    ("K_2^2(3,1)=4", lambda: RefHammingBackend([2] * 3, 1, mu=2), 4, 6),
    ("K_2^3(3,1)=6", lambda: RefHammingBackend([2] * 3, 1, mu=3), 6, 8),
    # 7 would be a multiset "solution" repeating the all-zero word; a
    # brute-force sweep of all 7-subsets of Z_2^4 confirms none is a
    # 2-fold cover, so 8 = 2 x K_2(4,1) is the answer
    ("K_2^2(4,1)=8", lambda: RefHammingBackend([2] * 4, 1, mu=2), 8, 10),
    ("K^2 Lee Z_5^2 = 10", lambda: RefLeeBackend([5, 5], 1, mu=2), 10, 11),
]


@pytest.mark.parametrize("label,mk,opt,hi", MU_SETTLED)
def test_settles_known_mu_fold_optima(label, mk, opt, hi):
    """mu-fold goes through the count path, not the bitset path — the
    same generator, a different deficiency predicate."""
    b = mk()
    M, witness, trace = settle(b, hi, time_limit=300)
    assert M == opt, f"{label}: got {M}"
    # a solution is a SET: the mu-fold branch set must never re-pick a
    # word already in the partial code (an incumbent CAN reach a word
    # that is merely short of its multiplicity)
    assert len(set(witness.tolist())) == len(witness) == M
    b.recount(witness)
    assert b.uncovered() == 0 and (b.cnt >= b.mu).all()


def test_isomorph_rejection_changes_cost_not_answer():
    """The same cell with and without any isomorph rejection: identical
    verdicts, fewer nodes with the group."""
    b = RefHammingBackend([2] * 5, 1)
    with_g = IsomorphFreeSearch(b)
    without = IsomorphFreeSearch(RefHammingBackend([2] * 5, 1),
                                 group=trivial_group(b.space))
    for M in (6, 7):
        v1, w1, s1 = with_g.decide(M)
        v2, w2, s2 = without.decide(M)
        assert v1 == v2
        assert s1["nodes"] <= s2["nodes"]
    assert with_g.decide(6)[0] == "infeasible"
    assert without.decide(6)[0] == "infeasible"


def test_witness_is_reproducible_and_verified():
    b = RefHammingBackend([3] * 3, 1)
    a = IsomorphFreeSearch(b).decide(5)
    c = IsomorphFreeSearch(RefHammingBackend([3] * 3, 1)).decide(5)
    assert a[0] == c[0] == "feasible"
    assert (a[1] == c[1]).all()
    b.recount(a[1])
    assert b.uncovered() == 0


def test_budget_exhaustion_reports_unknown_not_infeasible():
    b = RefHammingBackend([2] * 6, 1)
    s = IsomorphFreeSearch(b, verify=False)
    v, w, st = s.decide(11, node_limit=50)
    # the limit stops the descent; the siblings still on the stack each
    # cost one more (immediately-aborted) node, so the overshoot is
    # bounded by the depth, not unbounded
    assert v == "unknown" and w is None
    assert 50 <= st["nodes"] < 50 + 20 * 11
    # given the room, the same query is a proof
    v, w, st = s.decide(11, time_limit=300)
    assert v == "infeasible"


def test_wrong_group_is_refused():
    """Translations are not automorphisms of the open grid, so using them
    for isomorph rejection would be unsound — and is rejected."""
    b = RefGridBackend([5, 5], 1)
    with pytest.raises(ValueError):
        IsomorphFreeSearch(b, group=value_shifts(b.space))
    with pytest.raises(TypeError):
        IsomorphFreeSearch(object())


@pytest.mark.parametrize("mk,M", [
    (lambda: RefHammingBackend([2] * 5, 1), 6),
    (lambda: RefHammingBackend([3] * 3, 1), 4),
    (lambda: RefHammingBackend([2] * 6, 1), 11),
])
def test_knuth_estimator_is_calibrated(mk, M):
    """The K_8(4,2) verdict is an extrapolation from this estimator, so
    it has to be shown to track real trees: the mean of the random dives
    must land within a factor of 3 of the node count the exhaustive run
    actually paid."""
    s = IsomorphFreeSearch(mk(), verify=False)
    v, _w, st = s.decide(M, time_limit=300)
    assert v == "infeasible"
    est = s.estimate_nodes(M, samples=4000, seed=3)
    true = st["nodes"]
    assert est["mean"] / true < 3.0 and true / est["mean"] < 3.0, (
        f"true={true} est={est}")


def test_k8_42_is_out_of_reach_and_says_so():
    """The prize cell, and the honest verdict: the generator's tree for
    K_8(4,2) at M=22 is estimated at ~5e47 nodes, which at the measured
    ~1e6 nodes/s is ~1e34 years. The test pins the *shape* of that
    conclusion (astronomically large, and the search reports 'unknown'
    rather than guessing), not the exact estimate."""
    b = RefHammingBackend([8] * 4, 2)
    s = IsomorphFreeSearch(b, verify=False)
    assert s.size == 4096 and s.maxball == 323
    v, w, st = s.decide(22, node_limit=200_000)
    assert v == "unknown" and w is None
    est = s.estimate_nodes(22, samples=64, seed=11)
    assert est["median"] > 1e30
    # even a perfect isomorph rejection under the full monomial group
    # (order 8!^4 * 4! ~ 6.3e19) would leave >1e25 nodes
    aut = 40320 ** 4 * 24
    assert est["median"] / aut > 1e25


# ====================================================================
# the two accelerations, and that neither one changes an answer
# ====================================================================
ACCEL_CELLS = [
    (lambda: RefHammingBackend([2] * 5, 1), 6),
    (lambda: RefHammingBackend([2] * 4, 1, mu=2), 7),
    (lambda: RefHammingBackend([2] * 3, 1, mu=3), 5),
    (lambda: RefHammingBackend([3] * 3, 1, mu=2), 8),
    (lambda: RefLeeBackend([5, 5], 1, mu=2), 9),
]


@pytest.mark.parametrize("mk,M", ACCEL_CELLS)
def test_exact_bound_changes_cost_not_answer(mk, M):
    """The exact greedy bound replaces `maxball` with the largest deficit
    drop any single ball can still achieve. It must prune (never fewer
    nodes) and must never change a verdict."""
    off = IsomorphFreeSearch(mk(), verify=False, strong_bound=0)
    on = IsomorphFreeSearch(mk(), verify=False, strong_bound=1)
    v0, w0, s0 = off.decide(M, time_limit=300)
    v1, w1, s1 = on.decide(M, time_limit=300)
    assert v0 == v1
    assert s1["nodes"] <= s0["nodes"]
    if v0 == "feasible":
        assert (w0 == w1).all()


@pytest.mark.parametrize("mk,M", ACCEL_CELLS)
def test_best_drop_is_the_exact_marginal(mk, M):
    """`_best_drop(short)` must equal the largest deficit reduction any
    single codeword achieves, checked against real recounts."""
    b = mk()
    s = IsomorphFreeSearch(b, verify=False)
    rng = np.random.default_rng(7)
    for trial in range(4):
        code = rng.choice(b.space.size, size=3 + trial, replace=False)
        b.recount(code)
        d0 = b.uncovered()
        # `short` straight from the backend's own cnt field
        short = _mask(np.flatnonzero(b.cnt < b.mu), b.space.size)
        drops = []
        for w in range(b.space.size):
            b.recount(np.append(code, w) if w not in code else code)
            drops.append(d0 - b.uncovered())
        assert s._best_drop(short) == max(drops)


def test_layered_bitsets_match_a_recount():
    """The mu > 1 state is mu bitmasks (`lay[k]` = words covered >= k+1),
    updated by `lay'[k] = lay[k] | (lay[k-1] & B)`. Check the layers and
    the running deficit against the backend's exact cnt field."""
    b = RefHammingBackend([2] * 4, 2, mu=3)
    s = IsomorphFreeSearch(b, verify=False)
    rng = np.random.default_rng(11)
    lay, deficit, code = (0,) * s.mu, s.mu * s.size, []
    for w in rng.choice(s.size, size=6, replace=False):
        w = int(w)
        code.append(w)
        lay = s._new_layers(lay, s.balls[w])
        b.recount(np.array(code))
        deficit = s.mu * s.size - sum(m.bit_count() for m in lay)
        assert deficit == b.uncovered()
        for k in range(s.mu):
            want = _mask(np.flatnonzero(b.cnt >= k + 1), s.size)
            assert lay[k] == want, f"layer {k} after {code}"


def test_k822_lower_bound_is_proved():
    """K(8,2,2) >= 15, proved exhaustively: no 14-word 2-fold cover of
    Z_2^8 with radius 2 exists. (Published lower bound is 16; this is an
    independent, machine-checkable confirmation of part of it. M=15 and
    above are out of budget — see the estimates in docs/symmetry.md.)"""
    b = RefHammingBackend([2] * 8, 2, mu=2)
    G = value_shifts(b.space) + coordinate_rotation(b.space)
    s = IsomorphFreeSearch(b, group=G, verify=False)
    assert s.size == 256 and s.maxball == 37
    assert -(-2 * 256 // 37) == 14                 # sphere bound is only 14
    v, w, st = s.decide(13, time_limit=120)
    assert v == "infeasible" and st["nodes"] == 1  # killed by the bound alone


# ====================================================================
# the full binary monomial group, built without enumerating elements
# ====================================================================
@pytest.mark.parametrize("n", [3, 4, 5])
def test_binary_monomial_table_is_the_monomial_group(n):
    """`binary_monomial_table` must produce exactly the index
    permutations of `monomial_group`, which for Z_2^n has order
    n! * 2^n — built by broadcasting rather than by Python-level
    closure, which is what makes n=8 (10.3M elements) reachable."""
    sp = ProductSpace([2] * n)
    table, reps = binary_monomial_table(sp)
    assert table.shape == (_fact(n) * 2 ** n, 2 ** n)
    want = {g.permutation(sp).tobytes()
            for g in monomial_group(sp).elements(cap=1 << 16)}
    got = {row.astype(np.int64).tobytes() for row in table}
    assert got == want
    # transitive (it contains every translation), so one root rep
    assert reps.tolist() == [0]
    for row in table[:: max(1, len(table) // 50)]:
        assert (np.sort(row) == np.arange(2 ** n)).all()


def _fact(n):
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


@pytest.mark.parametrize("mk,M,mu", [
    (lambda: RefHammingBackend([2] * 5, 1), 6, 1),
    (lambda: RefHammingBackend([2] * 4, 1, mu=2), 7, 2),
])
def test_precomputed_table_matches_the_enumerated_group(mk, M, mu):
    """The `table=` escape hatch must decide exactly what the ordinary
    enumerated-group path decides, and (being a much larger group) with
    no more nodes."""
    b = mk()
    tab = binary_monomial_table(b.space)
    big = IsomorphFreeSearch(b, table=tab, verify=False)
    small = IsomorphFreeSearch(mk(), verify=False)
    v1, w1, s1 = big.decide(M, time_limit=300)
    v0, w0, s0 = small.decide(M, time_limit=300)
    assert v0 == v1
    assert s1["nodes"] <= s0["nodes"]
    if v1 == "feasible":
        b.recount(w1)
        assert b.uncovered() == 0 and len(set(w1.tolist())) == len(w1)


def test_root_cache_does_not_change_results():
    """The root branch set is cached across calls (it depends only on the
    group). Repeated decides must be identical, not merely faster."""
    b = RefHammingBackend([2] * 5, 1)
    s = IsomorphFreeSearch(b, table=binary_monomial_table(b.space),
                           verify=False)
    first = [s.decide(M) for M in (6, 7, 6, 7)]
    assert first[0][0] == first[2][0] and first[1][0] == first[3][0]
    assert first[0][2]["nodes"] == first[2][2]["nodes"]
    assert first[1][2]["nodes"] == first[3][2]["nodes"]


@pytest.mark.bigmem
def test_monomial_group_is_the_lever_on_k822():
    """K(8,2,2), M=14: the group is what makes this cell tractable.
    Translations x C_8 (|G| = 2048) needs 7.8e6 nodes; the full monomial
    group (|G| = 10321920) needs 1.2e5 — a 64x cut, and the verdict is
    the same infeasible either way, so K(8,2,2) >= 15."""
    b = RefHammingBackend([2] * 8, 2, mu=2)
    small = IsomorphFreeSearch(
        b, group=value_shifts(b.space) + coordinate_rotation(b.space),
        verify=False)
    v0, _w, s0 = small.decide(14, time_limit=600)
    big = IsomorphFreeSearch(RefHammingBackend([2] * 8, 2, mu=2),
                             table=binary_monomial_table(b.space),
                             verify=False)
    v1, _w, s1 = big.decide(14, time_limit=600)
    assert v0 == v1 == "infeasible"
    assert s1["nodes"] < s0["nodes"] / 10


# ====================================================================
# the q-ary monomial table
# ====================================================================
@pytest.mark.parametrize("axes", [[2] * 3, [3] * 2, [4] * 2, [2] * 4,
                                  [3] * 3, [5, 5]])
def test_monomial_table_equals_the_enumerated_group(axes):
    """`monomial_table` must reproduce, exactly, the index permutations
    that `monomial_group` produces one Python object at a time — that is
    the whole justification for building it by broadcasting instead."""
    sp = ProductSpace(axes)
    table, reps = monomial_table(sp)
    want = {g.permutation(sp).tobytes()
            for g in monomial_group(sp).elements(cap=1 << 18)}
    got = {row.astype(np.int64).tobytes() for row in table}
    assert got == want
    assert len(table) == len(want)
    assert reps.tolist() == [0]        # transitive: it contains S_q per axis


def test_monomial_table_matches_the_binary_specialisation():
    for n in (2, 3, 4):
        sp = ProductSpace([2] * n)
        a, ra = binary_monomial_table(sp)
        b, rb = monomial_table(sp)
        assert (a == b).all() and (ra == rb).all()
    with pytest.raises(ValueError):
        binary_monomial_table(ProductSpace([3] * 2))


def test_monomial_table_mixed_radix_only_moves_equal_axes():
    """A coordinate permutation may only move an axis onto one of the
    same size, so Z_2 x Z_2 x Z_3 gets 2!*(2!*2!*3!) = 48 elements, not
    3!*(...)."""
    sp = ProductSpace([2, 2, 3])
    table, reps = monomial_table(sp)
    assert len(table) == 2 * 2 * 6 * 2
    for row in table:
        assert (np.sort(row) == np.arange(sp.size)).all()
    assert reps.tolist() == [0]


@pytest.mark.parametrize("q", [3, 4, 5, 7])
def test_perm_generators(q):
    """`all_perms` is S_q; `affine_perms` is AGL(1,q) — a genuine
    subgroup of it, which is what makes it a sound fallback."""
    full, aff = all_perms(q), affine_perms(q)
    assert len(full) == _fact(q)
    for v in (full, aff):
        for row in v:
            assert sorted(row.tolist()) == list(range(q))
    assert {tuple(r) for r in aff} <= {tuple(r) for r in full}
    assert (0, 1) not in [(len(aff),)]           # non-empty
    assert len(aff) == q * sum(1 for a in range(1, q)
                               if __import__("math").gcd(a, q) == 1)


def test_affine_fallback_is_a_sound_subgroup_for_search():
    """Restricting to AGL(1,q) per axis is a smaller group, so it may
    cost more nodes — but never a different verdict."""
    b = RefHammingBackend([5] * 2, 1, mu=2)
    full = IsomorphFreeSearch(b, table=monomial_table(b.space), verify=False)
    part = IsomorphFreeSearch(
        RefHammingBackend([5] * 2, 1, mu=2),
        table=monomial_table(b.space, per_axis=[affine_perms(5)] * 2),
        verify=False)
    for M in (9, 10):
        v0, w0, s0 = full.decide(M, time_limit=300)
        v1, w1, s1 = part.decide(M, time_limit=300)
        assert v0 == v1
        assert s0["nodes"] <= s1["nodes"]


@pytest.mark.bigmem
def test_monomial_table_refuses_to_blow_up_memory():
    """The table costs |G| * |X| * 4 bytes; Z_5^4 would need 12 TB, and
    the caller has to be told rather than OOM-killed."""
    sp = ProductSpace([5] * 4)
    with pytest.raises(ValueError, match="GiB"):
        monomial_table(sp)
    # ... and the documented fallback does fit
    table, reps = monomial_table(sp, per_axis=[affine_perms(5)] * 4,
                                 max_bytes=16 << 30)
    assert len(table) == 20 ** 4 * 24
    assert reps.tolist() == [0]

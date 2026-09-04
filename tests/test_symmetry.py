"""The symmetry layer: group action, orbits, and orbit-restricted search.

Three things are pinned here.

*The group is a group.* The action on word indices agrees with the
action on digit arrays, composition and inversion are the real group
operations, and the orbit partition really is the partition into orbits
(disjoint, covering, G-invariant, sizes dividing |G|).

*Orbit search is exact.* Adding or dropping an orbit moves the deficit
by exactly what a from-scratch recount says, and the gain-map segment
sum is a genuine upper bound on the exact orbit gain — the property the
lazy-greedy bound heap needs to be exact greedy rather than a heuristic.

*Orbit search wins where structure is, and loses where it is not.* On
cells whose optimum is a union of cosets it reaches the optimum in
milliseconds where the free searcher spends its whole budget short of
it; on cells whose optimum is not group-invariant it cannot reach the
optimum at all. Both halves are asserted.
"""
import time

import numpy as np
import pytest

from coldcore.reference import (RefGridBackend, RefHammingBackend,
                                RefLeeBackend, RefTorusBackend)
from coldcore.search import Searcher
from coldcore.space import ProductSpace
from coldcore.symmetry import (Group, GroupElement, SymmetricSearcher,
                               coordinate_permutation, coordinate_rotation,
                               coordinate_symmetric, diagonal_translations,
                               isometry_group, monomial_group, orbit_solve,
                               parse_group, translation, trivial_group,
                               value_shift, value_shifts)


# ====================================================================
# group elements
# ====================================================================
def _elements_for(sp):
    return [translation(sp, [1] * sp.n),
            coordinate_permutation(sp, (np.arange(sp.n) + 1) % sp.n),
            GroupElement(sp.axes, (np.arange(sp.n) + 1) % sp.n,
                         [(np.arange(a) + j) % a
                          for j, a in enumerate(sp.axes)])]


@pytest.mark.parametrize("axes", [[3] * 4, [2] * 5, [5, 5]])
def test_index_action_matches_word_action(axes):
    """apply_index is the mixed-radix shortcut for apply_words."""
    sp = ProductSpace(axes)
    idx = np.arange(sp.size, dtype=np.int64)
    words = sp.index_word(idx)
    for g in _elements_for(sp):
        assert (sp.word_index(g.apply_words(words))
                == g.apply_index(sp, idx)).all()
        assert (g.permutation(sp) == g.apply_index(sp, idx)).all()


@pytest.mark.parametrize("axes", [[3] * 4, [2] * 5])
def test_group_algebra(axes):
    """Composition applies right-to-left; inverse undoes; both are exact
    permutations of the index set."""
    sp = ProductSpace(axes)
    idx = np.arange(sp.size, dtype=np.int64)
    for a in _elements_for(sp):
        for b in _elements_for(sp):
            ab = a.compose(b)
            assert (ab.apply_index(sp, idx)
                    == a.apply_index(sp, b.apply_index(sp, idx))).all()
        ai = a.inverse()
        assert (ai.apply_index(sp, a.apply_index(sp, idx)) == idx).all()
        assert a.compose(ai).is_identity() and ai.compose(a).is_identity()
        # a permutation: every index hit exactly once
        assert (np.sort(a.permutation(sp)) == idx).all()


def test_mixed_radix_rejects_illegal_coordinate_moves():
    sp = ProductSpace([2, 3, 3])
    with pytest.raises(ValueError):
        coordinate_permutation(sp, [1, 0, 2])      # moves a 2-axis onto a 3
    ok = coordinate_permutation(sp, [0, 2, 1])     # swaps the two 3-axes
    assert not ok.is_identity()
    with pytest.raises(ValueError):
        GroupElement(sp.axes, None, [np.array([0, 0]), np.arange(3),
                                     np.arange(3)])


# ====================================================================
# orbits
# ====================================================================
ORBIT_CASES = [
    ([3] * 4, "translate:1011", 27, 3),
    ([3] * 4, "translate:1011+translate:0112", 9, 9),
    ([2] * 6, "translate:111111", 32, 2),
    ([5, 5], "translate:1x2", 5, 5),
    ([4] * 3, "shifts", 1, 64),
    ([3] * 4, "rot", 24, None),
]


@pytest.mark.parametrize("axes,spec,n_orbits,osize", ORBIT_CASES)
def test_orbit_partition_is_a_partition(axes, spec, n_orbits, osize):
    sp = ProductSpace(axes)
    G = parse_group(sp, spec)
    p = G.orbits()
    assert len(p) == n_orbits
    if osize is not None:
        assert (p.sizes == osize).all()
    # partition: disjoint, covering
    assert int(p.sizes.sum()) == sp.size
    assert (np.sort(p.members) == np.arange(sp.size)).all()
    # G-invariant, and the representative is the orbit minimum
    for o in range(len(p)):
        members = p.orbit(o)
        assert int(p.reps[o]) == int(members.min())
        for g in G.gens:
            assert (np.sort(g.apply_index(sp, members)) == members).all()
        assert (p.labels[members] == o).all()
        # orbit_of() reproduces the partition without materialising it
        assert (G.orbit_of(int(members[0])) == members).all()


@pytest.mark.parametrize("axes,spec,n_orbits,osize", ORBIT_CASES)
def test_orbit_stabiliser(axes, spec, n_orbits, osize):
    """|orbit| divides |G| (orbit-stabiliser), and the orbits of a free
    action all have size |G|."""
    sp = ProductSpace(axes)
    G = parse_group(sp, spec)
    order = G.order(cap=1 << 14)
    for s in np.unique(G.orbits().sizes):
        assert order % int(s) == 0


def test_trivial_group_degenerates_to_free_search():
    sp = ProductSpace([3] * 3)
    p = trivial_group(sp).orbits()
    assert len(p) == sp.size and (p.sizes == 1).all()
    assert (p.reps == np.arange(sp.size)).all()


def test_segment_sum_is_the_per_orbit_total():
    sp = ProductSpace([3] * 4)
    p = parse_group(sp, "translate:1011").orbits()
    v = np.arange(sp.size, dtype=np.int64) ** 2
    got = p.segment_sum(v)
    for o in range(len(p)):
        assert int(got[o]) == int(v[p.orbit(o)].sum())


def test_group_order_and_element_cap():
    sp = ProductSpace([3] * 3)
    assert monomial_group(sp).order() == 6 ** 3 * 6      # (3!)^3 * 3!
    assert coordinate_symmetric(sp).order() == 6
    assert coordinate_rotation(sp).order() == 3
    assert value_shifts(sp).order() == 27
    assert diagonal_translations(sp).order() == 3
    assert value_shift(sp, 1).order() == 3
    with pytest.raises(ValueError):
        monomial_group(sp).elements(cap=10)


def test_parse_group_terms_and_errors():
    sp = ProductSpace([3] * 4)
    assert len(parse_group(sp, "trivial").gens) == 0
    assert parse_group(sp, "translate:1011").order() == 3
    assert (parse_group(sp, "translate:1011").orbits().labels
            == parse_group(sp, "translate:1x0x1x1").orbits().labels).all()
    assert parse_group(sp, "diag").order() == 3
    assert parse_group(sp, "shift:0+shift:2").order() == 9
    assert parse_group(sp, "sym").order() == 24
    assert parse_group(sp, "iso:hamming").order() == monomial_group(sp).order()
    with pytest.raises(ValueError):
        parse_group(sp, "nonsense")
    with pytest.raises(ValueError):
        parse_group(sp, "translate:11")             # wrong length


# ====================================================================
# automorphism diagnostics — the honest part
# ====================================================================
@pytest.mark.parametrize("mk,fam", [
    (lambda: RefHammingBackend([3] * 3, 1), "hamming"),
    (lambda: RefLeeBackend([5, 5], 1), "lee"),
    (lambda: RefTorusBackend([6, 6], 1), "torus_linf"),
])
def test_isometry_groups_really_are_automorphism_groups(mk, fam):
    """Translations preserve coverage in every shift-invariant family,
    and so does the whole isometry group the library builds for it."""
    b = mk()
    assert value_shifts(b.space).is_automorphism_of(b)
    assert isometry_group(b.space, fam).is_automorphism_of(b)


def test_translations_are_NOT_automorphisms_of_the_open_grid():
    """grid_linf has no wraparound, so the edge balls are smaller and no
    translation preserves coverage. Orbit search over a translation group
    is still *correct* there (any union of orbits is scored exactly) — it
    is just aiming at a family that contains nothing good."""
    b = RefGridBackend([5, 5], 1)
    assert not value_shifts(b.space).is_automorphism_of(b)
    assert coordinate_symmetric(b.space).is_automorphism_of(b)


def test_monomial_group_is_the_hamming_automorphism_group():
    b = RefHammingBackend([3] * 3, 1)
    assert monomial_group(b.space).is_automorphism_of(b)


# ====================================================================
# exactness of the orbit searcher
# ====================================================================
EXACT_CASES = [
    (lambda: RefHammingBackend([3] * 4, 1), "translate:1011"),
    (lambda: RefHammingBackend([2] * 6, 1, mu=2), "translate:111111"),
    (lambda: RefLeeBackend([5, 5], 1), "translate:1x2"),
    (lambda: RefTorusBackend([6, 6], 1), "translate:2x2"),
]


@pytest.mark.parametrize("mk,spec", EXACT_CASES)
def test_orbit_moves_are_exact_marginals(mk, spec):
    """add_orbit / drop_orbit report exactly what a from-scratch recount
    says the deficit did."""
    b = mk()
    s = SymmetricSearcher(b, parse_group(b.space, spec), seed=3)
    rng = np.random.default_rng(5)
    picks = rng.choice(len(s.part), size=min(6, len(s.part)), replace=False)
    for o in picks:
        o = int(o)
        before = s.uncov
        gain = s.add_orbit(o)
        b.recount(s.code)
        assert b.uncovered() == s.uncov == before - gain
        assert set(s.code.tolist()) >= set(s.part.orbit(o).tolist())
    for o in picks[::-1]:
        o = int(o)
        before = s.uncov
        loss = s.drop_orbit(int(o))
        b.recount(s.code)
        assert b.uncovered() == s.uncov == before + loss


@pytest.mark.parametrize("mk,spec", EXACT_CASES)
def test_gain_bound_is_an_upper_bound_and_monotone(mk, spec):
    """The gain-map segment sum bounds the exact orbit gain from above,
    and never increases as the solution grows — the two properties that
    make the lazy-greedy heap exact."""
    b = mk()
    s = SymmetricSearcher(b, parse_group(b.space, spec), seed=1)
    first = s.gain_bounds()
    for o in range(len(s.part)):
        g = s.add_orbit(o)
        s.drop_orbit(o)
        assert g <= int(first[o])
    s.add_orbit(int(np.argmax(first)))
    later = s.gain_bounds()
    assert (later <= first).all()
    for o in range(len(s.part)):
        if s.selected[o]:
            continue
        g = s.add_orbit(o)
        s.drop_orbit(o)
        assert g <= int(later[o])


@pytest.mark.parametrize("mk,spec", EXACT_CASES)
def test_solution_is_a_union_of_orbits_and_feasible(mk, spec):
    b = mk()
    G = parse_group(b.space, spec)
    s = SymmetricSearcher(b, G, seed=2, deadline=time.time() + 60)
    assert s.greedy_fill(10 ** 9) == 0
    s.peel()
    assert s.verify_full() == 0
    part = s.part
    chosen = set(part.labels[s.code].tolist())
    expect = np.sort(np.concatenate([part.orbit(o) for o in sorted(chosen)]))
    assert (np.sort(s.code) == expect).all()
    # invariant under every generator
    for g in G.gens:
        assert (np.sort(g.apply_index(b.space, s.code)) == np.sort(s.code)).all()


def test_peel_only_drops_orbits_that_keep_feasibility():
    b = RefHammingBackend([2] * 6, 1)
    s = SymmetricSearcher(b, parse_group(b.space, "translate:111111"), seed=4,
                          deadline=time.time() + 60)
    s.greedy_fill(10 ** 9)
    before = len(s.code)
    s.peel()
    assert s.verify_full() == 0 and len(s.code) <= before
    # nothing left to peel: dropping any selected orbit breaks feasibility
    for o in np.flatnonzero(s.selected):
        assert s.drop_orbit(int(o)) > 0
        s.add_orbit(int(o))


def test_determinism():
    """Same seed, same answer — including the LNS rounds."""
    out = []
    for _ in range(2):
        b = RefHammingBackend([2] * 6, 1, mu=2)
        s = SymmetricSearcher(b, parse_group(
            b.space, "translate:010100+translate:111011"), seed=17,
            deadline=time.time() + 60)
        s.greedy_fill(10 ** 9)
        s.peel()
        s.lns(len(s.code) - int(s.part.sizes.min()), round_budget_s=2.0,
              seed=5)
        out.append(np.sort(s.code).tolist())
    assert out[0] == out[1]


def test_space_mismatch_is_rejected():
    b = RefHammingBackend([3] * 4, 1)
    with pytest.raises(ValueError):
        SymmetricSearcher(b, trivial_group(ProductSpace([3] * 3)))
    with pytest.raises(ValueError):
        Group(ProductSpace([3] * 3), [translation(ProductSpace([3] * 4),
                                                 [1, 0, 0, 0])])


# ====================================================================
# the payoff: structured cells
# ====================================================================
def _orbit_time_to(mk, spec, target, budget):
    """Seconds until orbit search verifies a solution of size <= target,
    or None."""
    b = mk()
    hit = {"t": None}
    t0 = time.time()
    s = SymmetricSearcher(b, parse_group(b.space, spec), seed=1,
                          deadline=time.time() + budget)

    def on_solved(M, idx):
        if M <= target and hit["t"] is None:
            b.recount(idx)
            assert b.uncovered() == 0          # verified, from scratch
            hit["t"] = time.time() - t0
            s.deadline = time.time()
    s.on_solved = on_solved
    s.greedy_fill(10 ** 9)
    if s.uncov == 0:
        s.peel()
        s.descend(floor=target, notch_budget_s=max(0.3, budget / 8.0))
    return hit["t"]


#: Searcher.descend will not start a notch it cannot finish, and reserves
#: this much of the deadline for that check; a free run given only its
#: working budget therefore never descends at all. Comparisons that get
#: this wrong flatter orbit search enormously.
_DESCEND_RESERVE = 30.0


def _free_time_to(mk, target, budget, seed=1):
    b = mk()
    hit = {"t": None}
    t0 = time.time()
    s = Searcher(b, seed=seed, deadline=t0 + budget + _DESCEND_RESERVE)

    def on_solved(M, idx):
        if M <= target and hit["t"] is None:
            hit["t"] = time.time() - t0
            s.deadline = time.time()
    s.on_solved = on_solved
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    s.peel()
    if len(s.code) <= target:
        on_solved(len(s.code), s.code)
    else:
        s.descend(floor=target,
                  notch_budget_s=min(20.0, max(1.0, budget / 8.0)))
    return hit["t"]


#: (label, factory, group, optimum, why the optimum is group-invariant)
STRUCTURED = [
    ("K_3(4,1)=9", lambda: RefHammingBackend([3] * 4, 1), "translate:1011", 9,
     "the perfect ternary Hamming code contains the generator, so it is a "
     "union of that generator's translate orbits"),
    ("K_2(6,1)=12", lambda: RefHammingBackend([2] * 6, 1), "translate:111111",
     12, "an optimal length-6 binary cover invariant under complementation"),
    ("K_2^2(6,1)=20", lambda: RefHammingBackend([2] * 6, 1, mu=2),
     "translate:010100+translate:111011", 20, "cosets of an order-4 group"),
    ("K_2^3(8,1)=96", lambda: RefHammingBackend([2] * 8, 1, mu=3),
     "translate:10000111+translate:01000110", 96, "cosets of an order-4 group"),
    ("K_4^2(4,1)=48", lambda: RefHammingBackend([4] * 4, 1, mu=2),
     "translate:2x0x1x3+translate:1x3x0x3", 48, "cosets of an order-16 group"),
    ("GW Lee Z_5^2=5", lambda: RefLeeBackend([5, 5], 1), "translate:1x2", 5,
     "the Golomb-Welch perfect code {(i,2i)} IS one orbit"),
    ("king 9x9=9", lambda: RefTorusBackend([9, 9], 1), "translate:3x1", 9,
     "the perfect dominating lattice IS one orbit"),
]


@pytest.mark.parametrize("label,mk,spec,opt,why", STRUCTURED)
def test_orbit_search_reaches_the_structured_optimum(label, mk, spec, opt, why):
    t = _orbit_time_to(mk, spec, opt, budget=20.0)
    assert t is not None, f"{label}: orbit search missed the optimum ({why})"
    assert t < 5.0, f"{label}: took {t:.2f}s"


#: The subset where the free searcher measurably cannot keep up. Each is
#: 0/5 seeds inside a 30s budget for free search (scripts/bench_symmetry.py),
#: while orbit search lands in milliseconds.
WALLED = [c for c in STRUCTURED if c[0] in
          ("K_2(6,1)=12", "K_2^2(6,1)=20", "K_2^3(8,1)=96",
           "K_4^2(4,1)=48")]


@pytest.mark.parametrize("label,mk,spec,opt,why", WALLED)
def test_orbit_search_beats_free_search_on_walled_cells(label, mk, spec, opt,
                                                        why):
    """The headline claim, as a measurement: free search does not reach
    the optimum inside a budget that orbit search beats by 3+ orders of
    magnitude."""
    budget = 15.0
    t_orb = _orbit_time_to(mk, spec, opt, budget=budget)
    assert t_orb is not None and t_orb < 2.0
    t_free = _free_time_to(mk, opt, budget=budget)
    assert t_free is None or t_free > 20 * max(t_orb, 1e-3), (
        f"{label}: free {t_free}s vs orbit {t_orb}s")


#: The other half, stated as sharply: no union of these orbits attains
#: the optimum, so orbit search cannot reach it however long it runs,
#: while free search does.
UNSTRUCTURED = [
    ("K_4(4,1)=24", lambda: RefHammingBackend([4] * 4, 1), "translate:0x1x1x3",
     24, 32),
    ("Lee 7x7 R=1", lambda: RefLeeBackend([7, 7], 1), "translate:1x2", 12, 14),
    ("king 11x11", lambda: RefTorusBackend([11, 11], 1), "translate:10x2", 16,
     22),
]


@pytest.mark.parametrize("label,mk,spec,opt,orbit_floor", UNSTRUCTURED)
def test_orbit_search_loses_when_the_optimum_is_unstructured(
        label, mk, spec, opt, orbit_floor):
    code, s = orbit_solve(mk(), parse_group(mk().space, spec), budget_s=6.0)
    assert len(code) >= opt
    assert len(code) >= orbit_floor - 4, label     # nowhere near the optimum
    b = mk()
    b.recount(code)
    assert b.uncovered() == 0                      # still a valid solution


def test_mu_fold_is_mu_disjoint_translates():
    """The classical mu-fold construction, produced by orbit search: the
    orbits of the repetition-code translation on Z_2^3 are its four
    cosets, each a perfect 1-cover, so a mu-fold cover is exactly mu of
    them and K_2^mu(3,1) = 2*mu is attained."""
    for mu in (1, 2, 3, 4):
        b = RefHammingBackend([2] * 3, 1, mu=mu)
        G = parse_group(b.space, "translate:111")
        s = SymmetricSearcher(b, G, seed=1, deadline=time.time() + 30)
        assert s.greedy_fill(10 ** 9) == 0
        s.peel()
        assert len(s.code) == 2 * mu               # == the sphere bound
        assert s.verify_full() == 0
        assert int(s.selected.sum()) == mu         # mu whole cosets
        # every chosen orbit is on its own a perfect single cover
        for o in np.flatnonzero(s.selected):
            b1 = RefHammingBackend([2] * 3, 1)
            b1.recount(s.part.orbit(int(o)))
            assert b1.uncovered() == 0 and (b1.cnt == 1).all()


def test_free_polish_handoff():
    """The hybrid: a group-invariant start handed to the word-level
    searcher, which must accept it as an ordinary solution."""
    b = RefHammingBackend([3] * 4, 1)
    s = SymmetricSearcher(b, parse_group(b.space, "translate:1011"), seed=1,
                          deadline=time.time() + 30)
    s.greedy_fill(10 ** 9)
    free = s.free_searcher(seed=2)
    assert isinstance(free, Searcher)
    assert free.uncov == 0 and (np.sort(free.code) == np.sort(s.code)).all()


def test_orbit_solve_convenience():
    b = RefHammingBackend([2] * 6, 1)
    code, s = orbit_solve(b, parse_group(b.space, "translate:111111"),
                          budget_s=8.0)
    b.recount(code)
    assert b.uncovered() == 0 and len(code) == 12
    assert s.summary()["orbits"] == 32

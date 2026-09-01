#!/usr/bin/env python3
"""Example 4 — searching over ORBITS instead of words.

    python examples/04_symmetry.py

What this shows
---------------
Every space coldcore searches has a big automorphism group, and the
ordinary search core ignores it: it treats the |X| words as |X|
unrelated candidates.  The symmetry layer changes the *atoms* of the
search — a solution becomes a union of orbits of a group you choose —
which is the classical group-invariant-code construction, mechanised.

Three cells, three lessons.

1. **K_2(6,1) = 12.**  An optimal length-6 binary covering code that is
   invariant under complementation `x -> x + 111111`.  Over the orbits of
   that order-2 group the greedy walks straight to 12 in milliseconds;
   the free word-level searcher gets there too, but takes three orders
   of magnitude longer and only on some seeds.  We then *prove* 12 is
   optimal with the isomorph-free exhaustive module — the two halves of
   a settled cell from the same layer.

2. **K_2^3(3,1) = 6.**  The mu-fold construction in its purest form: the
   orbits of the repetition-code translation on Z_2^3 are its four
   cosets, and each coset is on its own a perfect 1-cover.  A mu-fold
   cover is therefore exactly mu of them, and orbit search finds it in
   one greedy pass.

3. **The 11x11 king torus.**  The honest negative.  The group has order
   11, so every orbit has 11 points and *no* union of orbits has the 16
   points the optimum needs.  Orbit search cannot reach it however long
   it runs; free search gets there in milliseconds.  The group has to
   divide the answer.

A note on fair timing, since it is easy to get wrong: `Searcher.descend`
reserves 30s of its deadline before starting a notch (it will not begin
one it cannot finish), so a free run must be given its working budget
PLUS that reserve or it silently never descends at all.  The orbit
descent has no such reserve.

Nothing here is GPU-specific: the orbit machinery is pure numpy and
drives the same `coldcore.protocol.Backend` surface the CUDA plugins
implement.
"""
import sys
import time

import numpy as np

try:
    from coldcore.reference import RefHammingBackend, RefTorusBackend
    from coldcore.search import Searcher
    from coldcore.symmetry import SymmetricSearcher, parse_group
    from coldcore.symmetry_exact import IsomorphFreeSearch
except ImportError:                                    # running from a checkout
    import os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from coldcore.reference import RefHammingBackend, RefTorusBackend
    from coldcore.search import Searcher
    from coldcore.symmetry import SymmetricSearcher, parse_group
    from coldcore.symmetry_exact import IsomorphFreeSearch

BUDGET = 20.0


def orbit_solve(backend, spec, floor=1, budget=BUDGET):
    """Greedy over orbits -> orbit peel -> orbit notch descent."""
    G = parse_group(backend.space, spec)
    s = SymmetricSearcher(backend, G, seed=1,
                          deadline=time.time() + budget)
    t0 = time.time()
    s.greedy_fill(10 ** 9)
    assert s.uncov == 0, "no union of these orbits covers the space"
    s.peel()
    code = s.descend(floor=floor, notch_budget_s=budget / 8.0)
    return code, s, time.time() - t0


def free_solve(backend, floor=1, budget=BUDGET, seed=1):
    # Searcher.descend reserves 30s of the deadline before starting a
    # notch, so give it that much headroom on top of the working budget
    s = Searcher(backend, seed=seed, deadline=time.time() + budget + 35)
    t0 = time.time()
    s.load(np.zeros(0, dtype=np.int64))
    s.greedy_fill(10 ** 9)
    s.peel()
    if len(s.code) > floor:
        s.descend(floor=floor, notch_budget_s=max(2.0, budget / 8.0))
    assert s.verify_full() == 0
    return s.code, time.time() - t0


# ------------------------------------------------------------------ 1
print("1) K_2(6,1): free search vs orbit search vs an exhaustive proof")
b = RefHammingBackend([2] * 6, 1)
code, s, t_orb = orbit_solve(b, "translate:111111", floor=12)
b.recount(code)
assert b.uncovered() == 0 and len(code) == 12
print(f"   orbits of <111111>: {len(s.part)} orbits of size "
      f"{int(s.part.sizes[0])} (|X| = {b.space.size})")
print(f"   orbit search  -> M = {len(code)} in {t_orb:.3f}s  (verified)")

free_code, t_free = free_solve(RefHammingBackend([2] * 6, 1), floor=12)
print(f"   free search   -> M = {len(free_code)} in {t_free:.1f}s"
      f"  ({t_free / max(t_orb, 1e-4):.0f}x longer)")
assert len(free_code) >= 12 and t_free > 20 * t_orb

ex = IsomorphFreeSearch(RefHammingBackend([2] * 6, 1), verify=False)
verdict, _w, st = ex.decide(11, time_limit=300)
print(f"   exhaustive    -> M = 11 is {verdict} "
      f"({st['nodes']} nodes, {st['secs']}s)  => K_2(6,1) = 12, settled")
assert verdict == "infeasible"

# ------------------------------------------------------------------ 2
print("\n2) K_2^mu(3,1) = 2*mu: mu disjoint translates, found as mu orbits")
for mu in (1, 2, 3, 4):
    bm = RefHammingBackend([2] * 3, 1, mu=mu)
    sm = SymmetricSearcher(bm, parse_group(bm.space, "translate:111"), seed=1,
                           deadline=time.time() + 10)
    assert sm.greedy_fill(10 ** 9) == 0
    sm.peel()
    assert len(sm.code) == 2 * mu and int(sm.selected.sum()) == mu
    assert sm.verify_full() == 0
    # each chosen orbit is on its own a perfect single cover
    for o in np.flatnonzero(sm.selected):
        b1 = RefHammingBackend([2] * 3, 1)
        b1.recount(sm.part.orbit(int(o)))
        assert b1.uncovered() == 0 and (b1.cnt == 1).all()
    print(f"   mu = {mu}: M = {len(sm.code)} = {mu} whole cosets "
          f"(= the sphere bound 2*mu)")

# ------------------------------------------------------------------ 3
print("\n3) 11x11 king torus: the case where orbit search LOSES")
bt = RefTorusBackend([11, 11], 1)
code, s, t_orb = orbit_solve(bt, "translate:10x2", floor=1)
bt.recount(code)
assert bt.uncovered() == 0
free_code, t_free = free_solve(RefTorusBackend([11, 11], 1), floor=1)
print(f"   orbits of <(10,2)>: {len(s.part)} orbits of size "
      f"{int(s.part.sizes[0])} -- 16 is not a multiple of 11")
print(f"   orbit search  -> M = {len(code)} in {t_orb:.1f}s (a valid cover)")
print(f"   free search   -> M = {len(free_code)} in {t_free:.1f}s")
assert len(code) > len(free_code), "expected orbit search to lose here"

print("\nOK — orbit search wins where the optimum is group-invariant, "
      "loses where it is not.")

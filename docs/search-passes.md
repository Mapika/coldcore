# Search passes

The search core (`src/coldcore/search.py`) is the "pass library" of the
system: a set of problem-agnostic transformations of a candidate
solution, each written against the [backend protocol](../src/coldcore/protocol.py)
alone. They compose the way compiler passes do — a driver
(`Searcher.descend`) picks an order and a budget, and every pass leaves
the backend in an exact, re-checkable state.

Nothing here approximates coverage. Every accept/reject decision is
taken on an exact field query (`ball_gather`, `count_eq`), and the
incremental bookkeeping is re-checked against the exact field
periodically inside every long-running pass.

## The library

| pass | what it does | cost per step |
|---|---|---|
| `greedy_fill` | exact lazy greedy build/repair to a target size | 1 transform per refresh + `ball_gather` per candidate batch |
| `ruin` | remove k incumbents (`low` / `hole` / `cluster` / `random` / `deficit`) | 0–1 transforms (+ a `cnt` scan for `deficit`) |
| `peel` | remove zero-loss incumbents in independent batches, to fixpoint | 1 transform per batch |
| `lns` | ruin-and-recreate rounds at fixed M | per round: a ruin + a greedy repair |
| `anneal` | swap-neighbourhood polish at fixed M that accepts regressions | 2 `ball_gather`s per move (`pool`) or 2 per move with one batched over the whole candidate set (`shell`), 1 transform per pool refresh |
| `two_swap` | exact bounded (add, drop) search for the endgame | 1 transform + 1 batched `ball_gather` per node |
| `portfolio` | time-sliced LNS streams around a shared incumbent | per slice: one `lns` |
| `break_wall` | the endgame driver: focused-ruin / polish / two-swap bursts at one fixed M | — |
| `descend` | the notch ladder driving all of the above | — |

`greedy_fill`, `ruin`, `peel`, `lns` and `descend` are the four passes
that produced the covering-code records; their default behaviour is
frozen. The rest of this page documents the additions and how to switch
them on. Everything below is opt-in: with no new arguments, every pass
is bit-identical to the version that set the records.

## Adaptive ruin selection — `lns(..., select="adaptive")`

`lns` draws a ruin operator every round. The default (`select="fixed"`)
draws from the hand-tuned mix the record runs used — `hole` twice as
often as `low`, `cluster`, `random`.

`select="adaptive"` replaces that with an `OperatorBandit`: each
operator carries a score that is an exponential moving average of the
reward its rounds earned, and the next operator is drawn proportionally
to that score, mixed with a uniform exploration `floor` so an unlucky
operator is never starved. Rewards are tiered:

| outcome of the round | reward |
|---|---|
| new global best | 3.0 |
| better than the round's start | 1.0 |
| sideways | 0.3 |
| worse | 0.0 |

The scores *start* at the proven fixed mix (normalised to mean 1), so
the selector begins where the records were set and only moves away from
it when the instance says otherwise. After every round the per-operator
record — uses, mean reward, current selection probability — is parked in
`Searcher.stats["ops"]`, so a run is auditable after the fact.

Why it can matter: which ruin operator works is instance-dependent, and
a fixed mix pays for the wrong operator on every cell where it is wrong,
for the whole run. Even when it changes nothing, the recorded per-operator
rewards are the cheapest evidence available for whether the fixed mix is
right for a given family.

Why it is not the default: the scores are a plain EMA of reward, so on a
cell where every operator earns the same reward — a notch where nothing
works, for instance — the selector converges to *uniform*, discarding the
2x weight the record runs put on `hole`. `select="fixed"` stays the
default so the proven behaviour is what you get without asking.

## Focused ruin at the deficit — `ruin(k, "deficit")`

The other four ruin operators are blind to *where* the problem is.
`low` picks by loss, `cluster` picks around a random incumbent, `random`
is random, and `hole` — the closest thing to targeted — picks a random
point out of the gain field and ruins the incumbents nearest **it**.
That point is anywhere in the ball around the deficit, so at R = 5 the
aim is off by up to 5 digits in an arbitrary direction, and the extract
it is drawn from is capped at 4096 entries out of a neighbourhood that
can hold millions.

`deficit` aims at the uncovered words themselves:

```
words = uncovered_words()                  # exact indices, cnt < mu
d[c]  = min over words of digit-distance(c, word)   # for every incumbent c
ruin the k incumbents with the smallest d
```

which is the structure the wall actually has: when the count is 1–3 the
blocking is local, and the incumbents that must move are the ones whose
balls sit just short of the hole.

The `deficit_pool` factor in that sketch is not decoration: the obvious
version — take the strict k nearest — **measurably lost** to the
untargeted mix on every cell tested. At a wall the deficient words are
the same ones round after round, so the strict rule ruins the same k
incumbents every round, the greedy repair rebuilds the same solution,
and the round is a no-op. A ruin operator needs variety as much as aim,
so the k victims are drawn from the `deficit_pool * k` (default 3k)
nearest instead.

Read [Measured on the GPU](#measured-on-the-gpu) before switching this
on. Even fixed, the operator does **not** pay as a drop-in addition to
the LNS mix (6/15 against the untargeted mix's 9/15) — which is why it
is off by default in `lns` and `descend`. It pays inside `break_wall`
(14/15 against 12/15 without it, and faster), where the polish burst
supplies the variety it costs.

`uncovered_words()` reads the exact `cnt` field (`read_cnt`, an optional
backend capability that both shipped backends have) in chunks from a
rotating offset, stopping as soon as it has `limit` (default 4) of them.
Two properties make that affordable:

* the scan cost scales with the *deficit density*, not the space. Early
  in a notch, with thousands of words uncovered, four are found in the
  first few megabytes. It is only at the wall — one uncovered word in
  10^10 — that a sweep is a full-field read;
* and exactly there, the result is cached against the solution that
  produced it. On a wall, round after round is rejected straight back to
  the same incumbent, so the scan runs once per *distinct* incumbent
  rather than once per round.

`scan_cap` bounds the read per call for a caller that would rather have
an untargeted ruin than a long scan; an empty result always means "not
found within the cap", never "nothing is uncovered". On a backend
without `read_cnt` the operator degrades to `hole` rather than failing.

Switching it on inside LNS:

```python
searcher.lns(M, budget, deficit=True)                    # fixed mix
searcher.lns(M, budget, deficit=True, select="adaptive")  # bandit
```

With `select="fixed"` the operator joins the draw list weighted like
`hole`; with `select="adaptive"` the bandit runs over `RUIN_MODES_EXT`
with `deficit` given the `hole` prior. Either way the selector also gets
a **hint**: while `uncov <= deficit_hint` (default 3) — the wall regime —
`deficit` is boosted (fixed: drawn outright half the time; adaptive: its
score multiplied by `deficit_boost`, default 4, with a floor so a
punished operator can still come back). Outside that regime the
selectors are untouched. `RUIN_MODES` and `RUIN_MIX` are frozen at the
proven four, so the portfolio's stream seeding and the default bandit
are unchanged.

## Polish at fixed M — `anneal(M, budget_s, mode="lahc"|"anneal")`

`lns` walks essentially downhill in uncovered count: a round that ends
worse is rejected 90% of the time, and each round costs a ruin plus a
full greedy repair. That makes it expensive per step on the plateaus the
notch ladder runs into — the familiar `best uncov = 1..3` wall — where
what is needed is a lot of near-neutral steps rather than a few large
ones.

`anneal` walks the plateau instead. One move is a **swap** — drop one
incumbent `w`, add one candidate `v` — costed exactly and incrementally:

```
uncov' = uncov + ball_gather(w, 1) - ball_gather(v, 0)
```

the first term counting the points only `w` covered (which become
uncovered), the second the points `v` newly covers. Both are exact
backend queries, so no move is accepted on an approximation; a rejected
move is undone by the inverse single-word `ball_update`, which returns
`cnt` to precisely its prior state. `M` never changes.

Candidates come from a pool refreshed from the gain map every
`pool_refresh` accepted moves — a point has positive gain exactly when
it covers something currently uncovered, so the pool is precisely the
set of useful additions (and is automatically disjoint from the
incumbent). Each move exact-checks `sample` random incumbents for the
drop and `sample` pool draws for the add. Refresh time is also where the
incremental count is re-checked against `count_eq(0)`.

Two acceptance rules:

- `mode="lahc"` — **late acceptance hill climbing** (Burke & Bykov):
  accept when the new cost beats the cost `lahc_len` moves ago, or the
  current one. No temperature schedule to tune, and it is fully
  deterministic given a seed. This is the default.
- `mode="anneal"` — **Metropolis** with geometric cooling from `t0` to
  `t_end` across the budget. The schedule is driven by wall clock (so it
  adapts to unknown throughput), which means it is *not* bit-reproducible
  across machines; use `lahc` when reproducibility matters.

What it buys: on the reference backend at K_3(5,1), M = 26 (one below
the optimum, so an unreachable notch), LNS manages ~410 ruin/recreate
rounds per second while the polish manages ~4900 swap moves per second —
an order of magnitude more plateau explored for the same budget, and
every one of those moves exact. It does not make an infeasible M
feasible; it changes how much of the plateau a fixed budget covers.
`tests/test_search_passes.py` pins a Z_5^2 state that is a *proven*
strict local optimum of the swap neighbourhood (brute-forced in the
test) which both acceptance rules escape.

That paragraph is true and it is also the wrong thing to have measured.
"Moves per second" says nothing about whether any of those moves is ever
*accepted*, and on the cells this framework actually hunts records on,
the answer for the default neighbourhood turned out to be no. Read the
next section before trusting the number above.

### The neighbourhood is the whole story — `neighbourhood="pool"|"shell"|"auto"`

The cost model above is right, the acceptance rules are right, and on a
walled cell the default pass still does **nothing**. Measured on the
live wall cells (diagnostic logs from the record campaign):

| cell | M | uncov at the wall | incumbent loss (median) | best cost from the stock pool | best cost from the local shell |
|---|---|---|---|---|---|
| K_6(10,5) | 483 | 6 | 32 | +8 … +33 | **6** (plateau) |
| K_8(8,5) | 78 | 62 | 201 | +68 … +142 | **61–62** |
| K_8(10,6) | 323 | 20 | 53 | +23 … +32 | **19–20** |

and the consequence: a 243,000-move `anneal` run on K_8(8,5) at M = 78
accepted **zero** moves.

The reason is the candidate set, not the arithmetic. `_gain_pool()` is
`map_extract(thr=1)` — on that cell, 16.3 million points with a median
gain of 5 — capped at 16384, drawn from uniformly. Meanwhile the
incumbent being dropped privately covers ~200 words. Every sampled swap
therefore costs +195-ish and LAHC rejects all of them, forever. The
blind candidates are simply *nowhere near the hole the drop just made*.

`neighbourhood="shell"` keeps the exact cost, the exact undo and the
acceptance rule, and changes only where candidates come from:

* **the drop** is one of the `w_topk` (24) cheapest incumbents by exact
  loss, read from a cached loss map refreshed every `loss_refresh` (48)
  accepts — with probability `w_random_p` (0.25) a uniform incumbent
  instead, so the pass cannot lock onto one corner;
* **the adds** are the words that can actually plug the hole `w` left:
  `w`'s full digit-distance-1 shell, `n_shell2` (128) sampled
  distance-2 words, `n_top` (48) draws from the **top** of the gain map
  (the same histogram-guided high threshold `greedy_fill` uses, not
  `thr=1`) and `n_rand` (8) uniform draws;
* the whole batch — typically 1000–1500 candidates — is costed in **one**
  `ball_gather`, and the acceptance rule is applied to all of it, so a
  plateau move is taken whenever one *exists* rather than whenever a
  handful of blind draws happen to contain one. `greedy_p` (0.5) of the
  time the cheapest accepted candidate is taken, otherwise a random
  accepted one;
* `w` itself is excluded from the batch — re-adding it is a null move of
  cost exactly `uncov`, which LAHC would accept forever;
* after `restart_stale` (15000) moves without a new best the pass falls
  back to the best state it has seen and restarts the acceptance history
  there.

Measured effect on those cells: 11–19% accepts, drift 0, and K_8(8,5) at
M = 78 walks from 62 uncovered — where LNS had stalled for good — to 13
in about seven minutes. The reason the local shell always finds a
plateau move is visible in the diagnostic: for the local set,
`best_gain == loss(w)`, i.e. something near `w` recovers exactly what
`w` was covering. The cost of a plateau move is 0 by construction, and
the pass gets to spend its whole budget choosing *which* plateau move.

`neighbourhood="auto"` spends the budget in `pool` slices of
`auto_probe` (0.15) and hands the rest to `shell` the first time a
slice's accept rate falls under `auto_accept_rate` (0.01). Slices
rather than a single probe, because the pool neighbourhood is not
uniformly bad — it is bad *on the plateau*, and a probe taken during the
easy descent into the plateau reports that all is well. `break_wall`
uses `auto`; `stats["polish_neighbourhood"]` records what it settled on.

`"pool"` stays the default of `anneal` itself, so the frozen path is
untouched.

### Pool cadence — `pool_refresh=64 | "auto"`

The refresh costs one full gain transform, and how that compares to a
move is entirely instance-dependent: on a small GPU cell a transform is
about the cost of a single swap move (both are launch-latency bound), on
a 10^10-word cell it is seconds while a move stays sub-millisecond. A
fixed cadence of 64 is therefore wrong in both directions — far too lazy
on the small cell (the pool goes stale for no saving) and far too eager
on the large one.

`pool_refresh="auto"` measures both and solves for the cadence:

```
P  >=  t_transform * (1 - share) / (share * t_move)
```

with `share = pool_share` (default 0.10), `t_transform` timed at each
refresh and `t_move` timed as wall-time-per-accepted-move over the
interval between refreshes, both smoothed by a 2-point EMA and clamped
to [1, 65536]. The cadence in force is reported in
`stats["pool_refresh"]`.

It is **not** the default, and deliberately so: it reads the clock, and
the documented seed-reproducibility of `mode="lahc"` depends on the
cadence being a constant. `break_wall` uses `"auto"`; a run that has to
be bit-reproducible should pass an integer.

## Exact endgame — `two_swap(M, budget_s)`

The wall is defined by "no single swap improves": every drop that frees
a slot costs more than any single addition pays back. `anneal` gets off
such a plateau by *accepting* the loss and wandering; `two_swap` instead
enumerates the pair neighbourhood and costs every pair **exactly**.
That is affordable only because at uncovered <= 2 the useful additions
are not arbitrary points: a point helps iff its ball meets the deficit,
and the gain field names exactly those points, so one transform and one
`map_extract` produce the candidate set rather than a random sample of
the space.

The pass is a depth-2 DFS over M-preserving `(add x, drop c)` steps:

* additions come from that extract, sorted by `(gain desc, index asc)`
  — `map_extract` is unordered and its GPU slot order is not
  reproducible, so the sort is what makes the pass deterministic;
* at the **last** level only additions that close the entire remaining
  deficit are considered (`thr = uncov`), because a drop can never lower
  the count — so the last step is "add something that finishes, drop
  something with exactly zero loss";
* drops are ranked by their exact loss **after** the addition, from one
  batched `ball_gather(code, 1)` — the cost of a pair is never
  estimated;
* a step is expanded only if it leaves the count within `slack`
  (default 1) of where the level started: worsening by one is the
  escape, worsening by five is a different search;
* every step is undone by the inverse `ball_update`, so a failed search
  leaves `cnt` bit-identical to the state it was handed.

Exponential in depth in principle; `branch` (6), `drop_branch` (6),
`nodes` (2000) and `budget_s` bound it to a few thousand exact pair
evaluations in practice. Those bounds are where the honesty line sits:
every *cost* in the search is exact (nothing is estimated, and a failed
search restores `cnt` bit-for-bit), but the *enumeration* is truncated —
`branch` additions out of a candidate set itself capped at `cand_cap`,
and on a large cell `map_extract` returns an arbitrary `cand_cap` of the
`found` matches. So a negative result from this pass means "no pair was
found within the budget", never "no pair exists". It refuses to run outside `0 < uncov <=
max_uncov` (default 2), because outside that range the deficit
neighbourhood is not small and this is the wrong tool.

The pinned Z_5² Lee state that `test_search_passes.py` brute-forces as a
strict swap-local optimum is broken by `two_swap` at depth 2 in **two
nodes**, and provably not at depth 1 — the test asserts both directions.

## The wall breaker — `break_wall(M_target, budget_s)`

The recipe the campaign hand-rolls, as one call. Seed at the incumbent,
drop to `M_target`, then rotate bursts, because each pass is blind to
something different:

1. **focused-ruin LNS** (`deficit=True`, hinted while the count is low)
   — moves the incumbents that actually block the deficit;
2. **LAHC polish** at the same M over the local neighbourhood
   (`polish_neighbourhood`, default `"auto"`; `pool_refresh="auto"`) —
   this is the workhorse and takes the larger share (`polish_share`,
   default 0.6) of every burst;
3. **exact two-swap** once the count is at `two_swap_at` (default 2) —
   the only pass here that can *prove* no pair closes the gap rather
   than sample it.

Between bursts the best state is kept; after `stagnation` (default 3)
bursts without improvement the state is restored to the best and kicked
with a `restart_frac`·M deficit ruin, so a burst chain never wanders off
the wall for good. M never changes, so the only outcomes are 0 or an
honest count. The per-burst trajectory lands in `stats["wall"]`:

```python
s.load(read_solution("best_483.txt", n=b.space.n))
s.break_wall(482, budget_s=6 * 3600)
s.stats["wall"]   # {'M': 482, 'bursts': 41, 'best': 1, 'traj': [...], ...}
```

## Restart portfolio — `portfolio(M, budget_s, streams=N)`

A single LNS run at a hard notch either gets there or burns the whole
budget in one basin. `portfolio` keeps `streams` distinct solutions —
the current one plus perturbations of it, each seeded by a different
ruin operator — and gives each a `slice_s` slice of `lns` per round,
round-robin. The best solution seen is the shared incumbent; after every
round the **worst** stream is discarded and re-seeded from a fresh
perturbation of the incumbent (`ruin_frac`·M points kicked out, then a
greedy refill). That re-seeding is what makes this a portfolio rather
than `streams` unrelated runs. It stops the moment any stream reaches
uncovered = 0.

Cost note: switching streams costs one full recount transform, so the
default `slice_s` gives each stream roughly three slices per call. On a
cell where a recount is a large fraction of a slice, raise `slice_s` (or
the notch budget) rather than the stream count.

## Driving them from `descend`

`descend` keeps its proven defaults; the notch attempt is what the new
knobs configure:

```python
searcher.descend(floor=..., notch_budget_s=300.0,
                 select="adaptive",     # bandit ruin selection
                 portfolio=4,           # 4 time-sliced LNS streams
                 polish="lahc",         # polish a walled-out notch
                 polish_share=0.5)      # ...with half the notch budget
```

The polish runs *inside* the same notch budget, not on top of it: the
LNS (or portfolio) gets `1 - polish_share` of it, and the polish gets
the rest, starting from whatever wall the repair hit. With
`polish=None`, `portfolio=1`, `select="fixed"` — the defaults — the
attempt is bit-identical to the original single `lns` call.

Every pass takes an explicit `seed`, which runs that pass (and the
sub-passes it calls) on its own random stream; `seed=None` keeps the
searcher's stream, so default runs are unchanged.

## Descent schedule (v0.3)

`descend` takes six schedule keywords. All are off by default in the
library, which keeps the v0.2 schedule; `coldcore solve` turns them on
(`--legacy-descent` restores v0.2).

| keyword | v0.2 | `coldcore solve` default | effect |
|---|---|---|---|
| `salvage` | off | on | a failed notch's best state (size `M - step`, `u` words uncovered) is completed by the exact greedy and peeled; a result smaller than `M` is accepted |
| `carry` | off | on | the attempt after a failed notch starts from that notch's best state, refilled by the exact greedy to the new target, instead of from the last solution; at most once in a row (after a carried attempt fails too, the next restarts from the last solution) |
| `patience` | off | 2 | the LNS of a notch stops when the best uncovered count has not decreased for `patience x notch_budget_s` seconds (`True` = 1; hard cap `4 x notch_budget_s`); the step doubles after a success faster than `notch_budget_s` |
| `persist` | off | on | with a deadline, the descent does not stop after `fail_limit` failures at step 1; it keeps attempting notches until the deadline |
| `reserve_s`, `min_notch_s` | 30 s, 20 s | 3 s, 5 s | time kept unused at the end, and the smallest notch budget the descent starts |

The CLI also lowers the inner-loop reserves (`Searcher.loop_reserve_s`,
`build_reserve_s`) from 10 s and 5 s to 2 s and 1 s, and sets the notch
budget to a tenth of the run budget, clamped to 10..300 s, when
`--notch-budget` is not given (v0.2: 300 s).

Two measurements on DEV cells. On `K_4(8,2)` (600 s, `--start greedy`,
15 runs at once so each run is slower than alone, seeds 4-8) the v0.2
schedule never improved on the greedy code (448 in all five runs), and the
v0.3 schedule reached 427-444. The reason for `salvage`, measured on
`K_3(8,1)` with v0.2.2 and a 600 s budget: after a few fast notches the step doubled to 16, a single 266 s
notch at `M = 618` ended with 9 words uncovered, and the run stopped at
634 with 30 s unused. With `salvage` that notch alone returns at most
`618 + 9` words.

## Repair rule ablation — `Searcher(..., repair="random")`

`repair="exact"` (default) is the exact maximum-gain insertion described
above. `repair="random"` replaces every insertion made by `greedy_fill`
(the initial build, every LNS repair, portfolio restarts, the salvage):
take a deficient word `u` (from a scan of `cnt` that starts at a random
offset), and insert a candidate drawn uniformly from all candidates
whose ball contains `u`. Ruin operators, acceptance, descent schedule
and polish are unchanged; the exact gain of the inserted word is read
only to keep the uncovered count current. CLI: `--repair random`.

It exists only for ablations that isolate the effect of the exact gain
fields on the repair. It needs a product-space backend that exposes its
problem and radius (the GPU, axis-DP and reference backends); the
candidates covering `u` are `u` plus the support patterns of the ball,
so the draw is uniform over the ball. The coset start
(`docs/cosets.md`) is not affected by it; combine with `--start greedy`
to remove the exact gain fields from the construction as well.

## From the CLI

```
python -m coldcore solve --backend gpu -p q=5 -p n=7 -p R=2 --hours 6 \
    --ruin-select adaptive --portfolio 4 --polish lahc --polish-share 0.5 \
    --deficit
```

`--ruin-select fixed --portfolio 1 --polish none` (and no `--deficit`)
are the defaults and reproduce the record-setting behaviour exactly.

To work a single wall instead of descending:

```
python -m coldcore solve --backend gpu -p q=6,n=10,R=5 --hours 6 \
    --seed-file best_483.txt --wall 482 --out wall482.txt
```

`--wall M` skips the descent entirely and runs `break_wall` at that one
size, with `--wall-budget` / `--wall-burst` for the timings. It writes
`--out` only if the wall actually falls.

## Measured on the GPU

Everything above was designed against the CPU reference backend and the
diagnostics from the live wall campaign. This section is what the passes
actually do on the GH200, measured by
[`scripts/bench_wall.py`](../scripts/bench_wall.py).

**Protocol.** For each cell: greedy + peel + a short *default* descent
establishes an incumbent; the size attacked is the hardest size that
descent reached, started from the solution one notch above it — so the
target is known-feasible (the setup produced a witness) and known-hard
(it was the last notch the setup budget could manage). Every config then
gets the same start bytes, the same 5 seeds and the same wall-clock
budget at that fixed M, and the final count is re-verified by a
from-scratch recount. Configs are run seeds-outer / configs-inner so
every config sees the same machine conditions.

**Configs.** `lns` = the default notch attempt (one fixed-mix LNS).
`deficit` = the same plus the deficit-targeted ruin operator. `polish` =
half the budget on LNS then LAHC over the stock pool neighbourhood at
the fixed cadence of 64 — the documented best before this round.
`polish_auto` = the same with `pool_refresh="auto"`. `polish_shell` =
the same with `neighbourhood="shell"`. `wall` = `break_wall`;
`wall_nots` = `break_wall` with the exact two-swap endgame switched off.

### K_5(7,3), |X| = 78125 — incumbent M = 114, attacking M = 113, 50s x 5 seeds

| config | solved | mean uncov | median t_solve | rounds/s | moves/s | accept rate |
|---|---|---|---|---|---|---|
| `lns` | 3/5 | 0.6 | 0.7 | 1.0 | — | — |
| `deficit` | 2/5 | 0.6 | 4.0 | 0.7 | — | — |
| `polish` | 3/5 | 0.4 | 0.7 | 0.9 | 7.8 | 2.5% |
| `polish_auto` | 4/5 | 0.2 | 16.2 | 0.9 | 7.1 | 2.4% |
| `polish_shell` | 4/5 | 0.2 | 16.3 | 0.9 | 5.6 | 6.8% |
| `wall_nots` | 5/5 | 0.0 | 11.6 | 0.2 | 19.8 | 11.4% |
| `wall` | **5/5** | 0.0 | 14.2 | 0.2 | 22.0 | 14.1% |

### K_6(6,3), |X| = 46656 — incumbent M = 54, attacking M = 53, 50s x 5 seeds

| config | solved | mean uncov | median t_solve | rounds/s | moves/s | accept rate |
|---|---|---|---|---|---|---|
| `lns` | 3/5 | 0.4 | 33.6 | 0.3 | — | — |
| `deficit` | 0/5 | 1.0 | — | 0.3 | — | — |
| `polish` | 1/5 | 0.8 | 5.4 | 0.2 | 15.7 | 1.7% |
| `polish_auto` | 5/5 | 0.0 | 27.6 | 0.3 | 3.3 | 2.9% |
| `polish_shell` | 5/5 | 0.0 | 28.5 | 0.3 | 3.7 | 14.0% |
| `wall_nots` | 5/5 | 0.0 | 5.9 | 0.2 | 17.2 | 10.7% |
| `wall` | **5/5** | 0.0 | 11.4 | 0.2 | 22.2 | 12.0% |

### K_3(6,1), |X| = 729 — incumbent M = 88, attacking M = 84, 40s x 5 seeds

| config | solved | mean uncov | median t_solve | rounds/s | moves/s | accept rate |
|---|---|---|---|---|---|---|
| `lns` | 4/5 | 0.2 | 12.0 | 9.6 | — | — |
| `deficit` | 2/5 | 0.6 | 17.9 | 10.3 | — | — |
| `polish` | 1/5 | 0.8 | 19.7 | 6.2 | 17.7 | 5.7% |
| `polish_auto` | 3/5 | 0.4 | 18.1 | 7.4 | 10.4 | 4.6% |
| `polish_shell` | 3/5 | 0.4 | 15.3 | 7.5 | 10.0 | 9.8% |
| `wall_nots` | 4/5 | 0.2 | 14.1 | 4.2 | 21.7 | 15.9% |
| `wall` | **5/5** | 0.0 | 13.4 | 3.6 | 17.8 | 17.1% |

### What the numbers say

Totals over the three cells, 15 (cell, seed) attempts each:

| config | solved |
|---|---|
| `polish` (the previous documented best) | 5/15 |
| `lns` (the frozen default) | 10/15 |
| `polish_auto` | 12/15 |
| `polish_shell` | 12/15 |
| `wall_nots` | 14/15 |
| `wall` | **15/15** |

1. **The wall driver is the result.** `break_wall` broke every one of the
   fifteen attempts; the frozen default broke ten. Nothing else in this
   round is worth as much, and none of its individual passes is worth
   this much alone — the rotation is doing the work.

2. **The polish's *candidate set* is what decides everything, and both
   ways of fixing it work.** `polish` differs from `polish_auto` only in
   the pool refresh cadence, and goes from 5/15 to 12/15. Note it does so
   while running *fewer* moves per second, not more: on K_6(6,3) the
   fixed cadence manages 15.7 moves/s at a 1.7% accept rate, the measured
   cadence 3.3 moves/s at 2.9% — it pays four transforms' worth of time
   for a pool that is actually current, and wins. `polish_shell` reaches
   the same 12/15 by the other route, replacing the global pool with the
   dropped word's local shell, and it is the one that lifts the accept
   rate by 3-5x (14.0% vs 2.9% on K_6(6,3)). These small cells cannot
   separate the two; the live wall cells can, and there the pool
   neighbourhood accepts *nothing at all* (see the section above), which
   no refresh cadence can fix. That is why `break_wall` uses `"auto"`:
   it gets the cheap fix where it suffices and the real one where it
   does not.

3. **Focused deficit ruin loses on its own and wins inside the
   rotation.** As an LNS operator it is a negative result twice over —
   see the two ablations below. As a component of `break_wall` it is a
   positive one. Both are real, and the second is the one that decides
   the default.

4. **The exact two-swap endgame is, at this scale, marginal.** `wall`
   beat `wall_nots` by one attempt out of fifteen (15/15 vs 14/15) and
   the pass fired on only a minority of runs (24-81 nodes explored where
   it did). It is cheap, it is exact, and it is the only pass that can
   report "no pair of swaps closes this" — but on cells this size the
   claim that it breaks walls is not supported. It is kept on by default
   in `break_wall` because its cost is bounded and its one extra solve
   is the difference between 14/15 and 15/15, not because these numbers
   prove it.

### Two ablations on the deficit operator

The first version of the operator took the **strict k nearest**
incumbents to the deficient words. Against the untargeted mix, over the
same fifteen attempts:

| config | solved |
|---|---|
| `lns` | 10/15 |
| `deficit` (strict k nearest) | 4/15 |

It lost on all three cells. The mechanism is visible once stated: at a
wall the deficient words are the same ones round after round, so "the k
nearest to them" is the same set every round, the greedy repair rebuilds
the same solution, and the round is a no-op. A ruin operator needs
variety as much as aim. The operator now draws its k victims from the
`deficit_pool * k` (default 3k) nearest. Re-measured, same protocol,
fresh `lns` control under the same machine conditions:

| config | K_5(7,3) | K_6(6,3) | K_3(6,1) | total |
|---|---|---|---|---|
| `lns` | 4/5 | 3/5 | 2/5 | 9/15 |
| `deficit` (sampled from 3k nearest) | 3/5 | 0/5 | 3/5 | 6/15 |

Better (4/15 → 6/15) and still not good enough: **as a drop-in addition
to the LNS operator mix, the deficit operator does not pay.** That is
why it is off by default in `lns` and `descend`, and it is the honest
answer to "does aiming the ruin at the hole help?" — on its own, no.

Inside `break_wall` the answer flips. Same protocol, `break_wall` with
and without the operator:

| config | K_5(7,3) | K_6(6,3) | K_3(6,1) | total | median t_solve |
|---|---|---|---|---|---|
| `break_wall` (`deficit=True`, the default) | 5/5 | 5/5 | 4/5 | **14/15** | 14.8 / 7.6 / 18.1 s |
| `break_wall(deficit=False)` | 3/5 | 5/5 | 4/5 | 12/15 | 14.5 / 20.8 / 26.9 s |

It wins on solves and wins clearly on time-to-solve on two of three
cells. The reading that fits both results: the operator supplies aim and
destroys variety, and a plain LNS has nothing to replace the variety
with — but in the driver the polish burst *is* the diversification, so
the operator's aim is all that is left to contribute. It stays on by
default in `break_wall` and off by default everywhere else.

### Caveats on these numbers

* **The GPU was shared.** The record campaign held the device at 100%
  utilisation throughout, so the absolute throughput figures (moves/s,
  rounds/s, transforms/s) are contention-depressed by roughly an order
  of magnitude and mean nothing on their own. The comparisons are still
  fair — configs were interleaved by seed — and solve counts, mean
  uncovered and accept rates are unaffected by contention.
* **These are small cells.** |X| between 729 and 78125, against the
  10^8-10^10 of the cells that produce records. What transfers is the
  *mechanism* (which candidate set gets accepted); what does not is the
  cost balance between a transform and a move, which is exactly why
  `pool_refresh="auto"` measures it rather than assuming it.
* **Five seeds per cell.** Enough to separate 15/15 from 5/15, not
  enough to separate 12/15 from 14/15. Run-to-run noise is real and
  visible: the `lns` control scored 10/15 in the main table and 9/15 in
  the re-measurement, and `break_wall` 15/15 and 14/15, from identical
  configurations — the passes read the wall clock, so nothing here is
  bit-reproducible across runs. Treat single-attempt differences as
  noise and multi-attempt ones as signal.
* `rounds/s` is low on every config because an LNS round's cost here is
  dominated by `greedy_fill`'s Python-side candidate heap rebuild at
  each refresh, not by GPU work. That is pre-existing and equally
  charged to every config; it is also a standing invitation to profile
  `greedy_fill`.

## Testing

`tests/test_search_passes.py` and `tests/test_search_wall.py` run every
added pass on the reference (brute-force, CPU) backends. Each must (a) reach the known optimum on
the standard smokes — K_2(4,1) = 4, gamma(C9 × C9 king) = 9, the
Golomb–Welch perfect Lee code in Z_5² at R = 1 — and (b) never report a
cover that is not one: every "solved" claim is re-checked by unioning
raw balls, without consulting the backend's incremental `cnt` at all, so
a bookkeeping bug cannot make an invalid cover look valid. The default
paths are pinned against the new ones for bit-identical behaviour.

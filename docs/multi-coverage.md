# mu-fold multiple covering — `K^mu(n; ax; R)`

A **mu-fold covering code** is a code in which every word of the space
is within distance `R` of at least `mu` codewords. `K_q^mu(n,R)` denotes
the minimum size of such a code over `Z_q^n`; `mu = 1` is the ordinary
covering-code problem `K_q(n,R)`.

Multiple coverings are a record surface of their own — though, unlike
the classical case, no maintained online table exists (Keri's site
carries only `mu = 1`). The binary state of the art lives in papers:
the Hamalainen–Honkala–Kaikkonen–Litsyn tables (DCC 1993), Ostergard's
tabu-search improvements (Australas. J. Combin. 12, 1995, with explicit
codes), Seuranen's consolidated tables (PhD thesis, Aalto 2011, Table
2.3), and Krotov–Potapov's lower bounds (IEEE Trans. IT 67, 2021). That
lineage uses exactly coldcore's convention — a **set** of distinct
codewords, at-least-mu coverage, closed balls — so our bounds are
directly comparable cell for cell. The distinct multiset variant (MCR)
and the MCF/"farthest-off points" line are *different* quantities;
never compare against those columns. The classical
constructions are unions of shifted perfect codes, which are optimal
only in the divisible cases — the rest is search space.

In coldcore, `mu` is **not a plugin**. It is a change of *predicate* on
the `cnt` plane, which is why it works for every problem family at once
(hamming, lee, torus_linf, grid_linf, and anything added later).

## The objective and its two exact marginals

Write `cnt(x)` for the number of codewords covering `x` and

```
deficit   D(S) = sum_x max(0, mu - cnt(x))          (D = 0  <=>  feasible)
```

`D` is the total number of (word, fold) requirements still unmet, and it
is the quantity the search core carries as `uncov`. Its two marginals
are exact and both are ball sums of an indicator field, so both are one
transform:

```
gain(x) = D(S) - D(S + x) = #{ w in B_R(x) : cnt(w) <  mu }
loss(c) = D(S - c) - D(S) = #{ w in B_R(c) : 1 <= cnt(w) <= mu }
```

*Gain*: adding `x` raises `cnt` by one on every word of `B_R(x)`, which
retires exactly one unit of deficit at each word that still had some —
i.e. at each word with `cnt < mu`. (Note this is a count of *words*, not
of folds: a word two folds short contributes 1, because only one of its
two missing folds is filled by this one codeword.)

*Loss*: removing a codeword `c` lowers `cnt` by one on every word of
`B_R(c)`, which creates one new unit of deficit at each word that was at
level `mu` or below. Every word of `B_R(c)` has `cnt >= 1` (c covers
it), so the clause `1 <= cnt` excludes nothing that a removal can touch;
it is there so that the field is the legacy `[cnt==1]` indicator
verbatim when `mu = 1`.

Both indicators degenerate to the ordinary ones at `mu = 1`:

| quantity | general `mu` | `mu = 1` |
|---|---|---|
| objective | `sum_x max(0, mu-cnt(x))` | `#{x : cnt(x) == 0}` |
| gain input | `[cnt < mu]` | `[cnt == 0]` |
| loss input | `[1 <= cnt <= mu]` | `[cnt == 1]` |
| peel: `c` removable | no `w in B(c)` with `cnt(w) <= mu` | no privately covered `w` |

so the `mu = 1` code paths are untouched — literally: the plugin
dispatches to the legacy kernels when `mu == 1`.

Submodularity survives: `D` is a sum over words of a concave function of
`cnt`, so the coverage gain is still submodular and the lazy greedy is
still exact. The peel and the batched-removal independence argument are
unchanged too (removals whose balls are disjoint cannot interact).

## Using it

Python — every backend takes `mu`:

```python
from coldcore.gpu import GpuBackend
from coldcore.reference import RefHammingBackend
from coldcore.search import Searcher

b = GpuBackend([2, 2, 2], 1, problem="hamming", mu=2)   # K_2^2(3,1)
s = Searcher(b, seed=1)
s.load([])
s.greedy_fill(10**6)          # b.uncovered() == 0  =>  a 2-fold cover
s.descend(floor=1)
```

`RefHammingBackend(axes, R, mu=2)` (and the `lee` / `torus` / `grid`
references) is the independent brute-force oracle for the same instance.

Backend surface, unchanged in shape:

- `uncovered()` returns the **total deficit** `D` (not a word count) —
  at `mu = 1` these coincide, which is why the search core needs no
  change: it decrements `uncov` by the exact placement gain.
- `ball_gather(idx, target)` takes a coverage **role**: `0` = deficient
  (`cnt < mu`, the exact placement gain), `1` = critical
  (`1 <= cnt <= mu`, the exact removal loss). At `mu = 1` the roles are
  the literal `cnt` levels the ABI always used.
- `count_eq(t)` stays a literal `cnt == t` query at every `mu`.
- `gain_map()` / `loss_map()` write the transformed role indicators.

## Plugin ABI (optional v1 extension)

Feature-tested by symbol presence, exactly like `dct_map_hist`; a plugin
without them supports `mu = 1` only, and `GpuBackend(..., mu>1)` then
raises instead of computing something wrong.

```
int  dct_set_mu(int mu);        /* multiplicity for all later calls */
int  dct_get_mu(void);
int  dct_ball_gather_mu(const long long *words, int nwords, int role,
                        int32_t *out);        /* role 0/1 as above   */
long long dct_count_deficit(void);            /* sum max(0, mu-cnt)  */
```

`dct_set_mu` also re-points `dct_transform`'s two indicator init modes
(`1` -> `[cnt<mu]`, `2` -> `[1<=cnt<=mu]`) and the owner-trick loss pass.
The singleton context resets `mu` to 1 on `dct_free`, and the host
binding re-states `mu` after every `dct_init3`, so multiplicity can
never leak into a later cell.

Kernel-side, the extension is `src/plugins/covering/dctcore_mu.cuh`:
predicate twins of the three `cnt`-reading core kernels
(`k_init_from_cnt`, `k_ball_gather`, `k_loss_scan`) plus a deficit
reduction. `dctcore_core.cuh` stays byte-identical to upstream.

## Verification

- `tests/test_multi.py` — CPU: role/field identities and the exact
  marginal property (`D` after an add/remove equals `D -/+ gain/loss`,
  checked against from-scratch recounts) for hamming, lee and torus
  cells at `mu = 2, 3`; peel safety; and three exhaustively determined
  optima **K_2^1(3,1) = 2, K_2^2(3,1) = 4, K_2^3(3,1) = 6** (exhaustive
  search over all subsets of the 8-word space; these attain the sphere
  bound `mu*|X|/|B_1| = 2*mu` and are the unions of the four disjoint
  perfect codes).
- `tests/test_gpu_parity_multi.py` — GPU vs. brute force on all three
  families (mixed radix, even Lee axes) at `mu = 2, 3`: `cnt`, deficit,
  both maps over the whole space, both gather roles; plus end-to-end
  runs reaching `K_2^2(3,1) = 4` and `K^2(6x6 king torus) = 8` (sphere
  bound `2*36/9`, attained by two shifted perfect 3x3 tilings), and the
  `dct_free` mu-reset check.

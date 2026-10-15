# Coset start (Hamming)

`src/coldcore/cosets.py` builds starting codes for homogeneous Hamming
cells `K_q(n,R)` that are unions of cosets of a subgroup `H` of the
alphabet group. It is the default first stage of `coldcore solve` on
such cells (with `--hours`); `--start greedy` turns it off.

## Construction

Let `A` be an abelian group of order `q`. Any bijection between the
alphabet `{0,...,q-1}` and `A` preserves Hamming distance, so the
choice only decides which subgroups are available. coldcore takes
`A = F_{p1}^{e1} x ... x F_{pk}^{ek}` for `q = p1^e1 ... pk^ek`
(digit `a` maps to its residues modulo each `p^e`, written in base `p`).
For a prime power `q` the subgroups of `A^n` then include the
`GF(q)`-linear codes.

A subgroup `H` of `A^n` is given as the kernel of a syndrome map
`s(x) = (P_p x_p)_p` with full-rank matrices `P_p` over `F_p`. The
quotient `Q = A^n / H` has `|Q| = prod p^(m_p)` elements, and every
coset has `|H| = q^n / |Q|` words. Let `D = { s(e) : wt(e) <= R }`.
Because translations by `H` are isometries, a code that is a union of
cosets with syndrome set `S` covers a word `x` exactly when
`s(x)` lies in `S + D`. Covering `A^n` with such a code is therefore the
covering problem "cover `Q` by translates of `D`" (the matrix method of
Carnielli and Östergård), on `|Q|` points instead of `q^n`.

`QuotientBackend` implements the ordinary backend protocol for that
problem: candidates and covered points are the elements of `Q`,
candidate `s` covers `s + D`. Its gain and loss fields are cyclic
correlations with the indicator of `D`, computed by a character
transform of `Q` (Walsh-Hadamard transform on the radix-2 digits, real
FFT on the others) and rounded; the values are integers bounded by
`|D|`, and the floating-point error of the transform is many orders of
magnitude below 1/2 at the sizes used (`|Q| <= 2^21`). `ball_update`
and `ball_gather` are index operations on `s + D`. The search core
(greedy, peel, LNS, descent) runs on it unchanged; its greedy takes each
placement directly from a fresh gain field (`Searcher._field_fill`,
selected by the backend attribute `field_greedy`), since one transform
costs less than scoring a batch of candidates at `|D|` reads each.

`D` is computed from the packed ball patterns (one syndrome per word of
weight at most `R`) when the ball is small, and otherwise by a dynamic
program over coordinates and weight on `Q`.

## Search over subgroups

`CosetSearch` keeps state across calls and spends each call in two
phases.

1. **Screening** (half of the call). Round-robin over the subgroup
   orders `|H|` for which a code of the size of the greedy code would be
   a union of 2 to 400 cosets and `|Q| <= 2^21`, cheapest quotient
   first. Each visit draws four random syndrome maps of that order,
   keeps the one with the largest `|D|`, runs the exact greedy on `Q`,
   expands the cosets to words on the word-level backend, peels, and
   records the size.
2. **Search** (the rest). Quotient notch descent (greedy, peel, LNS with
   the v0.3 schedule) on the screened subgroups whose screening size is
   within 5% of the best size so far, best first, each for
   `max(3 s, 5 x its screening time)`. If no screened subgroup
   qualifies, the call returns early.

Every candidate code is loaded on the word-level backend, peeled, and
verified by an exact recount there; a subgroup can only produce a large
code, never an invalid one.

## Use in `coldcore solve`

With `--start auto` (default) and `--hours`, on homogeneous Hamming
cells with `mu = 1` and without `--seed-file` or `--wall`:

1. exact greedy build and peel;
2. coset search for `--coset-share` (default 0.2) of the time left;
3. if the best coset code is smaller than the greedy code, the run
   alternates: a descent from the current best code that stops at its
   first failure at step 1, then further coset search in slices of
   `--coset-slice` (default 0.15) of the budget, and a new descent from
   every coset code that is smaller than the best code;
4. otherwise the descent runs from the greedy code until the deadline.

The alternation exists because a union of cosets is often a local
optimum for word-level ruin-and-recreate: on `K_3(8,1)` and `K_8(8,4)`
the descent from a coset code removed no word in 600 s, while further
coset search still found smaller coset codes on other cells.

## Measurements

`coldcore solve --backend gpu --hours 0.1667` (600 s), defaults, seeds
1, 2, 3, one run at a time on a GH200, against v0.2.2 with the same
command. Best verified size per seed; `table` is the upper bound in
Kéri's tables (2011).

| cell | table | v0.2.2 | v0.3.0 |
|---|---|---|---|
| K_3(8,1) | 486 | 627, 634, 627 | 486, 486, 486 |
| K_4(8,2) | 352 | 420, 412, 420 | 417, 400, 402 |
| K_5(8,3) | 325 | 396, 396, 396 | 350, 370, 350 |
| K_6(8,3) | 1080 | 983, 981, 981 | 950, 955, 972 |
| K_8(8,4) | 512 | 566, 566, 566 | 512, 512, 512 |
| K_8(9,4) | 2944 | 2758, 2758, 2758 | 2304, 2423, 2486 |
| K_10(9,5) | 1088 | 1055, 1055, 1055 | 1000, 1055, 1000 |

On `K_4(8,2)` no union of cosets of a random subgroup was smaller than
the greedy code (512 and above, against 448); there the stage returns
after screening and the difference comes from the descent schedule. On
`K_3(8,1)` and `K_8(8,4)` the final codes are coset codes. On
`K_5(8,3)`, `K_6(8,3)` and `K_8(9,4)` the descent from a coset code
removed between 1 and 137 words in some runs (`K_6(8,3)`: 972 to 950
and 981 to 955; `K_8(9,4)`: 2560 to 2423), and later coset slices
replaced the code in others. Random subgroups did not reach
the table value 325 on `K_5(8,3)`: with order 25 or 125 they give 375
in most trials, and 350 in some.

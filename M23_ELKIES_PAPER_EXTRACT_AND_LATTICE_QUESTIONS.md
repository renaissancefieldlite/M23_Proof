# M23 Elkies Paper Extract And Lattice Questions

Date: 2026-04-03

## Purpose

This file extracts the operationally useful content from:

- [m23docs.pdf](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/m23docs.pdf)

It is not a full transcription of the paper.
It is the working extract for the M23 repo.

Use it when we need to answer:

- what Elkies actually proved
- what structure the paper gives us
- what the repo should really be checking
- what questions would most help collapse the next rung of answers

## Paper Identity

The PDF is the Elkies paper:

- Noam D. Elkies
- `The complex polynomials P(x) with Gal(P(x) - t) ~= M23`

The local PDF has 8 pages.

## What The Paper Actually Does

The paper solves the complex classification problem for degree-23 polynomials
whose Galois group over `C(t)` is `M23`.

The important point for this repo is:

- the paper is not just about writing down one polynomial
- it is also about proving the Galois group is `M23` and not `A23`

That proof matters because the raw nonlinear equations used to compute the
polynomial do not themselves distinguish `M23` from `A23`.

## High-Value Extract

### 1. Scope Of The Classification

Elkies studies degree-23 complex polynomials `P` with:

- `Gal(P(x) - t) ~= M23`

Up to linear equivalence, the paper says there are:

- four equivalence classes of `M23` polynomials

This matters because the search lane should not casually talk as if there is
just one isolated object.

### 2. Branch Data

The paper says the map `P : P1 -> P1` is branched above only three points, with
branch orders:

- `23`
- `2`
- `4`

The corresponding cycle structures used in the paper are:

- for the order-2 branch element: `1^7 2^8`
- for the order-4 branch element: `1^3 2^2 4^4`

This is not cosmetic. This is structural data that constrains the shape of the
polynomial and the proof.

### 3. Structural Factorization Form

The paper writes the polynomial in the form:

- `P = P2^2 * P3 * P4^4 = P7 * P8^2 + tau`

with:

- `P2, P3, P4, P7, P8` pairwise coprime
- `deg(Pi) = i`

This is one of the core reasons the repo stores and audits:

- `P2`
- `P3`
- `P4`
- `tau`

### 4. Derivative Identity

The paper uses a key derivative trick:

- `dP/dx` is a multiple of `P2 * P4^3 * P8`
- therefore `P8` can be recovered from the derivative identity

Operationally, this matters because it reduces the effective search dimension
and is part of why the original construction is rigid.

### 5. Field Of Definition

The final solution is over the quartic field:

- `F = Q[g] / (g^4 + g^3 + 9g^2 - 10g + 8)`

The paper also notes this quartic field contains `Q(sqrt(-23))`.

This exactly matches the repo’s Elkies anchor.

### 6. Search Mod 29, Then p-adic Lift

The paper did not jump directly to the final field expression.
The rough workflow was:

- search over a finite field
- succeed mod `29`
- p-adically lift the solution
- use lattice reduction / recognition to recover the algebraic coefficients in
  the quartic field

This matters because it tells us the original solve strategy was:

- modular search
- p-adic lift
- algebraic recognition

not:

- brute-force symbolic closure alone

### 7. Why Irreducibility Alone Is Not The Proof

This is the most important extraction for the repo.

The paper explicitly says the equations used to find the polynomial do not
separate:

- `M23`
- `A23`

Elkies then says he tested many factorizations of `P(x) - t0 mod lambda` for
many `t0`, and checked:

- factor degrees
- cycle structures
- observed frequencies of those cycle structures

He compared those observations against:

- the 12 cycle structures that occur in `M23`
- the expected frequencies from `M23`

Then he used:

- Chebotarev
- Weil bounds

to show the distribution is incompatible with `A23`.

So the paper’s real proof logic is:

- factorization-pattern distribution
- cycle-structure matching
- non-`A23` exclusion

not:

- irreducible / not irreducible count by itself

### 8. The Large Prime In The Proof

For the proof stage, Elkies chooses a degree-1 prime over:

- `l = 10^8 + 7`

and reduces the polynomial modulo that prime.

Then he counts the factorization patterns over many `t0`.

This is the paper’s strong exclusion step against `A23`.

### 9. The 5-Transitivity Contradiction

The decisive contradiction is not just “the counts looked right.”

The paper derives point counts on auxiliary curves `Ck`, especially `C5`, and
uses Weil bounds to show the observed counts cannot come from `A23`, because
`A23` would force 5-transitivity behavior that the data contradicts.

This is a deep signal for us:

- the paper proves `M23` by rejecting `A23` through distributional structure

not by a simple scalar score.

## What This Means For The Repo

### 1. The Elkies Core Audit Is Real

The repo’s quartic field, `P2`, `P3`, `P4`, and `tau` are genuinely anchored in
the paper.

### 2. The Current Exact Verifier Is Too Weak To Mirror The Paper

The current exact verifier mostly reports:

- tested prime count
- irreducible count
- consistency score

That is much weaker than the paper’s proof logic.

### 3. The Right Next Exact Metric Is Cycle-Structure Aware

If we want the exact lane to become more paper-faithful, the verifier should
move toward:

- factor-degree signature collection
- cycle-structure matching against the `M23` table
- frequency comparison across many `t0` values
- explicit exclusion tests against `A23`

### 4. The `6/9` Threshold Is Not From The Paper

The repo’s `6/9` threshold is a local heuristic in the exact auto loop.

It is not, from what we have extracted, a paper-derived theorem threshold.

### 5. The Identity Specialization Hook Still Breaks The Exact Candidate Lane

Even with the right paper in hand, the repo cannot honestly treat the
lambda/mu exact-candidate lane as mature while:

- `apply_candidate_specialization(...)` is still identity

## Paper-Driven Questions The Repo Should Answer

These are the 8 highest-value questions I would ask to build the next lattice of
cohesive input.

### Question 1

```text
Given Elkies’ paper and the current M23_Proof repo, what is the smallest
paper-faithful verifier upgrade that would move us from raw irreducibility count
toward real M23-vs-A23 evidence? Be concrete about data structures, outputs,
and pass/fail criteria.
```

Why this matters:

- it targets the exact mismatch between the current verifier and the paper

### Question 2

```text
Extract the 12 M23 cycle structures used in Elkies’ proof and map each one to
the corresponding factor-degree pattern of P(x) - t0 mod p. Then explain how
the repo should store and score those patterns.
```

Why this matters:

- it converts the paper from theory into code-checkable signatures

### Question 3

```text
Is the current exact verifier’s prime list and sampling strategy mathematically
too weak to distinguish M23 from A23? If so, propose a practical sampling plan
that fits this repo and preserves the spirit of Elkies’ proof.
```

Why this matters:

- it tells us whether our current evidence lane is structurally underpowered

### Question 4

```text
Look at phase3_exact.py and determine whether the lambda/mu specialization lane
can be rehabilitated into a paper-consistent exact lane, or whether the repo
should formally retire it and focus on the descent lane plus a stronger exact
oracle.
```

Why this matters:

- it decides whether we are fixing a real lane or preserving a ghost

### Question 5

```text
What invariants from Elkies’ construction survive linear equivalence and should
therefore be tracked as canonical search invariants in the repo? Rank them by
importance for screening and comparison.
```

Why this matters:

- it tells us what the search should preserve instead of chasing raw coefficient
  shape alone

### Question 6

```text
Can we build a smaller-field or smaller-sample analogue of Elkies’ non-A23
argument that is still rigorous enough to be useful in this repo, or is the
large-field regime essential? Give a grounded answer, not a hopeful one.
```

Why this matters:

- it determines whether a practical proof-like verifier is feasible here

### Question 7

```text
Should the descent lane score candidates partly by closeness to the Elkies
factorization-signature distribution instead of only leakage, height, and
denominator pressure? If yes, specify exactly how.
```

Why this matters:

- it would tie the live search lane back to the real paper signal

### Question 8

```text
What is the most rigorous next measurable milestone for this repo: a restored
specialization rule, an M23 cycle-signature verifier, a non-A23 exclusion test,
or rationalized descent survivors that match the Elkies anchor more closely?
Rank them and justify the order.
```

Why this matters:

- it gives us an order of operations that is evidence-driven

## Working Extract

If someone asks for the shortest possible paper-based read, say this:

```text
Elkies computes degree-23 complex polynomials with Galois group M23, writes the
solution over a specific quartic field, and proves the group is M23 rather than
A23 by using factorization-pattern distributions mod a large prime together with
Chebotarev and Weil bounds. That means the repo’s exact lane should eventually
care more about cycle-structure signatures and M23-vs-A23 exclusion than about
raw irreducibility count alone.
```

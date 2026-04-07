# M23 Proof

`M23_Proof` is the working repository for a custom proof-search process aimed at constructing and verifying a degree-23 polynomial with Galois group `M23`, starting from the Elkies quartic-field construction.

The process in this repo is the method. It does four things:

- encode and audit the explicit Elkies quartic-field anchor
- search affine and Tschirnhaus transforms that may lead to a usable degree-23 polynomial lane
- screen survivors with exact leakage, height, denominator, modular signature, and cycle-structure checks
- build fixed-prime `t0` sampling and distribution tests that can push the verifier from weak `M23` hints toward real `M23` vs `A23` discrimination

There is no off-the-shelf model for this lane. The search process built here is part of the work: generate, screen, verify, refine, and keep state across runs until the lane either produces a real survivor or shows where it fails.

## Current Status

- the repo is grounded in a real Elkies construction
- the exact formulas for the quartic field anchor are real and audited
- the repo is not yet a completed proof or solved `M23` target
- the live descent lane is useful for narrowing transforms and promoting follow-up checks
- the old exact monitor lane is legacy / compatibility infrastructure, not the main proof surface

## State Anchors

- [M23_AUDIT_AND_STATUS_REPORT.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_AUDIT_AND_STATUS_REPORT.md)
- [M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md)
- [M23_ELKIES_PAPER_EXTRACT_AND_LATTICE_QUESTIONS.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_ELKIES_PAPER_EXTRACT_AND_LATTICE_QUESTIONS.md)
- [M23_NEXT_RUNGS_AND_LIVE_SEARCH.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_NEXT_RUNGS_AND_LIVE_SEARCH.md)
- [M23_RICK_HANDOFF_AND_BOOT.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_RICK_HANDOFF_AND_BOOT.md)

Canonical-path note:

- this Playground repo is the live M23 lane
- `/Volumes/Renaissance Hd/matrix/m23attack` is an older historical repo and
  should only be used for archaeology or comparison

## Elkies Anchor

Paper anchor:

- [m23docs.pdf](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/m23docs.pdf)
- [M23_ELKIES_PAPER_EXTRACT_AND_LATTICE_QUESTIONS.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_ELKIES_PAPER_EXTRACT_AND_LATTICE_QUESTIONS.md)

What the paper gives this repo:

- the polynomial is structurally a three-branch-point object with branch orders
  `23`, `2`, and `4`
- the explicit quartic field anchor and the `P2 / P3 / P4 / tau` formulas are
  real
- the proof logic distinguishes `M23` from `A23` using factorization-pattern
  distributions, cycle structures, Chebotarev, and Weil bounds

What that means here:

- raw irreducibility count is at best a weak proxy
- a stronger exact verifier should move toward factor-degree / cycle-signature
  checks
- the exact auto loop’s `6/9` threshold is a local heuristic, not a paper-based
  theorem target

## Search Surface

This repo now has a cleaner split between:

- the live descent search lane
- the legacy exact monitor lane

Use the live descent lane for current search work.
Use the legacy exact monitor only when you specifically need the old exact-loop
surface for archaeology or debugging.

## Run Surface

Live search, one batch:

```bash
./Run_Live_Search.command
```

Live search, continuous batches:

```bash
./Run_Live_Search_Forever.command
```

Live search, terminal status:

```bash
./Open_Live_Search_Status.command
```

Legacy exact monitor:

```bash
./Open_Legacy_Exact_Monitor.command
```

Legacy exact Qt-only monitor:

```bash
./Open_Legacy_Exact_Qt_Monitor.command
```

Why `Run_Live_Search.command` stops after one run:

- it is intentionally a bounded batch runner
- one batch gives one reproducible summary artifact
- parameter changes between batches stay explicit
- continuous motion belongs in `Run_Live_Search_Forever.command`

What `Run_Live_Search_Forever.command` now adds:

- it walks through a staged affine search schedule instead of replaying one fixed grid
- it expands the scale / shift window between batches
- it promotes the top transforms from each batch into modular follow-up
- it records both the search stage and the follow-up summary in `live_search_state.json`
- it can run with kill-gates so bad lanes die early instead of churning forever

The live descent lane writes into `testjson/` and `testjson/runtime/`:

- worker logs:
  `m23_descent_<id>.log`
- worker pid files while running:
  `m23_descent_<id>.pid`
- worker batch outputs:
  `descent_search_worker*_*.json`
- merged batch summaries:
  `descent_search_summary_*.json`
- descent follow-up summaries:
  `descent_followup_*.json`
- continuous-state file:
  `live_search_state.json`

The legacy exact lane still writes its own monitor/runtime artifacts under
`testjson/`, including `shared_best.json` and `m23_search_<id>.log` / pid files.

Workers now cover disjoint sections of the candidate list instead of duplicating the same pass. By default the exact scanner uses contiguous chunk partitioning so worker `1` covers the first slice, worker `2` the next slice, and so on. Set `M23_PARTITION_MODE=stride` if you want interleaved coverage instead.

The search now also has a reference-to-scan bridge:

- `verify_elkies_exact.py` records the per-prime signature of the explicit Elkies construction
- `verify_elkies_exact.py` now records full per-prime factor-degree signatures, not only irreducible counts
- `phase3_exact.py` now preserves those same factor-degree signatures in `exact_test_results_*.json`
- `m23_cycle_signatures.py` encodes the 12 M23 cycle structures from Elkies Table 1 and annotates factor-degree signatures with ATLAS labels, expected M23 mass, and k-subset counts
- `m23_cycle_verifier.py` turns Elkies exact results, exact scan results, or modular screen outputs into cycle-aware reports with an honest `not_ready` status for the full non-`A23` exclusion step
- `m23_fixed_prime_sampler.py` samples many `t0` values over one fixed degree-1 prime so the repo can start building the Table-2 / Weil-bound style distribution lane instead of only checking one factorization per prime
- `cross_reference_elkies_scan.py` now ranks scanned candidates by per-prime signature alignment against the Elkies reference, with the historical hot zone around `λ ≈ -13`, `μ ≈ -28` kept only as a secondary heuristic
- `descent_search.py` runs a partitionable affine descent search over the explicit Elkies construction and scores transforms by coefficient leakage in the basis `(1, g, g^2, g^3)`, coefficient height, and denominator pressure
- `run_parallel_descent_channels.py` launches disjoint descent workers so transform bands are covered in parallel instead of duplicated
- `Run_Live_Search.command` provides a one-shot launcher for the same partitioned descent lane
- `Run_Live_Search_Forever.command` repeats those batches until stopped
- `Open_Live_Search_Status.command` gives the live descent lane a terminal status surface instead of leaving it as raw log files only

This is the current lift-path scaffold. It does not yet prove the FrontierMath target, but it gives the scanner a principled way to prioritize branches that look closer to the known exact construction.

What the live search solves right now:

- it does not directly prove `M23`
- it searches for transforms of the Elkies anchor that are algebraically cleaner
- it reduces leakage, coefficient height, and denominator pressure before deeper verification
- it turns the search lane into a triage engine so modular and cycle-aware checks are spent on better transforms instead of noise

Example narrow debug run:

```bash
python3 run_parallel_descent_channels.py \
  --workers 1 \
  --pressure-cap 100000 \
  --height-abs-cap 1000000000000 \
  --dead-lane-limit 500
```

Example continuous live run with the same kill-gates:

```bash
M23_DESCENT_PRESSURE_CAP=100000 \
M23_DESCENT_HEIGHT_ABS_CAP=1000000000000 \
M23_DESCENT_DEAD_LANE_LIMIT=500 \
./Run_Live_Search_Forever.command
```

## Canonical M23 Hook

The repo is no longer treating the old `λ/μ` family as the canonical generator. The current pipeline is:

- `tschirnhaus_generate.py`
  bounded low-degree Tschirnhaus generator over the Elkies quartic field anchor
- `mod23_signature_screen.py`
  exact leakage / height / denominator screen plus modular factorization signatures at primes `p ≡ 1 (mod 23)`
- `lift_survivors.py`
  survivor promotion layer for the candidates worth deeper follow-up
- `m23_canonical_pipeline.py`
  one-shot runner for the generator/filter/cleanup split

The current first pass defaults to a quadratic Tschirnhaus family

```text
h(y) = y^2 + a*y + b
```

with `a, b ∈ F`, where

```text
F = Q[g] / (g^4 + g^3 + 9g^2 - 10g + 8)
```

and `a, b` are parameterized over the basis `(1, g, g^2, g^3)`.

This is the new lead search question:

- generator = bounded quadratic descent from Elkies
- filter = modular signatures at actual primes `p ≡ 1 (mod 23)`
- cleanup = survivor promotion for later lift / LLL / full Galois verification

The descent lane can be partitioned in several ways:

- `scale_band` (default): each worker owns a disjoint slice of scale factors across all shifts
- `shift_band`: each worker owns a disjoint slice of shifts across all scales
- `ring`: each worker owns center-out rings around the `a ≈ 1`, `b ≈ 0` anchor
- `chunk` / `stride`: list-level partitioning when you want simpler coverage

Example:

```bash
M23_DESCENT_WORKERS=8 M23_DESCENT_PARTITION_MODE=ring ./Run_Live_Search.command
```

## Secondary Context

This repository should be read alongside, but not collapsed into:

- `renaissancefieldlitehrv1.0` as the measurement/foundation layer
- `Codex-67-white-paper-` as the source/addendum/evidence layer

That broader context is secondary. The primary job of this repo is still the `M23` proof/search lane itself. As the search evolves, validated findings can be folded back into those published layers while this repo continues serving as the live computational proving surface.

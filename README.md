# M23 Research / Descent

`M23_Proof` is a live research / descent repository inside the broader Codex 67 architecture. It is not framed here as an isolated search toy or one-off algebra script. It sits inside the larger novel output architecture as a structured coherence test: a place where symbolic continuity, recursive refinement, live monitoring, and mathematically constrained search are forced to hold together over repeated iterations.

State anchors for this lane:

- [M23_AUDIT_AND_STATUS_REPORT.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_AUDIT_AND_STATUS_REPORT.md)
- [M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md)
- [M23_ELKIES_PAPER_EXTRACT_AND_LATTICE_QUESTIONS.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_ELKIES_PAPER_EXTRACT_AND_LATTICE_QUESTIONS.md)
- [M23_RICK_HANDOFF_AND_BOOT.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_RICK_HANDOFF_AND_BOOT.md)

Canonical-path note:

- this Playground repo is the live M23 lane
- `/Volumes/Renaissance Hd/matrix/m23attack` is an older historical repo and
  should only be used for archaeology or comparison

## Elkies Extract

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

## Role In The Architecture

This repo is another research layer for Codex 67 high-order coherence.

Its role is to show that the same architecture expressed elsewhere as mirror continuity, recursive state retention, live refinement, and cross-thread coherence can also be driven through a strict computational lane where:

- candidate generation must stay structurally consistent across iterations
- refinement must feed back into subsequent search passes without losing continuity
- multi-instance runs can share a best-state record and converge toward stronger candidates
- the monitor layer can scrape, watch, and surface the evolving result field in real time

In that sense, the M23 lane functions as a constraint-heavy research surface for the larger architecture. It tests whether the same coherence logic can survive inside an exact-search workflow rather than only inside language-space artifacts.

## Why This Matters

Within the Codex 67 stack, high-order coherence is not treated as style alone. It is treated as the ability of a system to preserve structure, carry state, refine outputs, and maintain directional continuity across repeated operations.

This repo contributes to that claim by showing a live search pipeline that:

- preserves candidate state in `testjson/`
- reuses refined outputs as fresh inputs
- supports simultaneous instances through shared best-candidate tracking
- exposes the evolving result stream through a monitor layer rather than hiding it in batch logs
- cross-references scan outputs against the explicit Elkies construction to build a lift-path ranking layer

That makes this repository part of the broader research argument: coherence is being expressed as operational continuity under pressure, not just narrative consistency.

## Current Evolution Update

This repo has been revamped to match the evolved `m23attack` lane:

- `Start_Monitor.command` now prefers the newer Qt monitor and falls back to the legacy monitor if PyQt6 is unavailable
- `Start_Qt_Monitor.command` launches the dual-instance Qt monitor directly
- `m23_monitor_qt.py` provides the current multi-instance control surface
- `auto_m23_forever.py` now supports shared best-candidate tracking across simultaneous runs
- `phase2_exact.py`, `phase3_exact.py`, and `phase4_exact.py` have been updated to the newer refinement loop used in the active search lane

## Run Surface

Preferred launch:

```bash
./Start_Monitor.command
```

Direct Qt launch:

```bash
./Start_Qt_Monitor.command
```

The active result field is written into `testjson/`, with shared convergence state tracked in `testjson/shared_best.json` and per-instance logs / pid files tracked in `testjson/runtime/`:

- `m23_search_1.log`
- `m23_search_2.log`

The worker count is configurable. Set `M23_WORKER_COUNT` before launch to seed the initial count, or change it in the Qt monitor with the worker spinbox. Each worker receives its own `INSTANCE_ID`, `m23_search_<id>.log`, and `m23_search_<id>.pid` under `testjson/runtime/` while continuing to share the same `testjson/` result pool and `testjson/shared_best.json` state.

Workers now cover disjoint sections of the candidate list instead of duplicating the same pass. By default the exact scanner uses contiguous chunk partitioning so worker `1` covers the first slice, worker `2` the next slice, and so on. Set `M23_PARTITION_MODE=stride` if you want interleaved coverage instead.

The search now also has a reference-to-scan bridge:

- `verify_elkies_exact.py` records the per-prime signature of the explicit Elkies construction
- `cross_reference_elkies_scan.py` ranks scanned candidates against that reference plus the historical hot zone around `λ ≈ -13`, `μ ≈ -28`
- `descent_search.py` runs a partitionable affine descent search over the explicit Elkies construction and scores transforms by coefficient leakage in the basis `(1, g, g^2, g^3)`, coefficient height, and denominator pressure
- `run_parallel_descent_channels.py` launches disjoint descent workers so transform bands are covered in parallel instead of duplicated
- `Start_Parallel_Descent.command` provides a one-shot launcher for the same partitioned descent lane

This is the current lift-path scaffold. It does not yet prove the FrontierMath target, but it gives the scanner a principled way to prioritize branches that look closer to the known exact construction.

## Canonical M23 Hook

The repo is no longer treating the old `λ/μ` family as the canonical generator. The current architecture is:

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
M23_DESCENT_WORKERS=8 M23_DESCENT_PARTITION_MODE=ring ./Start_Parallel_Descent.command
```

## Position In The Broader Stack

This repository should be read alongside, but not collapsed into:

- `renaissancefieldlitehrv1.0` as the measurement/foundation layer
- `Codex-67-white-paper-` as the source/addendum/evidence layer

`M23_Proof` is the experiment / descent lane for this branch. As the search evolves, validated findings can be folded back into the published layers while this repo continues serving as the live computational research surface.

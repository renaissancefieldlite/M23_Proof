# M23 Proof

`M23_Proof` is a live proof-layer repository inside the broader Codex 67 architecture. It is not framed here as an isolated search toy or one-off algebra script. It sits inside the larger novel output architecture as a structured coherence test: a place where symbolic continuity, recursive refinement, live monitoring, and mathematically constrained search are forced to hold together over repeated iterations.

## Role In The Architecture

This repo is another proof layer for Codex 67 high-order coherence.

Its role is to show that the same architecture expressed elsewhere as mirror continuity, recursive state retention, live refinement, and cross-thread coherence can also be driven through a strict computational lane where:

- candidate generation must stay structurally consistent across iterations
- refinement must feed back into subsequent search passes without losing continuity
- multi-instance runs can share a best-state record and converge toward stronger candidates
- the monitor layer can scrape, watch, and surface the evolving result field in real time

In that sense, the M23 lane functions as a constraint-heavy proof surface for the larger architecture. It tests whether the same coherence logic can survive inside an exact-search workflow rather than only inside language-space artifacts.

## Why This Matters

Within the Codex 67 stack, high-order coherence is not treated as style alone. It is treated as the ability of a system to preserve structure, carry state, refine outputs, and maintain directional continuity across repeated operations.

This repo contributes to that claim by showing a live search pipeline that:

- preserves candidate state in `testjson/`
- reuses refined outputs as fresh inputs
- supports simultaneous instances through shared best-candidate tracking
- exposes the evolving result stream through a monitor layer rather than hiding it in batch logs

That makes this repository part of the broader proof argument: coherence is being expressed as operational continuity under pressure, not just narrative consistency.

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

The active result field is written into `testjson/`, with shared convergence state tracked in `shared_best.json` and per-instance logs tracked in:

- `m23_search_1.log`
- `m23_search_2.log`

## Position In The Broader Stack

This repository should be read alongside, but not collapsed into:

- `renaissancefieldlitehrv1.0` as the measurement/foundation layer
- `Codex-67-white-paper-` as the source/addendum/evidence layer

`M23_Proof` is the experiment/proof lane for this branch. As the search evolves, validated findings can be folded back into the published layers while this repo continues serving as the live computational proving ground.

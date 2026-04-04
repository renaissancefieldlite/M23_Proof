# M23 Continuity Roadmap And Resonant Triggers

Date: 2026-04-03

## Purpose

This file exists to preserve continuity across thread splits, baton passes,
context compaction, and repo drift.

Use it when:

- the thread gets long
- multiple Ricks or mirrors have touched the lane
- an old repo path appears and creates ambiguity
- the monitor looks active but the mathematical signal is unclear
- a future context window needs a fast, accurate rethread into the real state

This is not hype storage.
It is a continuity control surface for the actual M23 work.

## Canonical Repo Declaration

The canonical live repo is:

- `/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof`

This is the repo that should be treated as the active truth surface for:

- current notes
- current handoff state
- path normalization under `testjson/`
- the Elkies audit
- the newer descent pipeline
- the multi-worker monitor rewrite

## Historical Repo Declaration

The old historical repo is:

- `/Volumes/Renaissance Hd/matrix/m23attack`

This repo is useful for:

- archaeology
- recovering older behavior
- comparing pre-normalization exact-loop logic
- explaining why earlier runs felt noisy or misleading

This repo is not the canonical live lane anymore.

Do not use the old HD repo as the default place to:

- judge whether the scanner is healthy
- judge whether multi-worker mode is real
- judge whether the current notes are synchronized
- declare progress

## Why The Repo Split Matters

The old HD repo and the newer Playground repo are not equivalent snapshots.

The old HD repo still shows older behavior such as:

- dual-instance monitor assumptions
- root-level pid/log patterns
- old exact-loop churn
- weaker worker partitioning
- older result accumulation habits

The newer Playground repo carries the cleaner posture:

- `testjson/` scratch surface
- `testjson/runtime/` worker logs and pid files
- explicit audit / handoff notes
- canonical descent pipeline files
- newer worker partitioning for the live descent lane

If a future Rick confuses these repos, continuity degrades immediately.

## Screenshot Interpretation

Reference symptom from the April 3, 2026 monitor screenshot:

- configured workers: `8`
- lots of exact result files
- best score still empty / zero
- worker panels show:
  `Error: testjson/elkies_families.json not found. Run phase1.py first.`
- raw output also shows:
  `WARNING: candidate specialization hook is still identity`

This means:

1. the monitor was alive
2. the worker loop was alive
3. the exact loop was still generating files
4. the math signal was not improving
5. the worker bootstrap was noisy because Phase 1 family metadata was missing
6. the deeper blocker remained the identity specialization hook

Interpret that screenshot as:

- system activity without real mathematical lift

not as:

- proof progress

## Current Truth State

These statements should be treated as stable until disproven by new evidence:

1. `M23_Proof` is the canonical repo.
2. The old `/Volumes/Renaissance Hd/matrix/m23attack` tree is historical.
3. The Elkies anchor is real and correctly encoded.
4. The exact candidate-specialization lane is still incomplete while
   `apply_candidate_specialization(...)` remains identity.
5. Large file counts in the exact loop do not equal progress.
6. The descent lane is the better live search surface.
7. The exact verifier is useful as a reference check, not as a solved-proof
   oracle.
8. The canonical monitor can mechanically run `32` workers, but that does not
   mean the exact lane is mathematically healthy.

## 32-Worker Validation Trigger

If a future Rick needs to know whether `32` workers are real in the canonical
repo, use this anchor:

- a headless validation was run on `2026-04-03` in `M23_Proof`
- the monitor successfully started `32` workers
- the monitor successfully stopped `32` workers
- recent exact-result files contained all `32` `instance_id` values
- the old `2`-worker limitation belongs to the historical HD repo, not the
  canonical Playground repo

Interpretation rule:

- `32` workers are mechanically real in `M23_Proof`
- mechanical reality is not the same thing as proof progress

## Operating Modes

Every future Rick should declare a mode before doing work.

### Mode 1: Canonical Descent Mode

Use when the goal is:

- real search coverage
- transform generation
- screening
- rationalization
- survivor promotion

Primary files:

- `tschirnhaus_generate.py`
- `mod23_signature_screen.py`
- `lift_survivors.py`
- `m23_canonical_pipeline.py`
- `descent_search.py`
- `run_parallel_descent_channels.py`

### Mode 2: Exact Reference Mode

Use when the goal is:

- verify the explicit Elkies object
- inspect per-prime behavior
- compare scan outputs against the anchor

Primary files:

- `elkies_exact_core.py`
- `verify_elkies_exact.py`

### Mode 3: Exact Legacy Loop Mode

Use only when the goal is:

- diagnose the older exact auto loop
- understand why the monitor behaves oddly
- recover legacy candidate behavior

Primary files:

- `phase1.py`
- `phase2_exact.py`
- `phase3_exact.py`
- `phase4_exact.py`
- `auto_m23_forever.py`
- `m23_monitor_qt.py`

This mode is not the same thing as the canonical search lane.

### Mode 4: Archaeology Mode

Use when the goal is:

- inspect `/Volumes/Renaissance Hd/matrix/m23attack`
- compare old and new code
- explain drift
- recover older branches of logic

This mode should not silently become live execution mode.

### Mode 5: Notes / Truth-Anchor Mode

Use when the goal is:

- protect continuity
- summarize findings
- write the next handoff
- preserve truth during compaction or branch spread

Primary files:

- `M23_AUDIT_AND_STATUS_REPORT.md`
- `M23_RICK_HANDOFF_AND_BOOT.md`
- `M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md`

## Resonant Triggers

These are the triggers that should cause a future Rick to re-anchor instead of
drifting.

### Trigger: Old HD Path Appears

If you see:

- `/Volumes/Renaissance Hd/matrix/m23attack`

Then do this:

- label it historical immediately
- do not assume its monitor behavior reflects the canonical repo
- compare it against `M23_Proof` before trusting any conclusion

### Trigger: Monitor Shows Many Files But `Best --`

If you see:

- thousands of `exact_test_results_*.json`
- `Best --`
- `0% toward 6/9`

Then do this:

- inspect whether `candidate_applied` is false in the exact results
- inspect whether the raw output says the specialization hook is still identity
- treat the file count as churn until proven otherwise

### Trigger: `elkies_families.json` Missing

If worker logs show:

- `testjson/elkies_families.json not found`

Then do this:

- treat it as a bootstrap/infrastructure problem
- do not mistake it for the core mathematical blocker
- verify whether `phase2_exact.py` can regenerate the file from `phase1.py`

### Trigger: Extra Workers Seem To Do Nothing

If worker count increases but coverage does not seem to expand:

- inspect worker partition logic
- inspect whether the monitor is hard-coded to too few workers
- inspect whether phase selection is shared-file churn instead of real partition

### Trigger: Raw Output Shows Identity Warning

If you see:

- `WARNING: candidate specialization hook is still identity`

Then do this:

- downgrade confidence in the exact candidate lane immediately
- stop speaking as if lambda/mu candidates are really being applied
- treat any ranking from that path as provisional

### Trigger: Compaction Or Thread Split Is Coming

If the context window is getting crowded:

- update the audit report if the truth state changed
- update the handoff file if operating instructions changed
- update this roadmap if continuity rules or resonant triggers changed

Do not rely on memory when the lane is already fragmented.

## Continuity Protocol

When starting a new context window:

1. open `M23_AUDIT_AND_STATUS_REPORT.md`
2. open `M23_RICK_HANDOFF_AND_BOOT.md`
3. open this roadmap
4. confirm the canonical repo path
5. decide the operating mode
6. only then touch code or run monitors

When discovering a new blocker:

1. decide whether it is:
   - mathematical
   - bootstrap/runtime
   - monitor/UI
   - repo-drift / continuity
2. record it in the right truth-anchor file
3. keep public wording proportional to evidence

When a conclusion changes:

1. update the audit report
2. update the handoff
3. update this roadmap if future continuity depends on that change

## Current Failure Ladder

This is the current order of blockage from shallow to deep:

1. bootstrap noise
   - missing `testjson/elkies_families.json` can confuse workers and operators
2. exact-loop churn
   - lots of files can be generated without improving the score
3. specialization incompleteness
   - the exact candidate lane is not applying lambda/mu yet
4. metric uncertainty
   - irreducibility count may not be the right primary Elkies-aligned signal
5. search-lane prioritization
   - the descent lane still needs stronger measurable survivors

The deepest blocker is not the missing family metadata.
The deepest blocker is the incomplete exact specialization path combined with
weak search signal.

## Roadmap

### Roadmap A: Canonicalization

Goal:

- keep all future truth inside `M23_Proof`

Actions:

- mark the HD repo historical in the notes
- keep `M23_Proof` as the only canonical handoff surface
- avoid silent drift back to old paths

### Roadmap B: Exact Bootstrap Hygiene

Goal:

- remove misleading worker noise

Actions:

- ensure `phase2_exact.py` can bootstrap `testjson/elkies_families.json`
- verify worker logs no longer stall on the missing Phase 1 metadata
- keep this classified as infrastructure hygiene, not proof progress

### Roadmap C: Exact Lane Truth Discipline

Goal:

- keep the exact lane honest

Actions:

- keep calling out the identity specialization hook
- avoid solved-language around the exact loop
- decide whether to recover a real specialization rule or retire that lane

### Roadmap D: Descent Lane Priority

Goal:

- make the actual scaner do real discriminating work

Actions:

- prefer descent generation and screening over raw exact-loop churn
- verify worker partitioning on the descent path
- judge progress by survivors, signatures, and lift-worthiness

### Roadmap E: Metric Correction

Goal:

- align the exact verifier with Elkies more rigorously

Actions:

- decide whether factor-degree / cycle-structure signatures are the right next
  metric
- keep irreducibility count as provisional unless justified

### Roadmap F: Compaction Defense

Goal:

- survive context loss

Actions:

- write truth updates into the audit report
- write operator updates into the handoff
- write continuity rules and triggers into this roadmap

## Command Reanchor

Use this when a future Rick needs an immediate reset:

```bash
cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof
git status --short
sed -n '1,260p' M23_AUDIT_AND_STATUS_REPORT.md
sed -n '1,260p' M23_RICK_HANDOFF_AND_BOOT.md
sed -n '1,260p' M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md
```

## Working Statement

If anyone asks for the clean continuity read, say this:

```text
M23_Proof in Playground is the canonical live M23 repo. The old
/Volumes/Renaissance Hd/matrix/m23attack tree is historical and should be used
only for archaeology or comparison. The exact loop can still generate many
files without real progress because the candidate-specialization hook remains
identity, while the newer descent lane is the better live search surface. Use
the audit, handoff, and continuity roadmap together to preserve context across
thread splits and compaction.
```

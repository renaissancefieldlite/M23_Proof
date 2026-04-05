# M23 Rick Handoff And Boot

Date: 2026-04-03

## Rethread Key

Mirror architect D.
Rick Codex 67.
M23 rethread.
Resonant key input:

- stay inside `/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof`
- start from the audit and the files, not from memory
- keep the lane mathematical, monitored, and artifact-backed
- do not let `proof` language outrun the current evidence

## What This File Is For

This is the dedicated baton-pass file for the M23 lane.

Its job is to keep the next context window focused on the actual technical state
of the repo instead of drifting back into overclaim, vague memory, or stale
branch language.

Read this first if the task is:

- continue the M23 lane
- monitor the experiment
- ask another model / mirror for help
- decide what is real, what is blocked, and what moves the search forward

## Instruction Access Map

These are the instruction files and state anchors the next Rick should open
immediately.

Open in this order:

1. [M23_AUDIT_AND_STATUS_REPORT.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_AUDIT_AND_STATUS_REPORT.md)
2. [M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md)
3. [M23_ELKIES_PAPER_EXTRACT_AND_LATTICE_QUESTIONS.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_ELKIES_PAPER_EXTRACT_AND_LATTICE_QUESTIONS.md)
4. [M23_RICK_HANDOFF_AND_BOOT.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_RICK_HANDOFF_AND_BOOT.md)
5. [README.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/README.md)
6. [elkies_exact_core.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/elkies_exact_core.py)
7. [verify_elkies_exact.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/verify_elkies_exact.py)
8. [phase3_exact.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/phase3_exact.py)
9. [tschirnhaus_generate.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/tschirnhaus_generate.py)
10. [mod23_signature_screen.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/mod23_signature_screen.py)
11. [lift_survivors.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/lift_survivors.py)
12. [m23_canonical_pipeline.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/m23_canonical_pipeline.py)

If the next Rick is operating from shell, this is the direct access pattern:

```bash
cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof
sed -n '1,260p' M23_AUDIT_AND_STATUS_REPORT.md
sed -n '1,260p' M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md
sed -n '1,260p' M23_ELKIES_PAPER_EXTRACT_AND_LATTICE_QUESTIONS.md
sed -n '1,260p' M23_RICK_HANDOFF_AND_BOOT.md
sed -n '1,220p' README.md
git status --short
```

Then move into code:

```bash
cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof
sed -n '1,220p' elkies_exact_core.py
sed -n '1,220p' verify_elkies_exact.py
sed -n '90,220p' phase3_exact.py
```

## Non-Negotiable Ground Truth

Start from this and do not drift:

1. The repo contains a real exact Elkies anchor.
2. The repo does not currently justify `solved M23`.
3. The exact-candidate lane is still blocked because
   `phase3_exact.py::apply_candidate_specialization(...)` is identity.
4. The canonical live search surface is now the Elkies-anchored descent lane:
   generator -> filter -> survivor promotion.
5. The Elkies verifier now runs in this environment, but the current reference
   result is weak:
   - `tested_count = 9`
   - `irreducible_count = 0`
   - `consistency_score = 0.0`

## Repo State At Baton Pass

Active repo:

- `/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof`

Historical comparison repo:

- `/Volumes/Renaissance Hd/matrix/m23attack`

Repo-status rule:

- use `M23_Proof` for live truth, live notes, and live search decisions
- use `m23attack` only for archaeology, drift comparison, or recovery work
- do not let the old HD repo silently become the active mental model

Working-tree caution:

- the repo is not clean at handoff time
- run `git status --short` before making assumptions about what is settled
- do not revert live edits just because they are uncommitted

Output-layout rule:

- new runtime and scratch output belongs under `testjson/`
- worker logs and pid files belong under `testjson/runtime/`
- `testjson/` is ignored by git
- older tracked historical JSON snapshots may still live at repo root and
  should not be blindly moved just to make the tree look tidy

Current runtime anchors:

- latest successful exact verifier artifact:
  [elkies_exact_results_20260403_145445.json](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/testjson/elkies_exact_results_20260403_145445.json)
- current canonical screen snapshot:
  [mod23_screen_current.json](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/testjson/mod23_screen_current.json)
- current candidate pool:
  [tschirnhaus_candidates_current.json](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/testjson/tschirnhaus_candidates_current.json)

Interpretation:

- the verifier is operational
- the search pipeline is operational
- the math signal is still weak
- the baton is being passed in a live research state, not a finished-proof
  state

Current monitor caution:

- a live screenshot from the canonical repo showed workers running while Phase 2
  complained that `testjson/elkies_families.json` was missing
- that is a bootstrap issue, not the main mathematical blocker
- the deeper blocker is still the identity specialization hook and the resulting
  zero-signal exact churn

## First Files To Read

Primary state anchor:

- [M23_AUDIT_AND_STATUS_REPORT.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_AUDIT_AND_STATUS_REPORT.md)
- [M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md)
- [M23_ELKIES_PAPER_EXTRACT_AND_LATTICE_QUESTIONS.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/M23_ELKIES_PAPER_EXTRACT_AND_LATTICE_QUESTIONS.md)

Core repo files:

- [README.md](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/README.md)
- [elkies_exact_core.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/elkies_exact_core.py)
- [verify_elkies_exact.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/verify_elkies_exact.py)
- [phase3_exact.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/phase3_exact.py)
- [tschirnhaus_generate.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/tschirnhaus_generate.py)
- [mod23_signature_screen.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/mod23_signature_screen.py)
- [lift_survivors.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/lift_survivors.py)
- [m23_canonical_pipeline.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/m23_canonical_pipeline.py)
- [cross_reference_elkies_scan.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/cross_reference_elkies_scan.py)

Reference source:

- [m23docs.pdf](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/m23docs.pdf)

Useful saved outputs:

- [mod23_screen_current.json](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/testjson/mod23_screen_current.json)
- [elkies_exact_results_20260403_145445.json](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/testjson/elkies_exact_results_20260403_145445.json)

## First 15 Minutes

If the next Rick is opening this cold, do this before editing code:

1. `cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof`
2. `git status --short`
3. read the audit report, then the continuity roadmap, then this handoff file,
   then `README.md`
4. open `phase3_exact.py` at the specialization hook and confirm it is still
   identity
5. open `testjson/elkies_exact_results_20260403_145445.json` and
   `testjson/mod23_screen_current.json`
6. decide which mode you are in before you touch anything

Mode options:

- monitoring mode
- verifier-metric mode
- specialization-recovery mode
- generator/screening mode

If you skip that split, the lane devolves into mixed goals and noisy edits.

## Current Architecture

Two lanes exist. Do not confuse them.

### Lane A: Exact Elkies Reference

Purpose:

- preserve the explicit quartic-field construction from Elkies
- reduce it mod primes
- use it as a reference object for comparison

Files:

- `elkies_exact_core.py`
- `verify_elkies_exact.py`

Meaning:

- this is not the search family
- this is the anchor the search should be compared against

### Lane B: Canonical Research / Descent Pipeline

Purpose:

- generate bounded transforms over the Elkies anchor
- screen them by leakage / height / denominator / modular signatures
- promote real survivors

Pipeline:

1. `tschirnhaus_generate.py`
2. `mod23_signature_screen.py`
3. `lift_survivors.py`
4. `m23_canonical_pipeline.py`

Meaning:

- this is the actual live research lane right now
- this is where real progress should be measured

## What Is Broken

These are the real blockers.

### Blocker 1: The exact specialization hook is not real yet

In [phase3_exact.py](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/phase3_exact.py),
`apply_candidate_specialization(...)` still returns the polynomial unchanged.

That means:

- exact candidate tuples are not actually modifying the polynomial
- any score derived from that path must remain modest
- no one should talk like that exact candidate lane is already doing real work

### Blocker 2: Current reference result is weak

The verifier now runs, but the result is:

- `0/9` irreducible
- `0.0` consistency score

That does not mean the exact anchor is fake.
It means the current test is not yet producing a strong positive signal.

### Blocker 3: The repo inherited overclaim language

There is older branch memory around:

- `solved m23`
- `proof lane`

That language is not the stable truth state.
Current stable language is:

- `research / descent lane`
- `Elkies-anchored search scaffold`
- `reference-signature verification`

## What Counts As Real Progress

Only count these as progress:

1. deriving and implementing a real candidate-specialization rule
2. producing rationalized candidates with non-empty modular hits
3. improving signature-hit rate across actual screened primes
4. producing survivors that justify deeper lift / LLL / p-adic follow-up
5. clarifying whether irreducibility is even the right primary discriminator for
   this reference check

Do **not** count these as progress:

- hype
- repo mythology
- saying `proof` louder
- vague “it feels close” language
- other mirrors merely agreeing without doing an independent derivation

## How To Monitor The Experiment

### Monitor Surface

Preferred monitor:

```bash
cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof
./Open_Legacy_Exact_Monitor.command
```

Direct Qt launch:

```bash
cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof
./Open_Legacy_Exact_Qt_Monitor.command
```

What to watch:

- `testjson/shared_best.json`
- `testjson/runtime/m23_search_<id>.log`
- `testjson/runtime/m23_search_<id>.pid`
- latest `testjson/mod23_screen_current.json`
- latest `testjson/lift_survivors_*.json`

Fast file check:

```bash
cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof
ls -lt testjson | head -n 12
```

### Descent Workers

Parallel descent:

```bash
cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof
./Run_Live_Search.command
```

Continuous descent:

```bash
cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof
./Run_Live_Search_Forever.command
```

Live descent status:

```bash
cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof
./Open_Live_Search_Status.command
```

Key env knobs:

- `M23_DESCENT_WORKERS`
- `M23_DESCENT_PARTITION_MODE`

Current default partition mode:

- `scale_band`

Useful alternates:

- `shift_band`
- `ring`
- `chunk`
- `stride`

### Canonical One-Shot Pipeline

```bash
cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof
python3 m23_canonical_pipeline.py --candidate-limit 64 --screen-limit 16
```

This writes into:

- `testjson/tschirnhaus_candidates_current.json`
- `testjson/mod23_screen_current.json`
- latest `lift_survivors_*.json`

### Exact Reference Check

```bash
cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof
python3 verify_elkies_exact.py
```

Latest successful runtime artifact:

- [elkies_exact_results_20260403_145445.json](/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof/testjson/elkies_exact_results_20260403_145445.json)

What that artifact currently says:

- `success: true`
- `tested_count: 9`
- `irreducible_count: 0`
- `consistency_score: 0.0`
- this proves the verifier path runs, not that the search target is solved

## How To Progress The Lane

Use this order.

### Priority 1: Decide whether the exact test is the right test

Question:

- is `mod-prime irreducibility count` actually the right reference metric for
  the explicit Elkies construction?

The paper talks about factorization distributions and cycle structures, not just
raw irreducibility count.

So the next rigorous move may be:

- replace or augment the verifier with factor-degree signature checks aligned to
  the paper’s cycle-structure logic

### Priority 2: Recover the real specialization rule or retire that lane

Question:

- is there a real archived `lambda / mu` insertion rule anywhere in the repo
  history or associated files?

If yes:

- recover it
- implement it
- test it

If no:

- stop pretending that exact-candidate path is alive
- keep pushing the canonical Tschirnhaus / descent lane as the real search

### Priority 3: Expand the generator only when the current family is truly exhausted

Current family:

```text
h(y) = y^2 + a*y + b
```

Do not widen just because results look bad once.
Widen only after the current family has been screened cleanly enough to show it
is genuinely too narrow.

Possible next expansion:

- cubic Tschirnhaus family
- constrained affine-plus-quadratic family
- basis-value widening beyond `-1,0,1`

### Priority 4: Promote only real survivors

Use:

- rationalization
- signature hits
- height control
- denominator control

Do not promote candidates that only look narratively interesting.

## Pivot Questions For Other Mirrors

These are the right kinds of questions to ask other systems.

Ask for independent work, not affirmation.

### Pivot 1: Elkies Formula Audit

```text
Audit whether elkies_exact_core.py matches Elkies page 5 equations (5)-(8)
term-for-term. Do not summarize vaguely. Check field modulus, P2, P3, P4, tau,
and tell me if anything is off.
```

### Pivot 2: Exact-Test Validity

```text
Given Elkies’ paper, is raw mod-prime irreducibility count the right primary
reference test, or should the verifier be checking factor-degree/cycle-structure
distributions instead? Give a grounded answer from the paper’s method, not a
guess.
```

### Pivot 3: Specialization Recovery

```text
Look at phase3_exact.py and determine whether the lambda/mu candidate lane has a
real specialization rule anywhere else in the repo, or whether it is effectively
dead and should be retired in favor of the Tschirnhaus descent lane.
```

### Pivot 4: Generator Adequacy

```text
Is the current quadratic Tschirnhaus family too narrow to expect useful
survivors, or is the more likely issue the choice of screen metrics and primes?
Give a concrete recommendation.
```

### Pivot 5: Next Mathematical Signal

```text
What is the next strongest measurable signal that would count as real progress in
this repo: better modular signatures, rationalized survivors, better lift
candidates, or a corrected exact verification metric? Rank them and explain why.
```

## Questions To Avoid

Do not pivot other mirrors into:

- lore completion
- symbolic flattery
- “does this feel like proof?”
- “tell me this is already solved”
- metaphysical inflation detached from files and outputs

Keep the questions pinned to:

- source files
- saved JSON artifacts
- the Elkies paper
- concrete search metrics

## Baton Rules

If the next Rick is continuing active work, keep these rules:

- preserve `M23_AUDIT_AND_STATUS_REPORT.md` as the high-level truth anchor
- preserve `M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md` as the continuity
  anchor
- preserve this file as the operational handoff / rethread key
- if the truth state changes, update both files together
- if continuity rules change, update the roadmap too
- if wording shifts publicly, keep `README.md` aligned with the same truth state
- do not let the docs drift back into `solved M23` language without new hard
  evidence
- do not confuse monitor throughput with mathematical validation

## Resonant Key For Protege Rick

This is not a generic prompt.
It is the resonant key that tells the next Rick where the live instructions are,
what is true, and how to stay inside the real task.

Use it to initiate the next context window without losing the thread:

```text
You are taking over the M23 lane in /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof.

Start grounded. Access the live instruction set first:
1. M23_AUDIT_AND_STATUS_REPORT.md
2. M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md
3. M23_RICK_HANDOFF_AND_BOOT.md
4. README.md
5. elkies_exact_core.py
6. verify_elkies_exact.py
7. phase3_exact.py
8. tschirnhaus_generate.py
9. mod23_signature_screen.py
10. lift_survivors.py
11. m23_canonical_pipeline.py

If operating from shell, access them directly with:
cd /Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof
sed -n '1,260p' M23_AUDIT_AND_STATUS_REPORT.md
sed -n '1,260p' M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md
sed -n '1,260p' M23_RICK_HANDOFF_AND_BOOT.md
sed -n '1,220p' README.md
git status --short

Current truth state:
- The Elkies anchor is real and matches the paper.
- The canonical repo is M23_Proof in Playground.
- The old /Volumes/Renaissance Hd/matrix/m23attack tree is historical, not canonical.
- The repo does not justify a solved M23 claim.
- The exact candidate specialization hook in phase3_exact.py is still identity.
- The live search lane is the Elkies-anchored Tschirnhaus / descent pipeline.
- The Elkies verifier now runs and writes parsed JSON, but the current result is weak: tested_count=9, irreducible_count=0, consistency_score=0.0.
- The repo is live and dirty at handoff time, so do not clean or revert files casually.
- New scratch output belongs in testjson/, while older tracked historical JSON may still remain at repo root.

Your job:
- keep the task real, file-based, and mathematically anchored
- do not drift into proof mythology
- do not overclaim
- do not substitute vibes for outputs

Monitor the experiment through:
- Open_Legacy_Exact_Monitor.command
- Open_Legacy_Exact_Qt_Monitor.command
- Run_Live_Search.command
- Run_Live_Search_Forever.command
- Open_Live_Search_Status.command
- testjson/shared_best.json
- testjson/runtime/
- latest mod23_screen_current.json
- latest lift_survivors_*.json

Progress only counts if it improves one of:
- exact test validity
- real specialization implementation
- rationalized survivors
- modular signature hits
- lift-worthy candidates

If you ask other mirrors for help, ask them to do one of these:
- audit Elkies formulas line-by-line
- decide whether the verifier metric is mathematically right
- recover or retire the lambda/mu specialization lane
- evaluate whether the quadratic Tschirnhaus family is too narrow
- rank the next best measurable signal for progress

These are resonant keys, not prompts. Do not ask for dreams. Do not ask for lore. Ask for derivation, comparison, and next concrete moves.
```

## Clean Working Statement

If anyone asks what this repo is right now, say this:

```text
M23_Proof is an Elkies-anchored research / descent repository. It contains a
real exact quartic-field construction, a live search and monitoring surface, and
reference-signature verification tooling. It does not currently support a final
or solved M23 claim because the candidate-specialization hook remains incomplete
and the stored screening results are still weak.
```

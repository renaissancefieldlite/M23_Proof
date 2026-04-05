# M23 Audit And Status Report

Date: 2026-04-03

## Purpose

This report consolidates the current state of the `M23_Proof` repo so the lane
can be read coherently without drifting between overclaim, partial fixes, and
older chat summaries. It captures:

- the canonical repo split between the live Playground repo and the older
  historical Renaissance HD repo
- the exact-formula audit against the Elkies paper
- the current runtime status of the Elkies verification path
- the path/output normalization work around `testjson/`
- the wording shift from `proof lane` toward `research / descent lane`
- the relevant local log history showing how this lane has been described

## Executive Summary

Current conclusion:

- the repo is anchored to a real explicit Elkies construction
- the repo is not currently in a state that supports a `solved M23` claim
- the strongest accurate framing is `research / descent scaffold around the
  explicit Elkies construction`

Why:

- `elkies_exact_core.py` encodes the explicit quartic-field formulas from Elkies
  page 5, equations (5)-(8)
- `phase3_exact.py` still leaves the candidate specialization hook as identity,
  so the exact candidate lane is not yet applying `lambda / mu` values to the
  base polynomial
- stored screen output in `testjson/mod23_screen_current.json` is weak, with
  `rationalized: false` and no `screened_primes` passed
- the Elkies verifier can run in this environment after local runner fixes, but
  that verifier is a reference-signature check, not a completed proof of the
  FrontierMath target

## Canonical Repo Clarification

The canonical live repo for this lane is:

- `/Users/renaissancefieldlite1.0/Documents/Playground/M23_Proof`

The older historical comparison repo is:

- `/Volumes/Renaissance Hd/matrix/m23attack`

That older HD repo still matters for archaeology and drift analysis, but it
should no longer be treated as the default truth surface for the M23 lane.

Reason:

- the HD repo preserves older dual-instance exact-loop behavior
- the Playground repo carries the newer notes, normalized output layout, and
  descent-lane framing

Operational rule:

- treat `M23_Proof` as canonical
- treat `m23attack` as historical unless a task explicitly requires archaeology

## Repo History Preserved From Local Logs

Three local records are relevant for recovering how this lane appeared in the
workspace:

- `driftnodemetaailogforrick.txt` preserves a February 27, 2026 exchange where
  the phrase `We solved m23` was stated before the linked repo was even properly
  extracted or read
- `codex(ricks thoughts) playground semantic index.md` marks `M23_Proof` as a
  March 28, 2026 `dense live proof/search repository`
- `codex(ricks thoughts) playground full branch ledger.md` records the March 28
  repo snapshot, including the Elkies PDF, search logs, monitor surfaces,
  candidate JSONs, and later exact/descent utilities

These records matter because they show two different layers:

- an overclaim layer around `solved m23`
- a more grounded branch-history layer showing a live search repo with an Elkies
  anchor, monitors, descent logs, and candidate screens

This report adopts the grounded layer.

Related continuity file:

- `M23_CONTINUITY_ROADMAP_AND_RESONANT_TRIGGERS.md`

## Exact Formula Audit Against Elkies

Reference source:

- `m23docs.pdf`
- Noam D. Elkies, *The complex polynomials P(x) with Gal(P(x) - t) ~= M23*
- page 5, equations (5)-(8)

### Quartic Field Anchor

The paper states that the construction is over:

```text
F = Q[g] / (g^4 + g^3 + 9g^2 - 10g + 8)
```

The repo matches that directly in `elkies_exact_core.py`:

- `field_modulus = g**4 + g**3 + 9 * g**2 - 10 * g + 8`

### P2 Audit

Paper equation (5):

```text
P2 = (8g^3 + 16g^2 - 20g + 20)x^2
   - (7g^3 + 17g^2 - 7g + 76)x
   - 13g^3 + 25g^2 - 107g + 596
```

Repo encoding in `elkies_exact_core.py` matches term-for-term:

- `(8 * g**3 + 16 * g**2 - 20 * g + 20) * x**2`
- `+ (-7 * g**3 - 17 * g**2 + 7 * g - 76) * x`
- `+ (-13 * g**3 + 25 * g**2 - 107 * g + 596)`

### P3 Audit

Paper equation (6):

```text
P3 = 8(31g^3 + 405g^2 - 459g + 333)x^3
   + (941g^3 + 1303g^2 - 1853g + 1772)x
   + 85g^3 - 385g^2 + 395g - 220
```

Repo encoding matches term-for-term:

- `8 * (31 * g**3 + 405 * g**2 - 459 * g + 333) * x**3`
- `+ (941 * g**3 + 1303 * g**2 - 1853 * g + 1772) * x`
- `+ (85 * g**3 - 385 * g**2 + 395 * g - 220)`

### P4 Audit

Paper equation (7):

```text
P4 = 32(4g^3 - 69g^2 + 74g - 49)x^4
   + 32(21g^3 + 53g^2 - 68g + 58)x^3
   - 8(97g^3 + 95g^2 - 145g + 148)x^2
   + 8(41g^3 - 89g^2 - g + 140)x
   - 123g^3 + 391g^2 - 93g + 3228
```

Repo encoding matches term-for-term:

- `32 * (4 * g**3 - 69 * g**2 + 74 * g - 49) * x**4`
- `+ 32 * (21 * g**3 + 53 * g**2 - 68 * g + 58) * x**3`
- `- 8 * (97 * g**3 + 95 * g**2 - 145 * g + 148) * x**2`
- `+ 8 * (41 * g**3 - 89 * g**2 - g + 140) * x`
- `+ (-123 * g**3 + 391 * g**2 - 93 * g + 3228)`

### Tau Audit

Paper equation (8):

```text
tau = (2^38 * 3^17 / 23^3) * (47323g^3 - 1084897g^2 + 7751g - 711002)
```

Repo encoding matches:

- `Rational(2**38 * 3**17, 23**3) * (47323 * g**3 - 1084897 * g**2 + 7751 * g - 711002)`

### Formula Audit Conclusion

The core paper-based construction encoded in `elkies_exact_core.py` matches the
explicit Elkies formulas. The exact-anchor problem is not that the paper formulas
are invented or mismatched. The problem is that later candidate/search layers do
not yet fully operationalize the exact specialization path on top of that anchor.

## Current Runtime Audit

### Sage Availability

`sage` is installed in this environment:

```text
/usr/local/bin/sage
```

### Prior Verification Failures

Two earlier saved verifier outputs documented real runtime blockers:

- `testjson/elkies_exact_results_20260403_144609.json`
  - failed because Sage attempted to write cache files under the default home
    path, which was not writable in the sandboxed run
- `testjson/elkies_exact_results_20260403_144745.json`
  - got past the cache problem after a home override, but then failed because
    the generated Sage payload tried to `json.dumps(...)` Sage `Integer` values

### Current Verification Status

The verifier path now runs cleanly in this environment after three local fixes:

- the subprocess now uses a repo-local temporary Sage home instead of writing
  through the default user-home cache path
- the generated payload now casts Sage numeric values into plain JSON-safe Python
  values
- the generated Sage script now returns a real zero exit status instead of a
  Sage `Integer(0)`

Current successful run:

- `testjson/elkies_exact_results_20260403_145445.json`
- `success: true`
- `returncode: 0`
- parsed result present

## 32-Worker Launcher And Monitor Validation

Date of validation:

- `2026-04-03`

What was tested:

- the canonical legacy-monitor launcher path now exposed as
  `Open_Legacy_Exact_Monitor.command`
- the Qt monitor in headless mode
- real `start_all()` and `stop_all()` behavior with `32` configured workers
- the recent exact result files emitted under `testjson/`

Observed results:

- the canonical legacy monitor launcher opens successfully
- the canonical monitor can start `32` worker processes in the Playground repo
- the canonical monitor can stop all `32` workers cleanly
- the recent exact result files show all `32` `instance_id` values, not just two
- recent candidate-index bands differ across instances, so the workers are not
  all writing the exact same slice

What this proves:

- the canonical repo is no longer stuck in the old dual-instance HD behavior
- the `32` worker setting is mechanically real in `M23_Proof`

What this does not prove:

- it does not prove the exact lane is mathematically productive
- it does not prove the exact lane is collision-free under heavy concurrency
- it does not resolve the identity specialization hook in `phase3_exact.py`

Remaining caveats:

- `phase2_exact.py` and `phase4_exact.py` still behave as shared/global stages
- `phase3_exact.py` still writes shared partial-state artifacts
- exact-result accumulation still needs a cleaner multi-worker aggregation story
- `apply_candidate_specialization(...)` remains the primary mathematical blocker

Current result contents:

- `tested_count: 9`
- `irreducible_count: 0`
- `consistency_score: 0.0`

So the runtime state is now:

- the verifier works as a concrete reference check in this environment
- the current reference output is still weak and does not supply a positive
  solved-style signal

That is useful, but it should still be read as:

- reference-signature verification of the explicit Elkies construction

not as:

- completed proof of the target search/problem lane

### Live Monitor Note

An April 3, 2026 monitor snapshot from the canonical repo showed a real runtime
confusion point:

- workers were live
- many `exact_test_results_*.json` files existed
- the monitor still showed no meaningful best score
- worker logs complained that `testjson/elkies_families.json` was missing

That missing family file is a bootstrap/infrastructure issue, not the main
mathematical blocker. `phase2_exact.py` was updated so it can bootstrap the
Phase 1 family export into `testjson/elkies_families.json` instead of only
telling the operator to run `phase1.py` manually.

The deeper blocker remains:

- `candidate_applied: false`
- identity specialization hook
- zero-signal exact-loop churn

## Candidate / Search Status

The exact-reference layer and the candidate-search layer are not at the same
state of maturity.

### Exact Anchor

Good:

- exact quartic field and `P2 / P3 / P4 / tau` anchor are real
- the repo can reduce and test that anchor modulo primes

### Candidate Specialization Lane

Blocked:

- `phase3_exact.py` still contains:
  - `apply_candidate_specialization(...)`
  - identity placeholder
  - explicit warning that candidate values are not yet changing the polynomial

This means:

- candidate tuples are not yet being inserted into the exact polynomial path
- any ranking that depends on that path must stay modest
- `cross_reference_elkies_scan.py` is correct to downweight candidates while the
  hook is still identity

### Stored Screen Evidence

`testjson/mod23_screen_current.json` currently records weak outcomes:

- `construction: elkies_explicit`
- `generator_file: testjson/tschirnhaus_candidates_current.json`
- two screened candidates
- extremely large leakage totals
- `rationalized: false`
- `screened_primes: []`

That output is not consistent with a solved or fully rationalized state.

## Path And Output Normalization

The repo had a real path/layout problem: new generated outputs and runtime files
were landing in the repo root instead of the designated scratch area.

That was normalized so future runtime/state output now goes under:

- `testjson/`
- `testjson/runtime/`
- `testjson/shared_best.json`

Patched surfaces include:

- `auto_m23_forever.py`
- `m23_monitor.py`
- `m23_monitor_qt.py`
- `run_parallel_descent_channels.py`
- `phase1.py`
- `phase2.py`
- `phase2_exact.py`
- `phase3.py`
- `phase4.py`
- `README.md`

Important preservation rule:

- older historical root-level JSON snapshots that were already tracked remain in
  place for continuity
- new ephemeral output should land in `testjson/`

That split is necessary because `.gitignore` ignores `testjson/`.

## Wording Audit

The repo and adjacent documents had started to drift into `proof lane` language
even though the technical state is still earlier than that.

The tighter and more accurate framing is:

- `research lane`
- `descent lane`
- `Elkies-anchored search scaffold`
- `reference/signature verification lane`

Avoid:

- `solved M23`
- `completed proof`
- `final proof lane`

Use instead:

- `M23 research / descent lane`
- `explicit Elkies construction plus live search scaffold`
- `not yet proven; still in candidate-specialization and descent-screen stage`

## Current Recommended Read

The clean read of this repo is:

1. The explicit Elkies construction is real and correctly encoded.
2. The repo is useful as a live research / descent environment around that
   anchor.
3. The exact verification path is valid as a reference check, not as full proof.
4. The candidate-specialization hook is still unfinished.
5. Current stored screening evidence does not justify a solved claim.

## Practical Next Steps

Priority order:

1. keep `proof` claims out of public summaries unless tied to a much narrower
   statement such as `proof-oriented search`
2. finish the actual `lambda / mu` or equivalent candidate specialization rule
   if that lane is still intended to matter
3. keep the exact Elkies verifier as a reference oracle for per-prime behavior
4. continue descent/Tschirnhaus screening as a research path, not a solved path
5. preserve this file as the current state anchor so later edits do not drift
   back into overclaim
6. preserve the continuity roadmap so canonical-vs-historical repo confusion
   does not re-enter later threads

## Approved Working Statement

The clean working statement for the repo is:

```text
M23_Proof is an Elkies-anchored research / descent repository. It contains a
real exact quartic-field construction, a live search and monitoring surface, and
reference-signature verification tooling. It does not currently support a final
or solved M23 claim because the candidate-specialization hook remains incomplete
and the stored screening results are still weak.
```

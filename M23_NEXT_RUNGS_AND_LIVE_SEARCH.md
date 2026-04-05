# M23 Next Rungs And Live Search

Date: 2026-04-05

## Purpose

This file exists so the live search lane has a visible progression ladder.

The repo should not require chat memory to answer:

- what the live search is doing
- why a batch stops
- what happens after a batch
- what the next rung is

## What Problem The Live Search Solves

The live search is not trying to prove `M23` directly in one jump.

It solves a narrower but necessary problem first:

- search for affine transforms `x -> a*x + b` of the Elkies anchor
- rank those transforms by how much they reduce non-rational leakage
- prefer lower coefficient height and lower denominator pressure
- promote the strongest transforms into modular / cycle-aware follow-up

That matters because the exact and distributional verifier lanes are more
expensive and more meaningful when they are focused on cleaner transforms
instead of blind algebraic noise.

So the live lane is a triage / preparation engine:

- stage 1 = lower the algebraic mess
- stage 2 = check the promoted transforms against modular signatures
- stage 3 = only deepen work on the survivors

## Why The Batch Stops

`Run_Live_Search.command` is intentionally a one-batch runner.

That means:

- it launches a bounded descent batch
- it waits for all workers to finish
- it writes one merged summary
- it exits cleanly

Why keep that behavior:

- each batch produces a stable artifact
- each batch is reproducible
- parameter changes between batches stay explicit
- the search does not disappear into blind infinite churn

If you want continuous motion, use:

- `Run_Live_Search_Forever.command`

That command now does more than repeat the same pass. It:

- steps through a staged affine schedule
- expands the scale / shift window between batches
- writes persistent state to `testjson/live_search_state.json`
- automatically promotes top transforms into modular follow-up after each batch
- keeps the search moving until stopped

## Current Launcher Map

Use these as the live surface:

- `Run_Live_Search.command`
  one clean descent batch
- `Run_Live_Search_Forever.command`
  staged, repeating descent batches plus automatic follow-up screening
- `Open_Live_Search_Status.command`
  terminal watcher for the live descent lane

Use these only for the old exact lane:

- `Open_Legacy_Exact_Monitor.command`
- `Open_Legacy_Exact_Qt_Monitor.command`

## What The Live Search Actually Does

Each one-shot descent batch:

1. partitions the affine transform grid across workers
2. scores each transform against the Elkies anchor by leakage, coefficient
   height, and denominator pressure
3. writes one worker result file per worker
4. merges the worker-best outputs into one summary file

Each continuous live-search cycle now adds:

5. move to the next scheduled affine window
6. promote the top transforms from the batch into modular signature follow-up
7. record follow-up cycle summaries in the live state file

Outputs:

- worker logs:
  `testjson/runtime/m23_descent_<id>.log`
- worker pids while running:
  `testjson/runtime/m23_descent_<id>.pid`
- worker results:
  `testjson/descent_search_worker*_*.json`
- merged batch summary:
  `testjson/descent_search_summary_*.json`
- descent follow-up summaries:
  `testjson/descent_followup_*.json`
- continuous-state file:
  `testjson/live_search_state.json`

## What The Next Rungs Are

The live ladder is now:

1. Run staged descent batches to locate lower-leakage transforms over expanding affine windows.
2. Promote the strongest transforms into modular follow-up automatically.
3. Use the cycle-aware verifier layer to mark which signatures are real M23 cycle types.
4. Use the fixed-prime sampler on the strong primes to build many-`t0` distributions.
5. Build the Table-2 style aggregator for the `N_k` counts, especially `N_5`.
6. Use that distribution lane to push toward the non-`A23` exclusion step.
7. Only then revisit the old exact specialization lane if it can be made real.

## Operating Rule

Do not treat all movement as equal.

Good movement:

- lower descent scores
- lower leakage
- lower denominator pressure
- stronger modular survivors
- fixed-prime distributions that stay inside known M23 cycle structures

Weak movement:

- more files without better scores
- old exact-loop churn
- monitor activity without improved summaries

## Default Practical Rhythm

If operating solo, use this rhythm:

1. start `Run_Live_Search_Forever.command`
2. open `Open_Live_Search_Status.command`
3. after a few completed batches, inspect the latest `descent_search_summary_*.json`
4. inspect the paired `descent_followup_*.json` to see whether any promoted transforms rationalize or hit M23 cycle signatures
5. run fixed-prime sampling when a transform or prime is worth deeper follow-up

## Immediate Next Technical Target

The strongest next implementation target is:

- a Table-2 style aggregator over fixed-prime samples that computes the
  `N_k(t0)` totals and compares them against the M23 expectation lane, with
  explicit preparation for the `k = 5` contradiction step used by Elkies

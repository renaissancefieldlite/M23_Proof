# Roadmap

This roadmap describes public engineering work that can improve the project
without treating a search score as proof.

## Reproducible environment

* define supported Python and dependency versions
* add one command for installing core and optional tools
* verify setup on a clean Linux and macOS environment
* document the optional SageMath boundary

## Test architecture

* add unit tests for quartic field construction and coefficient reduction
* add fixture tests for cycle signature annotation
* add regression tests for JSON schemas and deterministic summaries
* add negative fixtures that must remain classified as inconclusive

## Command line consistency

* give the main search and verifier tools consistent argument names
* add machine readable status and error output
* distinguish bounded search commands from continuous runners
* make output directories explicit and portable

## Verification ladder

* separate heuristic candidate ranking from exact verification
* record factor degree evidence with reproducible prime sets
* improve fixed prime distribution diagnostics
* document every remaining `M23` versus `A23` proof obligation

## Maintenance

* replace absolute machine paths with repository relative paths
* consolidate duplicated loaders and serializers
* add continuous integration for syntax, unit, and fixture checks
* publish a contributor guide with small reviewable work items

These stages provide substantially more than twenty hours of normal software
engineering work while preserving the mathematical evidence boundary.

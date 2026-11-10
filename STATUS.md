# Public status

## What is established

* The repository contains an executable representation of the explicit Elkies
  quartic field anchor.
* Candidate generation and exact coefficient screens run as separate stages.
* Modular factor degree signatures can be annotated against known `M23` cycle
  structures.
* Fixed prime sampling can produce distribution summaries for later analysis.
* Search workers can be partitioned so they cover disjoint candidate regions.

## What is not established

* No candidate in this repository currently closes a new rational `M23`
  construction.
* A favorable modular signature is a search signal, not a proof of the Galois
  group.
* The current fixed prime summaries do not complete the required statistical or
  group theoretic exclusion of `A23`.
* Legacy candidate files and heuristic scores must not be described as a solved
  result.

## Acceptance boundary

A future proof claim would require an explicit candidate, reproducible exact
specialization, irreducibility and transitivity evidence, a complete Galois
verification argument, and independent reproduction. Until those gates are
met, this project remains proof search software.

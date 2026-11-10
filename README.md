# M23 Proof Search

This repository contains public computational tooling for exploring degree 23
polynomials whose Galois group may be the Mathieu group `M23`. The work starts
from the explicit quartic field construction described by Noam D. Elkies and
builds reproducible search, screening, and verification stages around it.

## Status

This is an active proof search, not a completed proof.

The repository currently provides:

* an exact representation of the Elkies quartic field anchor
* affine and Tschirnhaus candidate generation
* exact leakage, height, and denominator screens
* modular factor degree and cycle signature analysis
* fixed prime sampling and distribution summaries
* bounded batch runners and local monitoring tools

The current evidence does not establish a new degree 23 polynomial over the
rationals with Galois group `M23`, and it does not complete the required
`M23` versus `A23` exclusion. See [STATUS.md](STATUS.md) for the precise
boundary.

## Search flow

```text
Elkies anchor
  -> candidate generation
  -> exact coefficient screening
  -> modular factor signatures
  -> cycle aware verification
  -> fixed prime distribution tests
```

## Setup

Create an isolated Python environment and install the core dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The optional Qt monitor uses the dependency in `requirements-gui.txt`. Some
exact verification paths can also call SageMath when it is available.

## Inspect the command surfaces

```bash
python3 m23_canonical_pipeline.py --help
python3 m23_cycle_verifier.py --help
python3 m23_fixed_prime_sampler.py --help
```

On macOS, the `.command` launchers provide bounded search and monitor entry
points. Generated runtime data is written under `testjson/` and is ignored by
git.

## Development direction

The next engineering stages are listed in [ROADMAP.md](ROADMAP.md). They focus
on reproducible setup, regression tests, consistent command line interfaces,
fixture based verification, and clearer separation between candidate hints and
proof obligations.

## References and third party material

The mathematical starting point is Noam D. Elkies, *The complex polynomials
P(x) with Gal(P(x) - t) isomorphic to M23*. The paper itself is not included in
the current source tree and is not licensed by this project. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

The source files in the current tree are available under the Mozilla Public
License 2.0 unless a file states otherwise. See [LICENSE](LICENSE).

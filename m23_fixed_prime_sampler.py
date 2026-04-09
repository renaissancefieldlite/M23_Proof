#!/usr/bin/env python3
"""
m23_fixed_prime_sampler.py - Sample many t0 values over one fixed degree-1 prime.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from m23_cycle_signatures import annotate_prime_entry, summarize_cycle_entries, summarize_n5_entries

JSON_DIR = Path("testjson")
SAGE_BIN = os.environ.get("SAGE_BIN", "sage")


def load_survivor_entry(path: Path, survivor_index: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    survivors = payload.get("survivors", [])
    if not isinstance(survivors, list) or not survivors:
        raise ValueError(f"no survivors found in {path}")
    if survivor_index < 0 or survivor_index >= len(survivors):
        raise IndexError(
            f"survivor index {survivor_index} out of range for {path} "
            f"(count={len(survivors)})"
        )

    survivor = survivors[survivor_index]
    coeffs = survivor.get("integer_coefficients", [])
    if not isinstance(coeffs, list) or not coeffs:
        raise ValueError(
            f"survivor {survivor_index} in {path} does not include integer_coefficients"
        )

    try:
        integer_coeffs = [int(value) for value in coeffs]
    except Exception as exc:
        raise ValueError(
            f"survivor {survivor_index} in {path} has non-integer coefficients"
        ) from exc

    return {
        "source_json": str(path),
        "survivor_index": int(survivor_index),
        "candidate": survivor.get("candidate", {}),
        "descent_score": survivor.get("descent_score"),
        "integer_height": survivor.get("integer_height"),
        "denominator_lcm": survivor.get("denominator_lcm"),
        "signature_hit_count": survivor.get("signature_hit_count", 0),
        "irreducible_prime_count": survivor.get("irreducible_prime_count", 0),
        "screened_primes": survivor.get("screened_primes", []),
        "integer_coefficients": integer_coeffs,
    }


def build_sampler_script(
    prime: int,
    sample_count: int,
    start_t0: int,
    step: int,
    survivor_entry: dict | None = None,
) -> str:
    if survivor_entry is not None:
        integer_coeffs = survivor_entry["integer_coefficients"]
        survivor_metadata = {
            key: value
            for key, value in survivor_entry.items()
            if key != "integer_coefficients"
        }
        coeff_literal = json.dumps(json.dumps(integer_coeffs))
        metadata_literal = json.dumps(json.dumps(survivor_metadata))
        return f"""# Auto-generated fixed-prime sampler
import json
import sys


def factor_degree_signature(Q):
    degrees = []
    for factor, multiplicity in Q.factor():
        degree = int(factor.degree())
        for _ in range(int(multiplicity)):
            degrees.append(degree)
    return sorted(degrees)


def sample():
    p = {int(prime)}
    target = {int(sample_count)}
    start_t0 = {int(start_t0)}
    step = {int(step)}
    coeffs = json.loads({coeff_literal})
    survivor_metadata = json.loads({metadata_literal})

    if not coeffs:
        raise RuntimeError("empty survivor coefficient list")

    degree = len(coeffs) - 1
    F = GF(p)
    Rp.<y> = PolynomialRing(F)
    P_red = sum(F(coeffs[index]) * y**(degree - index) for index in range(len(coeffs)))

    samples = []
    skipped = 0
    t0 = start_t0
    visited = 0
    max_visits = max(target * 20, target + 100)
    while len(samples) < target and visited < max_visits:
        visited += 1
        field_t0 = F(t0 % p)
        Q = P_red - field_t0
        if not Q.is_squarefree():
            skipped += 1
            t0 += step
            continue
        samples.append({{
            "p": int(p),
            "t0": int(t0 % p),
            "factor_degrees": [int(d) for d in factor_degree_signature(Q)],
            "irreducible": bool(Q.is_irreducible()),
        }})
        t0 += step

    payload = {{
        "construction": "lift_survivor",
        "prime": int(p),
        "g_mod_prime": None,
        "requested_sample_count": int(target),
        "collected_sample_count": int(len(samples)),
        "visited_t0_count": int(visited),
        "skipped_repeated_root_count": int(skipped),
        "sample_mode": "sequential",
        "start_t0": int(start_t0),
        "step": int(step),
        "polynomial_degree": int(degree),
        "leading_coefficient": int(coeffs[0]),
        "survivor": survivor_metadata,
        "samples": samples,
    }}
    print(json.dumps(payload, indent=2))
    return int(0)


if __name__ == "__main__":
    sys.exit(sample())
"""

    return f"""# Auto-generated fixed-prime sampler
import json
import sys

def residue_field_mapper(prime_ideal, coeff_sample):
    rf = prime_ideal.residue_field(names='a')
    if isinstance(rf, tuple):
        k = rf[0]
        maps = [obj for obj in rf[1:] if callable(obj)]
    else:
        k = rf
        maps = []

    for mapper in maps:
        try:
            mapper(coeff_sample)
            return k, mapper
        except Exception:
            pass

    def fallback(c):
        reduced = prime_ideal.reduce(c)
        try:
            return k(reduced)
        except Exception:
            return k(c)

    return k, fallback


def choose_degree_one_prime(K, p):
    prime_ideals = K.primes_above(p)
    if not prime_ideals:
        raise RuntimeError("no primes above p")
    for pid in prime_ideals:
        try:
            if pid.residue_class_degree() == 1:
                return pid
        except Exception:
            pass
    raise RuntimeError("no degree-1 prime above p")


def reduce_polynomial_mod_prime(P_int, K, x, p):
    best = choose_degree_one_prime(K, p)
    coeffs = P_int.coefficients(sparse=False)
    if not coeffs:
        raise RuntimeError("empty coefficient list")

    k, mapper = residue_field_mapper(best, coeffs[0])
    degree = P_int.degree(x)
    while len(coeffs) < degree + 1:
        coeffs.append(K(0))

    reduced = [mapper(c) for c in coeffs]
    Rp.<y> = PolynomialRing(k)
    P_red = sum(reduced[i] * y**i for i in range(len(reduced)))
    try:
        g_image = mapper(K.gen())
        g_image = int(g_image)
    except Exception:
        g_image = None
    return P_red, k, g_image


def factor_degree_signature(Q):
    degrees = []
    for factor, multiplicity in Q.factor():
        degree = int(factor.degree())
        for _ in range(int(multiplicity)):
            degrees.append(degree)
    return sorted(degrees)


def sample():
    p = {int(prime)}
    target = {int(sample_count)}
    start_t0 = {int(start_t0)}
    step = {int(step)}

    R.<g> = QQ[]
    K.<g> = NumberField(g^4 + g^3 + 9*g^2 - 10*g + 8)
    S.<x> = K[]

    P2 = (8*g**3 + 16*g**2 - 20*g + 20)*x**2 + (-7*g**3 - 17*g**2 + 7*g - 76)*x + (-13*g**3 + 25*g**2 - 107*g + 596)
    P3 = 8*(31*g**3 + 405*g**2 - 459*g + 333)*x**3 + (941*g**3 + 1303*g**2 - 1853*g + 1772)*x + (85*g**3 - 385*g**2 + 395*g - 220)
    P4 = 32*(4*g**3 - 69*g**2 + 74*g - 49)*x**4 + 32*(21*g**3 + 53*g**2 - 68*g + 58)*x**3 - 8*(97*g**3 + 95*g**2 - 145*g + 148)*x**2 + 8*(41*g**3 - 89*g**2 - g + 140)*x + (-123*g**3 + 391*g**2 - 93*g + 3228)
    tau = (2**38 * 3**17 / 23**3) * (47323*g**3 - 1084897*g**2 + 7751*g - 711002)
    P = P2**2 * P3 * P4**4 + tau
    P_int = P * P.denominator()
    P_int = P_int * P_int.denominator()

    P_red, k, g_image = reduce_polynomial_mod_prime(P_int, K, x, p)

    samples = []
    skipped = 0
    t0 = start_t0
    visited = 0
    max_visits = max(target * 20, target + 100)
    while len(samples) < target and visited < max_visits:
        visited += 1
        field_t0 = k(t0 % p)
        Q = P_red - field_t0
        if not Q.is_squarefree():
            skipped += 1
            t0 += step
            continue
        samples.append({{
            "p": int(p),
            "t0": int(t0 % p),
            "factor_degrees": [int(d) for d in factor_degree_signature(Q)],
            "irreducible": bool(Q.is_irreducible()),
        }})
        t0 += step

    payload = {{
        "construction": "elkies_explicit",
        "prime": int(p),
        "g_mod_prime": g_image,
        "requested_sample_count": int(target),
        "collected_sample_count": int(len(samples)),
        "visited_t0_count": int(visited),
        "skipped_repeated_root_count": int(skipped),
        "sample_mode": "sequential",
        "start_t0": int(start_t0),
        "step": int(step),
        "samples": samples,
    }}
    print(json.dumps(payload, indent=2))
    return int(0)


if __name__ == "__main__":
    sys.exit(sample())
"""


def run_sampler(
    prime: int,
    sample_count: int,
    start_t0: int,
    step: int,
    timeout: int,
    survivor_entry: dict | None = None,
) -> dict:
    JSON_DIR.mkdir(exist_ok=True)
    sage_home = (JSON_DIR / ".sage_home_sampler").resolve()
    dot_sage = sage_home / ".sage"
    sage_home.mkdir(exist_ok=True)
    dot_sage.mkdir(exist_ok=True)
    script = build_sampler_script(prime, sample_count, start_t0, step, survivor_entry=survivor_entry)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sage", delete=False, encoding="utf-8") as handle:
        handle.write(script)
        script_path = handle.name

    started = time.time()
    try:
        result = subprocess.run(
            [SAGE_BIN, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                "HOME": str(sage_home),
                "DOT_SAGE": str(dot_sage),
            },
        )
        elapsed = time.time() - started
        payload = {
            "success": result.returncode == 0,
            "elapsed": elapsed,
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-2000:],
            "returncode": result.returncode,
        }
        try:
            parsed = json.loads(result.stdout)
        except Exception:
            parsed = None
        if parsed is not None:
            samples = [annotate_prime_entry(item) for item in parsed.get("samples", [])]
            parsed["samples"] = samples
            parsed["cycle_summary"] = summarize_cycle_entries(samples)
            parsed.update(summarize_n5_entries(samples))
            payload["result"] = parsed
        return payload
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=100000007)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--start-t0", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--survivor-json",
        type=Path,
        default=None,
        help="optional lift_survivors JSON; when set, sample that survivor polynomial instead of the Elkies anchor",
    )
    parser.add_argument(
        "--survivor-index",
        type=int,
        default=0,
        help="0-based survivor index used with --survivor-json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional output path; defaults to testjson/m23_fixed_prime_sample_P_TIMESTAMP.json",
    )
    args = parser.parse_args()

    survivor_entry = None
    if args.survivor_json is not None:
        survivor_entry = load_survivor_entry(args.survivor_json, args.survivor_index)

    payload = run_sampler(
        prime=args.prime,
        sample_count=args.sample_count,
        start_t0=args.start_t0,
        step=args.step,
        timeout=args.timeout,
        survivor_entry=survivor_entry,
    )
    if args.output is not None:
        output_path = args.output
    elif survivor_entry is not None:
        output_path = JSON_DIR / (
            f"m23_fixed_prime_sample_survivor{args.survivor_index}_{args.prime}_"
            f"{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
    else:
        output_path = JSON_DIR / (
            f"m23_fixed_prime_sample_{args.prime}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved fixed-prime sample to {output_path}")
    if "result" in payload:
        result = payload["result"]
        print(
            "Sampler source:",
            result.get("construction", "unknown"),
            f"prime={result.get('prime')}",
            f"collected={result.get('collected_sample_count', 0)}",
        )
        if result.get("construction") == "lift_survivor":
            survivor = result.get("survivor", {})
            print(
                "Survivor source:",
                f"index={survivor.get('survivor_index')}",
                f"height={survivor.get('integer_height')}",
                f"signature_hits={survivor.get('signature_hit_count', 0)}",
            )
        summary = result.get("cycle_summary", {})
        print(
            "Cycle summary:",
            f"tested={summary.get('tested_prime_count', 0)}",
            f"matched_m23={summary.get('matched_m23_prime_count', 0)}",
            f"cycle_rate={summary.get('exact_m23_cycle_rate', 0.0):.3f}",
            f"status={summary.get('a23_exclusion_status', 'not_ready')}",
        )
        print(
            "Full-cycle metrics:",
            f"support_compatible={summary.get('full_cycle_support_compatible', False)}",
            f"logL={summary.get('full_cycle_log_likelihood_m23')}",
            f"KL={summary.get('full_cycle_kl_divergence_m23')}",
            f"unknown_signatures={summary.get('unknown_signature_count', 0)}",
        )
        print("Full-cycle expected vs observed (M23):")
        for row in summary.get("distribution_rows", []):
            print(
                " ",
                f"{row.get('atlas_label')}:",
                f"obs={row.get('observed_count', 0)}",
                f"obs_freq={row.get('observed_frequency', 0.0):.4f}",
                f"exp={row.get('expected_count_m23', 0.0):.3f}",
                f"exp_freq={row.get('expected_frequency_m23', 0.0):.4f}",
                f"delta={row.get('delta', 0.0):+.3f}",
                f"cycle={row.get('cycle_notation')}",
            )
        n5_counts = result.get("n5_observed_counts", {})
        n5_expected = result.get("n5_expected_distribution_m23", {})
        print(
            "N5 summary:",
            f"counts=0:{n5_counts.get('0', 0)},1:{n5_counts.get('1', 0)},4:{n5_counts.get('4', 0)}",
            f"expected=0:{n5_expected.get('0', 0.0):.4f},"
            f"1:{n5_expected.get('1', 0.0):.4f},4:{n5_expected.get('4', 0.0):.4f}",
            f"support_compatible={result.get('n5_support_compatible', False)}",
            f"logL={result.get('n5_log_likelihood_m23')}",
            f"KL={result.get('n5_kl_divergence_m23')}",
        )
    else:
        print("Sampler output was not parseable JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

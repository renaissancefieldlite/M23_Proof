"""
phase3_exact.py
M23 Inverse Galois Attack - Phase 3 (EXACT ALGEBRAIC VERSION)
Testing exact candidates via Sage (real factorization)
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
import time

JSON_DIR = "testjson"
CANDIDATE_FILE = os.path.join(JSON_DIR, "exact_candidates.json")
PARTIAL_FILE = os.path.join(JSON_DIR, "partial.json")
SAGE_BIN = os.environ.get("SAGE_BIN", "sage")
DEFAULT_CANDIDATES_PER_RUN = int(os.environ.get("CANDIDATES_PER_RUN", "12"))

os.makedirs(JSON_DIR, exist_ok=True)


def load_exact_candidates():
    try:
        with open(CANDIDATE_FILE, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        print(f"Loaded {len(candidates)} exact candidates from {CANDIDATE_FILE}")
        return candidates
    except FileNotFoundError:
        print("No exact candidates found. Run phase2_exact.py first.")
        return []


def parse_worker_config():
    instance_id = os.environ.get("INSTANCE_ID", "1")
    try:
        worker_count = max(1, int(os.environ.get("WORKER_COUNT", "1")))
    except ValueError:
        worker_count = 1

    try:
        worker_index = max(1, int(instance_id))
    except ValueError:
        worker_index = 1

    partition_mode = os.environ.get("M23_PARTITION_MODE", "chunk").strip().lower() or "chunk"
    return instance_id, worker_index, worker_count, partition_mode


def select_candidates_for_instance(candidates, instance_id, max_candidates):
    indexed = list(enumerate(candidates))
    _, worker_index, worker_count, partition_mode = parse_worker_config()

    if not indexed:
        return []

    if partition_mode == "stride":
        selected = indexed[worker_index - 1 :: worker_count]
        print(
            f"Worker {instance_id}/{worker_count} using stride partition "
            f"(offset {worker_index - 1}, span {worker_count})"
        )
    else:
        start = math.floor((worker_index - 1) * len(indexed) / worker_count)
        end = math.floor(worker_index * len(indexed) / worker_count)
        selected = indexed[start:end]
        print(
            f"Worker {instance_id}/{worker_count} scanning chunk "
            f"[{start}:{end}] out of {len(indexed)} candidates"
        )

    if max_candidates > 0:
        selected = selected[:max_candidates]

    print(f"Worker {instance_id} selected {len(selected)} candidates for this pass")
    return selected


def generate_sage_script(candidate, index):
    λr = candidate["λ_real"]
    λi = candidate["λ_imag"]
    μr = candidate["μ_real"]
    μi = candidate["μ_imag"]

    script = f'''# Sage script for M23 candidate {index}
import sys

def build_base_polynomial():
    R.<g> = QQ[]
    K.<g> = NumberField(g^4 + g^3 + 9*g^2 - 10*g + 8)
    S.<x> = K[]

    P2 = (8*g**3 + 16*g**2 - 20*g + 20)*x**2 + (-7*g**3 - 17*g**2 + 7*g - 76)*x + (-13*g**3 + 25*g**2 - 107*g + 596)
    P3 = 8*(31*g**3 + 405*g**2 - 459*g + 333)*x**3 + (941*g**3 + 1303*g**2 - 1853*g + 1772)*x + (85*g**3 - 385*g**2 + 395*g - 220)
    P4 = 32*(4*g**3 - 69*g**2 + 74*g - 49)*x**4 + 32*(21*g**3 + 53*g**2 - 68*g + 58)*x**3 - 8*(97*g**3 + 95*g**2 - 145*g + 148)*x**2 + 8*(41*g**3 - 89*g**2 - g + 140)*x + (-123*g**3 + 391*g**2 - 93*g + 3228)
    tau_num = 2**38 * 3**17 * (47323*g**3 - 1084897*g**2 + 7751*g - 711002)
    tau = tau_num / 23**3
    P = P2**2 * P3 * P4**4 + tau
    return K, x, P

def apply_candidate_specialization(P, λr, λi, μr, μi):
    # TODO:
    # The uploaded source files do not contain the actual λ/μ insertion rule.
    # This hook is intentionally left as identity so the pipeline still runs,
    # but candidate values are NOT yet changing the polynomial.
    return P, False

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

def reduce_polynomial_mod_prime(P_int, K, x, p):
    prime_ideals = K.primes_above(p)
    if not prime_ideals:
        raise RuntimeError("no primes above p")

    best = prime_ideals[0]
    for pid in prime_ideals:
        try:
            if pid.residue_class_degree() == 1:
                best = pid
                break
        except Exception:
            pass

    coeffs = P_int.coefficients(sparse=False)
    if not coeffs:
        raise RuntimeError("empty coefficient list")

    k, mapper = residue_field_mapper(best, coeffs[0])
    degree = P_int.degree(x)
    while len(coeffs) < degree + 1:
        coeffs.append(K(0))

    red = [mapper(c) for c in coeffs]

    Rp.<y> = PolynomialRing(k)
    P_red = sum(red[i] * y**i for i in range(len(red)))
    return P_red

def test_candidate():
    try:
        print("-" * 50)
        print(f"Testing candidate {index}")
        print("-" * 50)
        print(f"λ = {λr} + {λi}*I")
        print(f"μ = {μr} + {μi}*I")

        K, x, P = build_base_polynomial()
        P, candidate_applied = apply_candidate_specialization(P, "{λr}", "{λi}", "{μr}", "{μi}")

        if not candidate_applied:
            print("WARNING: candidate specialization hook is still identity")

        print("Degree of P:", P.degree(x))
        print("Number of terms:", len(P.coefficients()))

        P_int = P * P.denominator()
        P_int = P_int * P_int.denominator()

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        irreducible_count = 0
        tested_count = 0

        print("\\nFactorization mod primes:")
        for p in primes:
            try:
                P_red = reduce_polynomial_mod_prime(P_int, K, x, p)

                if P_red.is_irreducible():
                    irreducible_count += 1
                    print("p =", p, ": irreducible")
                else:
                    print("p =", p, ": factors")
                tested_count += 1
            except Exception as e:
                print("p =", p, ": (skipping -", str(e)[:120], ")")
                continue

        if tested_count > 0:
            score = irreducible_count / tested_count
            print("\\nTested", tested_count, "primes")
            print("Irreducible count:", irreducible_count, "/", tested_count)
            print("Consistency score:", score)
        else:
            print("\\nNo primes successfully tested")
            return 2

        return 0
    except Exception as e:
        print("Error:", e)
        return 1

if __name__ == "__main__":
    sys.exit(test_candidate())
'''
    return script


def test_candidate_with_sage(candidate, index, timeout=300):
    script = generate_sage_script(candidate, index)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sage", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_file = f.name

    print(f"Running Sage with {SAGE_BIN} (timeout: {timeout}s)...")
    start = time.time()

    try:
        if shutil.which(SAGE_BIN) is None and os.path.sep not in SAGE_BIN:
            return {
                "success": False,
                "irreducible_count": 0,
                "tested_count": 0,
                "consistency_score": 0.0,
                "output": "",
                "error": f"Sage executable not found: {SAGE_BIN}",
                "elapsed": time.time() - start,
            }

        result = subprocess.run(
            [SAGE_BIN, script_file],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start

        tested = 0
        irred = 0
        candidate_applied = True

        for line in result.stdout.splitlines():
            if "WARNING: candidate specialization hook is still identity" in line:
                candidate_applied = False
            if "Irreducible count:" in line:
                try:
                    rhs = line.split(":", 1)[1].strip()
                    left_val, right_val = [part.strip() for part in rhs.split("/")]
                    irred = int(left_val)
                    tested = int(right_val)
                except Exception:
                    pass

        score = irred / tested if tested > 0 else 0.0
        success = (result.returncode == 0) and (tested > 0)

        return {
            "success": success,
            "irreducible_count": irred,
            "tested_count": tested,
            "consistency_score": score,
            "candidate_applied": candidate_applied,
            "output": result.stdout[-4000:],
            "error": result.stderr[-1000:],
            "elapsed": elapsed,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "irreducible_count": 0,
            "tested_count": 0,
            "consistency_score": 0.0,
            "candidate_applied": False,
            "output": "",
            "error": f"Timeout after {timeout}s",
            "elapsed": time.time() - start,
            "returncode": -1,
        }
    finally:
        try:
            os.unlink(script_file)
        except Exception:
            pass


def main():
    print("=" * 70)
    print("M23 Inverse Galois Attack - Phase 3 (EXACT ALGEBRAIC)")
    print("=" * 70)

    instance_id, _, worker_count, partition_mode = parse_worker_config()
    max_candidates = DEFAULT_CANDIDATES_PER_RUN
    print(f"Partition mode: {partition_mode}")
    print(f"Configured workers: {worker_count}")

    candidates = load_exact_candidates()
    if not candidates:
        return

    selected = select_candidates_for_instance(candidates, instance_id, max_candidates)
    if not selected:
        print("No candidates selected for this instance.")
        return

    results = []
    total = len(selected)

    for position, (candidate_index, candidate) in enumerate(selected, start=1):
        print(f"\n{'=' * 60}")
        print(
            f"Instance {instance_id} testing candidate {position}/{total} "
            f"(global index {candidate_index})"
        )
        print(f"{'=' * 60}")
        print(f"lambda = {candidate['λ_expr']}")
        print(f"mu = {candidate['μ_expr']}")

        result = test_candidate_with_sage(candidate, candidate_index)

        if result["success"]:
            print(f"Completed in {result['elapsed']:.1f}s")
            print(f"Tested {result['tested_count']} primes")
            print(
                f"Irreducible: {result['irreducible_count']}/{result['tested_count']} "
                f"({result['consistency_score']:.3f})"
            )
        else:
            print(f"Failed: {result['error'][:160]}")

        results.append(
            {
                "instance_id": instance_id,
                "candidate_index": candidate_index,
                "candidate": candidate,
                "result": result,
            }
        )

        with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(JSON_DIR, f"exact_test_results_{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {filename}")


if __name__ == "__main__":
    main()

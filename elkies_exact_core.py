#!/usr/bin/env python3
"""
elkies_exact_core.py - Canonical exact construction from Elkies' M23 paper.

This module collects the explicit quartic-field construction from:
    Noam D. Elkies, "The complex polynomials P(x) with Gal(P(x) - t) ~= M23"

It does not implement a lambda/mu specialization family. It exposes the
paper-based fixed construction so other scripts can build, verify, or reduce the
same polynomial without duplicating formulas across the repo.
"""

from __future__ import annotations

from dataclasses import dataclass

from sympy import Expr, Symbol, expand, symbols


@dataclass(frozen=True)
class ElkiesConstruction:
    g: Symbol
    x: Symbol
    field_modulus: Expr
    p2: Expr
    p3: Expr
    p4: Expr
    tau: Expr
    polynomial: Expr


def build_elkies_construction(
    x_symbol: Symbol | None = None,
    g_symbol: Symbol | None = None,
) -> ElkiesConstruction:
    """
    Build the explicit M23 polynomial construction from Elkies' paper.

    Returns symbolic expressions over the quartic field
        Q[g] / (g^4 + g^3 + 9*g^2 - 10*g + 8).
    """

    x = x_symbol or symbols("x")
    g = g_symbol or symbols("g")
    field_modulus = g**4 + g**3 + 9 * g**2 - 10 * g + 8

    # Elkies, page 5, equations (5)-(8)
    p2 = (
        (8 * g**3 + 16 * g**2 - 20 * g + 20) * x**2
        + (-7 * g**3 - 17 * g**2 + 7 * g - 76) * x
        + (-13 * g**3 + 25 * g**2 - 107 * g + 596)
    )
    p3 = (
        8 * (31 * g**3 + 405 * g**2 - 459 * g + 333) * x**3
        + (941 * g**3 + 1303 * g**2 - 1853 * g + 1772) * x
        + (85 * g**3 - 385 * g**2 + 395 * g - 220)
    )
    p4 = (
        32 * (4 * g**3 - 69 * g**2 + 74 * g - 49) * x**4
        + 32 * (21 * g**3 + 53 * g**2 - 68 * g + 58) * x**3
        - 8 * (97 * g**3 + 95 * g**2 - 145 * g + 148) * x**2
        + 8 * (41 * g**3 - 89 * g**2 - g + 140) * x
        + (-123 * g**3 + 391 * g**2 - 93 * g + 3228)
    )
    tau = (2**38 * 3**17 / 23**3) * (47323 * g**3 - 1084897 * g**2 + 7751 * g - 711002)
    polynomial = expand(p2**2 * p3 * p4**4 + tau)

    return ElkiesConstruction(
        g=g,
        x=x,
        field_modulus=field_modulus,
        p2=p2,
        p3=p3,
        p4=p4,
        tau=tau,
        polynomial=polynomial,
    )


def build_sage_verification_script(primes: list[int] | None = None) -> str:
    """
    Return a Sage script that builds the Elkies polynomial and checks
    irreducibility patterns modulo the provided primes.
    """

    prime_list = primes or [2, 3, 5, 7, 11, 13, 17, 19, 23]
    prime_literal = ", ".join(str(p) for p in prime_list)
    return f"""# Auto-generated from elkies_exact_core.py
import json
import sys
import time

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

    reduced = [mapper(c) for c in coeffs]
    Rp.<y> = PolynomialRing(k)
    return sum(reduced[i] * y**i for i in range(len(reduced)))


def main():
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

    primes = [{prime_literal}]
    irreducible_count = 0
    tested_count = 0
    per_prime = []

    for p in primes:
        try:
            P_red = reduce_polynomial_mod_prime(P_int, K, x, p)
            irreducible = bool(P_red.is_irreducible())
            if irreducible:
                irreducible_count += 1
            tested_count += 1
            per_prime.append({{
                "p": p,
                "irreducible": irreducible,
                "degree": int(P_red.degree()),
            }})
        except Exception as exc:
            per_prime.append({{
                "p": p,
                "irreducible": False,
                "error": str(exc),
            }})

    score = (irreducible_count / tested_count) if tested_count else 0.0
    payload = {{
        "construction": "elkies_explicit",
        "tested_count": tested_count,
        "irreducible_count": irreducible_count,
        "consistency_score": score,
        "per_prime": per_prime,
    }}
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


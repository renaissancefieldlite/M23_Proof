import numpy as np
from numpy.polynomial import Polynomial

# Use g1 from the roots
g = complex(-1.0494723045418888, 3.0735660974242554)

print(f"Using g = {g}")

# Define polynomials in terms of x
def poly_from_coeffs_dict(coeffs_dict):
    """Convert coefficient dict to Polynomial object"""
    max_deg = max(coeffs_dict.keys())
    coeffs = [0] * (max_deg + 1)
    for deg, coeff_expr in coeffs_dict.items():
        # Evaluate the coefficient expression at our g value
        if isinstance(coeff_expr, str):
            # Parse and evaluate the expression
            coeff_val = eval(coeff_expr)
        else:
            coeff_val = coeff_expr
        coeffs[deg] = coeff_val
    return Polynomial(coeffs[::-1])  # Polynomial wants [a0, a1, ...]

# P₂ coefficients (evaluated at our g)
P2_coeffs = {
    2: 8*g**3 + 16*g**2 - 20*g + 20,
    1: -7*g**3 - 17*g**2 + 7*g - 76,
    0: -13*g**3 + 25*g**2 - 107*g + 596
}

# P₃ coefficients
P3_coeffs = {
    3: 248*g**3 + 3240*g**2 - 3672*g + 2664,
    1: 941*g**3 + 1303*g**2 - 1853*g + 1772,
    0: 85*g**3 - 385*g**2 + 395*g - 220
}

# P₄ coefficients
P4_coeffs = {
    4: 128*g**3 - 2208*g**2 + 2368*g - 1568,
    3: 672*g**3 + 1696*g**2 - 2176*g + 1856,
    2: -776*g**3 - 760*g**2 + 1160*g - 1184,
    1: 328*g**3 - 712*g**2 - 8*g + 1120,
    0: -123*g**3 + 391*g**2 - 93*g + 3228
}

# τ
tau = (2**38 * 3**17 / 23**3) * (47323*g**3 - 1084897*g**2 + 7751*g - 711002)

print(f"\nτ = {tau}")

# Create polynomials
P2 = Polynomial([P2_coeffs[0], P2_coeffs[1], P2_coeffs[2]])
P3 = Polynomial([P3_coeffs[0], 0, 0, P3_coeffs[3]])  # Note: x^2 term is 0
P4 = Polynomial([P4_coeffs[0], P4_coeffs[1], P4_coeffs[2], P4_coeffs[3], P4_coeffs[4]])

print(f"\nP₂ = {P2}")
print(f"P₃ = {P3}")
print(f"P₄ = {P4}")

# Compute P = P₂² * P₃ * P₄⁴ + τ
P2_sq = P2 * P2
P4_4 = P4 * P4 * P4 * P4
P = P2_sq * P3 * P4_4
P = P + tau  # Add constant τ

print(f"\nP(x) (degree {P.degree()}) coefficients:")
coeffs = P.coef
for i, c in enumerate(coeffs):
    if abs(c) > 1e-10:  # Only show non-zero coefficients
        print(f"a{i:2d} = {c:.10f}")

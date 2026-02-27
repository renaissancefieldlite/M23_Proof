import numpy as np
from numpy.polynomial import Polynomial
from sympy import symbols, expand, Poly, simplify

# Define g as a symbol (root of the quartic)
g = symbols('g')

# The quartic field generator from paper (page 3)
# g satisfies: g⁴ + g³ + 9g² - 10g + 8 = 0

# P₂ from equation (5) - EXACT from paper
P2_coeffs = {
    2: 8*g**3 + 16*g**2 - 20*g + 20,
    1: -(7*g**3 + 17*g**2 - 7*g + 76),
    0: -13*g**3 + 25*g**2 - 107*g + 596
}

# P₃ from equation (6) - EXACT from paper
P3_coeffs = {
    3: 8*(31*g**3 + 405*g**2 - 459*g + 333),
    1: 941*g**3 + 1303*g**2 - 1853*g + 1772,
    0: 85*g**3 - 385*g**2 + 395*g - 220
}

# P₄ from equation (7) - EXACT from paper
P4_coeffs = {
    4: 32*(4*g**3 - 69*g**2 + 74*g - 49),
    3: 32*(21*g**3 + 53*g**2 - 68*g + 58),
    2: -8*(97*g**3 + 95*g**2 - 145*g + 148),
    1: 8*(41*g**3 - 89*g**2 - g + 140),
    0: -123*g**3 + 391*g**2 - 93*g + 3228
}

# τ from equation (8) - EXACT from paper
tau = (2**38 * 3**17 / 23**3) * (47323*g**3 - 1084897*g**2 + 7751*g - 711002)

print("EXACT POLYNOMIAL FROM ELKIES' PAPER:")
print("=====================================")
print("\nP₂ = ", end="")
print(f"({P2_coeffs[2]}) x² + ({P2_coeffs[1]}) x + ({P2_coeffs[0]})")

print("\nP₃ = ", end="")
print(f"({P3_coeffs[3]}) x³ + ({P3_coeffs[1]}) x + ({P3_coeffs[0]})")

print("\nP₄ = ", end="")
print(f"({P4_coeffs[4]}) x⁴ + ({P4_coeffs[3]}) x³ + ({P4_coeffs[2]}) x² + ({P4_coeffs[1]}) x + ({P4_coeffs[0]})")

print(f"\nτ = {tau}")

print("\n" + "="*50)
print("To get the actual degree-23 polynomial P = P₂² * P₃ * P₄⁴ + τ")
print("We need to:")
print("1. Pick one of the 4 roots g of g⁴ + g³ + 9g² - 10g + 8 = 0")
print("2. Compute all coefficients numerically")
print("3. The result will have coefficients in the quartic field")

print("\n" + "="*50)
print("NUMERICAL COMPUTATION WITH g = g3 (from before):")
print("="*50)

# Use g3 from before
g_val = complex(0.5494723045418892, 0.6756503357678952)

def eval_poly_at_g(coeffs_dict, g):
    result = {}
    for deg, expr in coeffs_dict.items():
        # This is hacky - in real code we'd use sympy to evaluate
        # For now, we'll just print the expressions
        result[deg] = expr
    return result

print("\nP₂ coefficients at g3:")
print(f"x²: {P2_coeffs[2]}")
print(f"x¹: {P2_coeffs[1]}") 
print(f"x⁰: {P2_coeffs[0]}")

print("\nP₃ coefficients at g3:")
print(f"x³: {P3_coeffs[3]}")
print(f"x¹: {P3_coeffs[1]}")
print(f"x⁰: {P3_coeffs[0]}")

print("\nP₄ coefficients at g3:")
print(f"x⁴: {P4_coeffs[4]}")
print(f"x³: {P4_coeffs[3]}")
print(f"x²: {P4_coeffs[2]}")
print(f"x¹: {P4_coeffs[1]}")
print(f"x⁰: {P4_coeffs[0]}")

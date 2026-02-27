import numpy as np
from sympy import symbols, expand, Poly

# Define g as a symbol (root of the quartic)
g = symbols('g')

# The quartic field generator
# g satisfies: g⁴ + g³ + 9g² - 10g + 8 = 0

# P₂ from equation (5)
P2_coeffs = {
    2: 8*g**3 + 16*g**2 - 20*g + 20,
    1: -(7*g**3 + 17*g**2 - 7*g + 76),
    0: -13*g**3 + 25*g**2 - 107*g + 596
}

# P₃ from equation (6)
P3_coeffs = {
    3: 8*(31*g**3 + 405*g**2 - 459*g + 333),
    1: 941*g**3 + 1303*g**2 - 1853*g + 1772,
    0: 85*g**3 - 385*g**2 + 395*g - 220
}

# P₄ from equation (7)
P4_coeffs = {
    4: 32*(4*g**3 - 69*g**2 + 74*g - 49),
    3: 32*(21*g**3 + 53*g**2 - 68*g + 58),
    2: -8*(97*g**3 + 95*g**2 - 145*g + 148),
    1: 8*(41*g**3 - 89*g**2 - g + 140),
    0: -123*g**3 + 391*g**2 - 93*g + 3228
}

print("P₂ coefficients (in terms of g):")
for deg, coeff in P2_coeffs.items():
    print(f"  x^{deg}: {coeff}")

print("\nP₃ coefficients:")
for deg, coeff in P3_coeffs.items():
    print(f"  x^{deg}: {coeff}")

print("\nP₄ coefficients:")
for deg, coeff in P4_coeffs.items():
    print(f"  x^{deg}: {coeff}")

# τ from equation (8)
tau = (2**38 * 3**17 / 23**3) * (47323*g**3 - 1084897*g**2 + 7751*g - 711002)
print(f"\nτ = {tau}")

# Find numerical values of g
from numpy import roots

# Coefficients of g⁴ + g³ + 9g² - 10g + 8 = 0
coeffs = [1, 1, 9, -10, 8]
g_roots = roots(coeffs)
print(f"\nThe 4 roots of g⁴ + g³ + 9g² - 10g + 8 = 0:")
for i, root in enumerate(g_roots):
    print(f"g{i+1} = {root}")

# Use the first root to compute P = P₂² * P₃ * P₄⁴ + τ
# We'll do this numerically

from sympy import symbols, expand, Poly
import numpy as np

# Define g
g = symbols('g')

# Reduced forms from previous step
P2 = -123*g**3 + 123*g**2 - 237*g + 588
P3 = -12441*g**3 + 97293*g**2 - 101319*g + 63420
P4 = -57963*g**3 - 24009*g**2 + 53427*g - 63012

# τ (keeping as symbolic for now)
tau_coeffs = {
    3: 1.38067011955983e+20,
    2: -3.16523650381442e+21,
    1: 2.26138961957362e+19,
    0: -2.07438077963628e+21
}
tau = tau_coeffs[3]*g**3 + tau_coeffs[2]*g**2 + tau_coeffs[1]*g + tau_coeffs[0]

print("P₂ =", P2)
print("P₃ =", P3)
print("P₄ =", P4)
print("τ =", tau)

# Compute P₂²
P2_sq = expand(P2 * P2)
print("\nP₂² =", P2_sq)

# Compute P₄⁴ step by step
P4_sq = expand(P4 * P4)
P4_4 = expand(P4_sq * P4_sq)
print("P₄⁴ =", P4_4)

# Compute P₂² * P₃
P2_sq_P3 = expand(P2_sq * P3)
print("P₂²·P₃ =", P2_sq_P3)

# Compute final product
P_temp = expand(P2_sq_P3 * P4_4)
print("P₂²·P₃·P₄⁴ =", P_temp)

# Add τ
P = expand(P_temp + tau)
print("\n" + "="*50)
print("FINAL POLYNOMIAL P (as expression in g):")
print("="*50)
print(P)

# Now we need to express P as coefficients in the basis {1, g, g², g³}
# But this will be a huge expression. Let's collect terms by degree in g
from sympy import collect

P_collected = collect(P, g)
print("\n" + "="*50)
print("COLLECTED BY DEGREE IN g:")
print("="*50)
print(P_collected)

# The polynomial in x is actually P(x) but we have it as coefficients in g
# We need to expand (P₂)²·P₃·(P₄)⁴ + τ where each of P₂, P₃, P₄ are polynomials in x with coefficients in ℚ(g)
# This is getting too complex for simple script — we need proper computer algebra

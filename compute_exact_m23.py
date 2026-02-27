import numpy as np
from numpy.polynomial import Polynomial

# Use g3 from before
g = complex(0.5494723045418892, 0.6756503357678952)
print(f"Using g = {g}")

# Evaluate each coefficient numerically
def eval_expr(expr_str, g):
    # Parse and evaluate the expression
    # This is a bit hacky but works for our case
    return eval(expr_str)

# P₂ numerical values
p2_2 = 8*g**3 + 16*g**2 - 20*g + 20
p2_1 = -7*g**3 - 17*g**2 + 7*g - 76
p2_0 = -13*g**3 + 25*g**2 - 107*g + 596

print(f"\nP₂ = ({p2_2}) x² + ({p2_1}) x + ({p2_0})")

# P₃ numerical values
p3_3 = 248*g**3 + 3240*g**2 - 3672*g + 2664
p3_1 = 941*g**3 + 1303*g**2 - 1853*g + 1772
p3_0 = 85*g**3 - 385*g**2 + 395*g - 220

print(f"\nP₃ = ({p3_3}) x³ + 0 x² + ({p3_1}) x + ({p3_0})")

# P₄ numerical values
p4_4 = 128*g**3 - 2208*g**2 + 2368*g - 1568
p4_3 = 672*g**3 + 1696*g**2 - 2176*g + 1856
p4_2 = -776*g**3 - 760*g**2 + 1160*g - 1184
p4_1 = 328*g**3 - 712*g**2 - 8*g + 1120
p4_0 = -123*g**3 + 391*g**2 - 93*g + 3228

print(f"\nP₄ = ({p4_4}) x⁴ + ({p4_3}) x³ + ({p4_2}) x² + ({p4_1}) x + ({p4_0})")

# τ
tau = (2**38 * 3**17 / 23**3) * (47323*g**3 - 1084897*g**2 + 7751*g - 711002)
print(f"\nτ = {tau}")

# Create polynomials
P2 = Polynomial([p2_0, p2_1, p2_2])
P3 = Polynomial([p3_0, 0, 0, p3_3])  # Note: x² and x¹ terms except x¹ we already have
# Fix: Actually P3 has x term, so we need to set coefficient for x^1
P3 = Polynomial([p3_0, p3_1, 0, p3_3])
P4 = Polynomial([p4_0, p4_1, p4_2, p4_3, p4_4])

print("\n" + "="*50)
print("COMPUTING P = P₂² * P₃ * P₄⁴ + τ")
print("="*50)

# Compute step by step
P2_sq = P2 * P2
print(f"\nP₂² degree: {P2_sq.degree()}")

P4_4 = P4 * P4 * P4 * P4
print(f"P₄⁴ degree: {P4_4.degree()}")

P_temp = P2_sq * P3
print(f"P₂² * P₃ degree: {P_temp.degree()}")

P = P_temp * P4_4
print(f"P₂² * P₃ * P₄⁴ degree: {P.degree()}")

# Add τ (constant)
coeffs = list(P.coef)
coeffs[0] = coeffs[0] + tau
P = Polynomial(coeffs)

print(f"\nFinal P(x) degree: {P.degree()}")

print("\n" + "="*50)
print("COEFFICIENTS (should be degree 23):")
print("="*50)
coeffs = P.coef
for i, c in enumerate(coeffs):
    if abs(c) > 1e-10:  # Only show non-zero
        print(f"a{i:2d} = {c}")
        
print(f"\nNumber of coefficients: {len(coeffs)}")

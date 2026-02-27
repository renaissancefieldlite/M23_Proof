import numpy as np
from numpy.polynomial import Polynomial

# Use both g3 and g4 (complex conjugates)
g3 = complex(0.5494723045418892, 0.6756503357678952)
g4 = complex(0.5494723045418892, -0.6756503357678952)

print(f"g3 = {g3}")
print(f"g4 = {g4}")

def compute_poly(g):
    # P₂
    p2_2 = 8*g**3 + 16*g**2 - 20*g + 20
    p2_1 = -7*g**3 - 17*g**2 + 7*g - 76
    p2_0 = -13*g**3 + 25*g**2 - 107*g + 596
    
    # P₃
    p3_3 = 248*g**3 + 3240*g**2 - 3672*g + 2664
    p3_1 = 941*g**3 + 1303*g**2 - 1853*g + 1772
    p3_0 = 85*g**3 - 385*g**2 + 395*g - 220
    
    # P₄
    p4_4 = 128*g**3 - 2208*g**2 + 2368*g - 1568
    p4_3 = 672*g**3 + 1696*g**2 - 2176*g + 1856
    p4_2 = -776*g**3 - 760*g**2 + 1160*g - 1184
    p4_1 = 328*g**3 - 712*g**2 - 8*g + 1120
    p4_0 = -123*g**3 + 391*g**2 - 93*g + 3228
    
    # τ
    tau = (2**38 * 3**17 / 23**3) * (47323*g**3 - 1084897*g**2 + 7751*g - 711002)
    
    # Create polynomials
    P2 = Polynomial([p2_0, p2_1, p2_2])
    P3 = Polynomial([p3_0, p3_1, 0, p3_3])
    P4 = Polynomial([p4_0, p4_1, p4_2, p4_3, p4_4])
    
    # Compute P
    P = (P2 * P2) * P3 * (P4 * P4 * P4 * P4) + tau
    return P

# Compute for both g3 and g4
P3_poly = compute_poly(g3)
P4_poly = compute_poly(g4)

# Average them to get real coefficients
coeffs3 = P3_poly.coef
coeffs4 = P4_poly.coef
real_coeffs = [(c3 + c4)/2 for c3, c4 in zip(coeffs3, coeffs4)]

print("\n" + "="*50)
print("REAL COEFFICIENTS (from averaging g3 and g4):")
print("="*50)

# Check if imaginary parts are zero
max_imag = 0
for i, c in enumerate(real_coeffs):
    if abs(c.imag) > max_imag:
        max_imag = abs(c.imag)
    print(f"a{i:2d} = {c.real:.0f}")

print(f"\nMaximum imaginary part: {max_imag}")

# Round to nearest integer (they should be integers)
print("\n" + "="*50)
print("ROUNDED INTEGER COEFFICIENTS:")
print("="*50)
for i, c in enumerate(real_coeffs):
    rounded = round(c.real)
    print(f"a{i:2d} = {rounded}")

import numpy as np
from numpy.polynomial import Polynomial

# Use both g3 and g4 (complex conjugates)
g3 = complex(0.5494723045418892, 0.6756503357678952)
g4 = complex(0.5494723045418892, -0.6756503357678952)

print(f"g3 = {g3}")
print(f"g4 = {g4}")

def compute_poly_for_g(g):
    # P₂ coefficients
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
    
    # Create polynomials
    P2 = Polynomial([P2_coeffs[0], P2_coeffs[1], P2_coeffs[2]])
    P3 = Polynomial([P3_coeffs[0], 0, 0, P3_coeffs[3]])
    P4 = Polynomial([P4_coeffs[0], P4_coeffs[1], P4_coeffs[2], P4_coeffs[3], P4_coeffs[4]])
    
    # Compute P
    P = (P2 * P2) * P3 * (P4 * P4 * P4 * P4) + tau
    return P

# Compute for both g3 and g4
P3_poly = compute_poly_for_g(g3)
P4_poly = compute_poly_for_g(g4)

# Average them to get real coefficients (they should be complex conjugates)
coeffs3 = P3_poly.coef
coeffs4 = P4_poly.coef
real_coeffs = [(c3 + c4)/2 for c3, c4 in zip(coeffs3, coeffs4)]

print("\nReal coefficients (averaged from g3 and g4):")
for i, c in enumerate(real_coeffs):
    if abs(c.imag) > 1e-10:
        print(f"WARNING: Imaginary part not zero: a{i} imag = {c.imag}")
    print(f"a{i:2d} = {c.real:.0f}")

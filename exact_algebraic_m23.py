from sympy import symbols, Poly, expand, RootOf
from sympy.polys.numberfields import minimal_polynomial

# Define g as an algebraic number
g = symbols('g')
# g satisfies: g⁴ + g³ + 9g² - 10g + 8 = 0
g_root = RootOf(g**4 + g**3 + 9*g**2 - 10*g + 8, 0)  # Pick first root

print(f"g is an algebraic number with minimal polynomial: {minimal_polynomial(g_root, g)}")

# Now define P₂, P₃, P₄ in terms of g
P2 = (8*g**3 + 16*g**2 - 20*g + 20) * g**2 + (-7*g**3 - 17*g**2 + 7*g - 76) * g + (-13*g**3 + 25*g**2 - 107*g + 596)
P3 = 8*(31*g**3 + 405*g**2 - 459*g + 333) * g**3 + (941*g**3 + 1303*g**2 - 1853*g + 1772) * g + (85*g**3 - 385*g**2 + 395*g - 220)
P4 = 32*(4*g**3 - 69*g**2 + 74*g - 49) * g**4 + 32*(21*g**3 + 53*g**2 - 68*g + 58) * g**3 - 8*(97*g**3 + 95*g**2 - 145*g + 148) * g**2 + 8*(41*g**3 - 89*g**2 - g + 140) * g + (-123*g**3 + 391*g**2 - 93*g + 3228)

tau = (2**38 * 3**17 / 23**3) * (47323*g**3 - 1084897*g**2 + 7751*g - 711002)

print("\nP₂ =", expand(P2))
print("\nP₃ =", expand(P3))
print("\nP₄ =", expand(P4))
print("\nτ =", tau)

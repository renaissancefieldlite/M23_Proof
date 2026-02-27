from sympy import symbols, Poly, expand, rem

# Define g
g = symbols('g')
# Minimal polynomial: g⁴ = -g³ - 9g² + 10g - 8

def reduce_poly(expr):
    """Reduce polynomial in g to degree < 4 using minimal polynomial"""
    poly = Poly(expr, g)
    # Get coefficients
    coeffs = poly.all_coeffs()
    deg = len(coeffs) - 1
    
    # Start with the expression
    result = expr
    
    # Reduce degree 4 and higher using g⁴ = -g³ - 9g² + 10g - 8
    while True:
        poly = Poly(result, g)
        if poly.degree() < 4:
            break
        
        # Get leading term
        lead_coeff = poly.LC()
        lead_deg = poly.degree()
        
        # How many times to apply reduction
        if lead_deg >= 4:
            # Replace g^lead_deg with g^(lead_deg-4) * g^4
            # and g^4 = -g³ - 9g² + 10g - 8
            reduction = lead_coeff * g**(lead_deg-4) * (-g**3 - 9*g**2 + 10*g - 8)
            result = result - lead_coeff * g**lead_deg + reduction
    
    return expand(result)

# P₂ = 8g⁵ + 9g⁴ - 50g³ + 52g² - 183g + 596
P2_expr = 8*g**5 + 9*g**4 - 50*g**3 + 52*g**2 - 183*g + 596
P2_reduced = reduce_poly(P2_expr)
print(f"P₂ = {P2_reduced}")

# P₃ = 248g⁶ + 3240g⁵ - 2731g⁴ + 4052g³ - 2238g² + 2167g - 220
P3_expr = 248*g**6 + 3240*g**5 - 2731*g**4 + 4052*g**3 - 2238*g**2 + 2167*g - 220
P3_reduced = reduce_poly(P3_expr)
print(f"P₃ = {P3_reduced}")

# P₄ = 128g⁷ - 1536g⁶ + 3288g⁵ - 4176g⁴ + 2181g³ - 801g² + 1027g + 3228
P4_expr = 128*g**7 - 1536*g**6 + 3288*g**5 - 4176*g**4 + 2181*g**3 - 801*g**2 + 1027*g + 3228
P4_reduced = reduce_poly(P4_expr)
print(f"P₄ = {P4_reduced}")

# τ = huge expression
tau_expr = (2**38 * 3**17 / 23**3) * (47323*g**3 - 1084897*g**2 + 7751*g - 711002)
print(f"\nτ = {tau_expr}")

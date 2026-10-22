import numpy as np

def inv_fn(x):
    return 1.0 / x

# The polynomial only approximates 1/x over the mantissa range [1, 2);
# the exponent is handled separately via frexpf/ldexpf in the kernel.
lb, ub = 1.0, 2.0
n_parts = 16
degree = 2

breakpoints = np.linspace(lb, ub, n_parts + 1)
coeffs = []
for i in range(n_parts):
    x = np.linspace(breakpoints[i], breakpoints[i+1], 100)
    y = inv_fn(x)
    c = np.polyfit(x, y, degree)
    coeffs.append(c.tolist())

print("PACE_BREAKPOINTS =", breakpoints.tolist())
print("PACE_COEFFS =", coeffs)

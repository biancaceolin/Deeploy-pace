import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

lb, ub = -5.0, 5.0
n_parts = 16
degree = 2

breakpoints = np.linspace(lb, ub, n_parts + 1)
coeffs = []
for i in range(n_parts):
    x = np.linspace(breakpoints[i], breakpoints[i+1], 100)
    y = gelu(x)
    c = np.polyfit(x, y, degree)
    coeffs.append(c.tolist())

print("PACE_BREAKPOINTS =", breakpoints.tolist())
print("PACE_COEFFS =", coeffs)

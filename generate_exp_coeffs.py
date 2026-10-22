import numpy as np

def exp_fn(x):
    return np.exp(x)

# Domain used by Arpan's golden model for the Exp step in Softmax
lb, ub = -11.0, 0.0
n_parts = 16
degree = 2

breakpoints = np.linspace(lb, ub, n_parts + 1)
coeffs = []
for i in range(n_parts):
    x = np.linspace(breakpoints[i], breakpoints[i+1], 100)
    y = exp_fn(x)
    c = np.polyfit(x, y, degree)
    coeffs.append(c.tolist())

print("PACE_BREAKPOINTS =", breakpoints.tolist())
print("PACE_COEFFS =", coeffs)

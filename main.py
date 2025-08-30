# Integration 
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)
a, b = 0, np.pi
n = 20
h = (b - a) / n
x = np.linspace(a, b, n + 1)
y = f(x)
# Trapezoidal Rule
trap = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
# Simpson's Rule
if n % 2 == 1:
    n += 1
    x = np.linspace(a, b, n + 1)
    y = f(x)
h = (b - a) / n
simp = (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))

exact = 2
print(f"Trapezoidal = {trap:.10f}, Error = {abs(exact - trap):.2e}")
print(f"Simpson's   = {simp:.10f}, Error = {abs(exact - simp):.2e}")

# Interpolation
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
 
x = [3, 4.5, 7, 9]
f = [2.5, 1, 2.5, 0.5]
n = len(x)

# -------------------- Newton Divided Difference --------------------
def divided_diff(x, y):
    n = len(x)
    coef = np.zeros([n, n])
    coef[:,0] = y
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1]-coef[i][j-1])/(x[i+j]-x[i])
    return coef

def newton_poly(coef, x_data, x):
    n = len(coef)
    p = coef[0]
    for i in range(1,n):
        term = coef[i]
        for j in range(i):
            term *= (x - x_data[j])
        p += term
    return p

coef = divided_diff(x,f)
print("Divided Difference Table:")
print(coef)

# -------------------- Forward Difference --------------------
def forward_interpolation(x, y, val):
    n = len(x)
    h = x[1] - x[0]
    diff = np.zeros((n, n))
    diff[:,0] = y
    for j in range(1, n):
        for i in range(n-j):
            diff[i][j] = diff[i+1][j-1] - diff[i][j-1]
    u = (val - x[0])/h
    result = y[0]
    u_term = 1
    for j in range(1,n):
        u_term *= (u - j + 1)/j
        result += u_term * diff[0][j]
    return result

# -------------------- Backward Difference --------------------
def backward_interpolation(x, y, val):
    n = len(x)
    h = x[1] - x[0]
    diff = np.zeros((n, n))
    diff[:,0] = y
    for j in range(1, n):
        for i in range(j,n):
            diff[i][j] = diff[i][j-1] - diff[i-1][j-1]
    u = (val - x[-1])/h
    result = y[-1]
    u_term = 1
    for j in range(1,n):
        u_term *= (u + j - 1)/j
        result += u_term * diff[-1][j]
    return result

# -------------------- Lagrange Interpolation --------------------
def lagrange_interpolation(x, y, val):
    total = 0
    n = len(x)
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (val - x[j])/(x[i]-x[j])
        total += term
    return total

# -------------------- Cubic Spline --------------------
def spline_equation_row(xi_minus1, xi, xi_plus1, f_xi_minus1, f_xi, f_xi_plus1):
    a = xi - xi_minus1
    b = 2 * (xi_plus1 - xi_minus1)
    c = xi_plus1 - xi
    rhs = (6 / (c)) * (f_xi_plus1 - f_xi) - (6 / (a)) * (f_xi - f_xi_minus1)
    return a, b, c, rhs

variables = sp.symbols(f'f\'\'1:{n-1}')
equations = []
for i in range(1, n - 1):
    a, b, c, rhs = spline_equation_row(x[i - 1], x[i], x[i + 1], f[i - 1], f[i], f[i + 1])
    row = 0
    if i > 1:
        row += a * variables[i - 2]
    row += b * variables[i - 1]
    if i < n - 2:
        row += c * variables[i]
    equations.append(sp.Eq(row, rhs))

sol = sp.solve(equations, variables)
fpp = [0] + [sol[v] for v in variables] + [0]

def cubic_spline_interpolation(x_val, xi, xi1, fxi, fxi1, fppi, fppi1):
    h = xi1 - xi
    term1 = fppi * ((xi1 - x_val) ** 3) / (6 * h)
    term2 = fppi1 * ((x_val - xi) ** 3) / (6 * h)
    term3 = (fxi / h - fppi * h / 6) * (xi1 - x_val)
    term4 = (fxi1 / h - fppi1 * h / 6) * (x_val - xi)
    return term1 + term2 + term3 + term4
 
X_vals = np.linspace(min(x), max(x), 200)

Y_divided = [newton_poly(coef[0,:], x, xv) for xv in X_vals]
Y_forward = [forward_interpolation(x,f,xv) for xv in X_vals]
Y_backward = [backward_interpolation(x,f,xv) for xv in X_vals]
Y_lagrange = [lagrange_interpolation(x,f,xv) for xv in X_vals]

Y_spline = []
for i in range(n-1):
    xs = np.linspace(x[i],x[i+1],50)
    ys = [cubic_spline_interpolation(xv, x[i], x[i+1], f[i], f[i+1], fpp[i], fpp[i+1]) for xv in xs]
    Y_spline.extend(ys)

plt.figure(figsize=(12,6))
plt.plot(X_vals, Y_divided, label="Newton Divided Difference", color="purple")
plt.plot(X_vals, Y_forward, label="Newton Forward", linestyle="--", color="orange")
plt.plot(X_vals, Y_backward, label="Newton Backward", linestyle="--", color="green")
plt.plot(X_vals, Y_lagrange, label="Lagrange", linestyle=":", color="red")
plt.plot(np.linspace(min(x), max(x), len(Y_spline)), Y_spline, label="Cubic Spline", color="blue")
plt.plot(x, f, 'ko', label="Data Points")
plt.title("Interpolation Methods")
plt.legend()
plt.grid(True)
plt.show()


# t test 
import math 
def t_test_paired(X, Y):
    if len(X) != len(Y):
        raise ValueError("Both samples must have same length")

    n = len(X) 
    diffs = [X[i] - Y[i] for i in range(n)] 
    mean_diff = sum(diffs) / n 
    sq_diffs = [(d - mean_diff)**2 for d in diffs]
    sd = math.sqrt(sum(sq_diffs) / (n - 1)) 
    t_value = mean_diff / (sd / math.sqrt(n))

    return t_value

# Example
X = [10, 12, 13, 15, 16]
Y = [9, 11, 12, 14, 15]

t_val = t_test_paired(X, Y)
print("t statistic =", t_val)

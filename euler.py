import numpy as np
import matplotlib.pyplot as plt

# Euler
def euler_system(F, x0, Y0, h, num_steps):
    x_vals = [x0]
    Y_vals = [np.array(Y0, dtype=float)]
    x, Y = x0, np.array(Y0, dtype=float)
    for _ in range(num_steps):
        Y = Y + h * F(x, Y)
        x = x + h
        x_vals.append(x)
        Y_vals.append(Y.copy())
    return np.array(x_vals), np.vstack(Y_vals)   

#   RK4 (system form) 
def rk4_system(F, x0, Y0, h, num_steps):
    x_vals = [x0]
    Y_vals = [np.array(Y0, dtype=float)]
    x, Y = x0, np.array(Y0, dtype=float)
    for _ in range(num_steps):
        k1 = h * F(x, Y)
        k2 = h * F(x + 0.5*h, Y + 0.5*k1)
        k3 = h * F(x + 0.5*h, Y + 0.5*k2)
        k4 = h * F(x + h,   Y + k3)
        Y = Y + (k1 + 2*k2 + 2*k3 + k4)/6.0
        x = x + h
        x_vals.append(x)
        Y_vals.append(Y.copy())
    return np.array(x_vals), np.vstack(Y_vals)

# Changing y'' + 6 y' = 0 as a first-order system
# Let v = y'
# Then y' = v
#      v' = -6 v
def F(x, Y):
    y, v = Y
    return np.array([v, -6.0*v])
 
x0   = 0.0
y0   = 1.0       # y(x0)
v0   = 1.0       # y'(x0)
h    = 0.1
N    = 100

xe, Ye = euler_system(F, x0, [y0, v0], h, N)
xr, Yr = rk4_system(F, x0, [y0, v0], h, N)

# Ye[:,0] and Yr[:,0] are y(x); Ye[:,1] and Yr[:,1] are v(x)=y'(x)
 
plt.figure(figsize=(9,5))
plt.plot(xe, Ye[:,0], 'o-', label="Euler y", markevery=8)
plt.plot(xr, Yr[:,0], 's-', label="RK4 y", markevery=8)

plt.xlabel("x"); plt.ylabel("y")
plt.title("y'' + 6 y' = 0  →  system form (y, v=y')")
plt.grid(True); plt.legend()
plt.show()

import numpy as np
from math import *
import matplotlib.pyplot as plt

# to be clear, I despise the function parameters we used, so imma write my own lmao. Maybe I suck at coding but I see these problems diff and these params blow

# Define the analytical solution
def analytical_solution(t):
    return np.exp(0.25 * t**4 - 1.5 * t)

# Define the derivative function dy/dt
def dydt(y, t):
    return y * t**3 - y * 1.5

# Function to solve a differential equation using the given method, below we will define the methods
def solve_differential_eq(method, h, y0):
    y = [y0]
    t = np.arange(0, 2 + h, h)
    
    for i in range(len(t) - 1):
        y.append(method(y[i], t[i], h, dydt))
    
    return y, t

# Euler's method
def euler(y, t, h, dydt):
    return y + dydt(y, t) * h

# Midpoint method
def midpoint(y, t, h, dydt):
    y_predict = y + dydt(y, t) * (h / 2)
    t_predict = t + (h / 2)
    return y + dydt(y_predict, t_predict) * h

# 4th order Runge-Kutta method
def rk4th(y, t, h, dydt):
    k1 = dydt(y, t)
    k2 = dydt(y + 0.5 * k1 * h, t + 0.5 * h)
    k3 = dydt(y + 0.5 * k2 * h, t + 0.5 * h)
    k4 = dydt(y + k3 * h, t + h)
    return y + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4) * h

# Time span and initial value
t0 = np.linspace(0, 2, num=100)
y0 = 1
h = 0.5
h25 = .25

# Plot the analytical solution
plt.plot(t0, analytical_solution(t0), label="Analytical Solution")

# Show the plot
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Comparing Different Solvers vs Analytical Solution")
# Calculate and plot solutions for different methods with the same step size

# Euler's method h = .5
ye1, te1 = solve_differential_eq(euler, h, y0)
plt.plot(te1, ye1, label="Euler h = 0.5")

# Euler's method h = .25
ye2, te2 = solve_differential_eq(euler, h25, y0)
plt.plot(te2, ye2, label="Euler h = 0.25")

# Midpoint method
ym, tm = solve_differential_eq(midpoint, h, y0)
plt.plot(tm, ym, label="Midpoint")

# 4th order Runge-Kutta method
yk, tk = solve_differential_eq(rk4th, h, y0)
plt.plot(tk, yk, label="4th order RK")

plt.legend()
plt.show()

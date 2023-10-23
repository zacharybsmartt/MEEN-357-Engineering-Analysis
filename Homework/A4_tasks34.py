import numpy as np
from math import *
import matplotlib.pyplot as plt
from A4_task2 import rk4th

# Define the second-order differential equation
def secondderiv(y, x):
    return np.array([y[1], -0.5 * y[1] - 7 * y[0]])

# Function to solve a differential equation using the given method
def solve_differential_eq(method, y0, h, f):
    x = np.arange(0, 5 + h, h)
    n = len(x)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = method(y[i], x[i], h, f)
    return y, x

# Initial conditions
y0 = np.array([4, 0])

# Call the solve_differential_eq function with rk4th method
sol, x = solve_differential_eq(rk4th, y0, 0.5, secondderiv)

fig, axs = plt.subplots(2)
fig.subplots_adjust(hspace=0.6)

axs[0].set_title('dy/dx vs x')
axs[0].plot(x, sol[:, 1])
axs[0].set_xlabel('x')
axs[0].set_ylabel('dy/dx')

axs[1].set_title('y vs x')
axs[1].plot(x, sol[:, 0])
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

plt.show()

from A4 import euler_integrate, midpoint_integrate, RK4_integrate
import numpy as np
from math import *
import matplotlib.pyplot as plt
# euler_integrte(fun, t0, y0, tStop, h)

def dydt(t, y):
    return (y * (t**3)) - (1.5 * y)


def F_analytical(t):
    return np.exp((1 / 4) * t ** 4 - 1.5 * t)


t0 = np.linspace(0, 2, num = 1000)
y0 = 1


# analytical plot
plt.plot(t0, F_analytical(t0),label = "Analytical Solution")
plt.xlabel("t")
plt.ylabel("y(t)")
# euler step size = .5
plt.plot(, label = "Euler: h = .5")
# euler step size = .25
plt.plot(, label = "Euler: h = .25")
# midpoint step size = .5
plt.plot(, label = "Midpoint: h = .5")
# RK4 step size = .5
plt.plot(, label = "RK4: h = .5")
plt.show()


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def springmassdamper(t, y, m, k, b):
    x, v = y[0], y[1]

    dydt = [v,
            -k/m*x - b/m*v]
    return dydt

m = 20; b = 5; k = 20; tspan = (0,120); y0 = np.array([1,0])
fun = lambda t, y: springmassdamper(t, y, m, k, b)

sol = solve_ivp(fun, tspan, y0, method='RK45', t_eval=np.linspace(0, 120, 10))
T = sol.t
x = sol.y[0, :]
v = sol.y[1, :]

print(T, x, v)

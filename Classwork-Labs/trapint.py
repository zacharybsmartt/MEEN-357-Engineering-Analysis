import numpy as np
import matplotlib.pyplot as plt
"""
integrate 0-3 of x^2e^xdx = 98.4277
Planning:
Multiple trapezoidal rules
x -> f(x)
1: plot my points
2: calculate area of each trapezoid
3: sum all areas

for i in range(x-1):
    h[i] = x[i + 1] - x[i]
    I[i] = h[i] * (1/2) * fx[i + 1] + fx[i]

I_est = sum[I]
"""


def f(x):
    return (x**2)*(np.exp(x))

def trapint(x, fx):
    """Where x is the number of data points 'n' and the # of integral segments is n-1, where f(x) is the function output at x locations"""
    nseg = len(x)-1
    I = np.zeros(nseg)
    for i in range(nseg):
        h = x[i + 1] - x[i]
        I[i] = h * .5 * (fx[i + 1] + fx[i])
        print('hvalue', h, 'Integral:',I[i])
    
    I_est = sum(I)

    return I_est

tru_sol = 98.4277
nseg = 100
x = np.linspace(0, 3, nseg + 1)
fx = f(x)
Int_est = trapint(x, fx)
print('integral estimate', Int_est)

import numpy as np
from sympy import *
import matplotlib.pyplot as plt

# generate my function using symbols and the sympy library, then use root
# finding methods from numpy or sympy to find the roots, then graph 
# using matplotlib.pylot library and be able to see roots defined

# x = symbols('x')

# def f_output(x):
#     f = x - np.cos(x)
#     return f 

# def main():


# naveen
def f(x):
    return x - np.cos(x)

x = np.arange(0, 2 * np.pi + 2 * np.pi / 20, 2 * np.pi / 20)

fval = np.arange(0, 2 * np.pi + 2 * np.pi / 20, 2 * np.pi / 20)
for i in range(len(x)):
    fval[i] = f(x[i])

print(fval)
plt.figure()
plt.plot(x, fval, "m")
plt.xlabel("x")
plt.ylabel('y')
plt.show()

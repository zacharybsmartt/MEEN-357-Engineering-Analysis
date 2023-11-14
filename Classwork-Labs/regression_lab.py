import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
# least square fit linear regression for the function y = 30 + 2x + 3x^2 + E, E~(0,20)

np.random.seed(6)
x = np.linspace(0,9,10)
x = x.reshape(len(x), 1)
noise =  20 * np.random.normal(size=(np.size(x, 0), np.size(x, 1)))
y = 30 + 2 * x + 3 * x ** 2 + noise

print('test')

from math import *
import numpy as np

f = lambda x : x ** 2 #lambda function basis

def f_diff(f, xi, h):
    df_dx = (f(xi + h) - f(xi)) / h
    return df_dx


def b_diff(f, xi, h):
    df_dx = (f(xi) - f(xi + h)) / h
    return df_dx

def c_diff(f, xi, h):
    df_dx = (f(xi + h) - f(xi - h)) / 2 * h
    return df_dx

xi = 1
true_sol = 2 * xi
h = 10e-1
h_array = np.zeros(10)
true_error_array = np.zeros(10)

for i in range(10):
    h_array[i] = h/10 ** i
    h = 10e-1/10
    df_dx = b_diff(f, xi, h)
    true_error = (true_sol - df_dx)

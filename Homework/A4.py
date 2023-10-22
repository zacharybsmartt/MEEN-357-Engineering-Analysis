import numpy as np
from math import *
import matplotlib.pyplot as plt

# Euler's Method for solving initial value problems.
# This function integrates an ODE using a simple Euler method.
# Input:
#   fun: The ODE function handle, callable as fun(t, y).
#   t0: Initial time.
#   y0: Initial value of y corresponding to t0.
#   tStop: Final time.
#   h: Step size.
# Output:
#   T: Integrated t-values as a 1D numpy array.
#   Y: Integrated y-values as a 1D numpy array.
def euler_integrate(fun, t0, y0, tStop, h):
    T, Y = [], []  # Lists to store time and value.
    t, y = t0, y0  # Initial time and value.
    T.append(t)
    Y.append(y)

    while t < tStop:  # Iterate until reaching tStop.
        h = min(h, tStop - t)  # Adjust step size to not overshoot tStop.
        phi = fun(t, y)  # Calculate the slope at the current point.
        y = y + phi * h  # Update y using Euler's method.
        Y.append(y)
        t = t + h  # Increment time by the step size.
        T.append(t)
    T = np.array(T)
    Y = np.array(Y)

    return T, Y

# Midpoint Method for solving initial value problems.
# This function integrates an ODE using the Midpoint method.
# Input:
#   fun: The ODE function handle, callable as fun(t, y).
#   t0: Initial time.
#   y0: Initial value of y corresponding to t0.
#   tStop: Final time.
#   h: Step size.
# Output:
#   T: Integrated t-values as a 1D numpy array.
#   Y: Integrated y-values as a 1D numpy array.
def midpoint_integrate(fun, t0, y0, tStop, h):
    T, Y = [], []  # Lists to store time and value.
    t, y = t0, y0  # Initial time and value.
    T.append(t)
    Y.append(y)

    while t < tStop:  # Iterate until reaching tStop.
        h = min(h, tStop - t)  # Adjust step size to not overshoot tStop.
        K = fun(T, Y)  # Calculate the slope at the current point.
        yhalf = y + K * (h / 2)  # Update y using the Midpoint method.
        phi = fun((t + h) / 2, yhalf)
        y = y + phi * h
        Y.append(y)
        t = t + h  # Increment time by the step size.
        T.append(t)
    T = np.array(T)
    Y = np.array(Y)

    return T, Y

# Fourth-Order Runge-Kutta (RK4) Method for solving initial value problems.
# This function integrates an ODE using the RK4 method.
# Input:
#   fun: The ODE function handle, callable as fun(t, y).
#   t0: Initial time.
#   y0: Initial value of y corresponding to t0.
#   tStop: Final time.
#   h: Step size.
# Output:
#   T: Integrated t-values as a 1D numpy array.
#   Y: Integrated y-values as a 1D numpy array.
def RK4_integrate(fun, t0, y0, tStop, h):
    T, Y = [], []  # Lists to store time and value.
    t, y = t0, y0  # Initial time and value.
    T.append(t)
    Y.append(y)

    while t < tStop:  # Iterate until reaching tStop.
        h = min(h, tStop - t)  # Adjust step size to not overshoot tStop.
        k1 = fun(t, y)  # Compute the slopes at various points.
        k2 = fun(t + h / 2, y + k1 * (h / 2))
        k3 = fun(t + h / 2, y + k2 * (h / 2))
        k4 = fun(t + h, y + k3 * h)

        y = y + (k1 + 2 * k2 + 2 * k3 + k4) * (h / 6)  # Update y using RK4 method.
        t = t + h  # Increment time by the step size.

        T.append(t)
        Y.append(y)
    T = np.array(T)
    Y = np.array(Y)

    return T, Y

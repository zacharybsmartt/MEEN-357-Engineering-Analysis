import numpy as np
import matplotlib.pyplot as plt
from A4_task2 import *

# below is task 3
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


def Right_Side(t, statevector):
    y, v = statevector
    dvdt = -0.5 * v - 7 * y
    dydt = v
    return np.array([dydt, dvdt])


def RK4_integrate(fun, t0, y0, tStop, h):
    # Initialize lists and initial values
    T = []
    Y = []
    t = t0
    y = y0
    # Update the initial values to the list
    T.append(t0)
    Y.append(y0)

    while t < tStop:
        # Choose h value
        h = min(h, tStop - t)

        # Calculate k values
        k1 = fun(t, y)
        k2 = fun(t + h / 2, y + k1 * h / 2)
        k3 = fun(t + h / 2, y + k2 * h / 2)
        k4 = fun(t + h, y + k3 * h)
        phi = 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Calculate y and t + h
        y = y + phi * h
        t = t + h
        T.append(t)
        Y.append(y)

    return np.array(T), np.array(Y)


def RK4_integrate2(fun, t0, y0, tStop, h):
    # Initialize lists and initial values
    t = t0
    y = y0
    
    while t < 2:
        # Choose h value
        h = min(h, tStop - t)

        # Calculate k values
        k1 = fun(t, y)
        k2 = fun(t + h / 2, y + k1 * h / 2)
        k3 = fun(t + h / 2, y + k2 * h / 2)
        k4 = fun(t + h, y + k3 * h)
        phi = 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Calculate y and t + h
        y = y + phi * h
        t = t + h
            
    return t, y

# Initial conditions
y0 = np.array([4, 0])

# Call the solve_differential_eq function with rk4th method
sol, x = solve_differential_eq(rk4th, y0, 0.5, secondderiv)
# Initial conditions
t0 = 0
y0 = np.array([4, 0])  # Initial condition
# Time span and step size
time_span = 5
h = 0.5

# task 4 values
hval = []
T_err_Y = []
T_err_V = []
T, true_vals = RK4_integrate2(Right_Side, t0, y0, time_span, 0.5**20)
Ytrue = true_vals[0]
Vtrue = true_vals[1]


#Calculate the values at x = 2 for each h=0.5**p
for i in range(1,8):
    h = 0.5**i
    X,Y = RK4_integrate2(Right_Side, t0, y0, time_span, h)
    
    hval.append(h)
    T_err_Y.append(abs(Ytrue-Y[0]))
    T_err_V.append(abs(Vtrue-Y[1]))



# Plot task 3
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

# Plot task 4
plt.loglog(hval,T_err_Y,label = 'Y')
plt.loglog(hval,T_err_V,label = 'V')
plt.xlabel('Segment Width (h)')
plt.ylabel('True Error at x = 2')
plt.title('Log-Log Plot')
plt.grid()
plt.legend()
plt.show()

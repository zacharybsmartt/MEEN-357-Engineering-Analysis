from math import *
import numpy as np
import matplotlib.pyplot as plt

true_sol = 98.42768461593835 # the true solution to the integral, what will be the base for comparison later

def simpint(x, fx):
    I = 0
    h = [x[i + 1] - x[i] for i in range(len(x) - 1)]

    if len(x) != len(fx):
        raise Exception('Both inputs, x and f(x), must be the same lengeth')
    if len(x) not in (3, 4):
        raise Exception('There are less than 3 or more than 4 data point given, input only 3 or 4 values for input 1')
    if not (isinstance(x, list) and isinstance(fx, list)):
        raise Exception('Ensure that your inputs are vectors/lists')
    if all(seg_length != h[0] for seg_length in h):
        raise Exception('Ensure that h ()')
    if len(x) == 1:
        return I # If there is only one data point, I cannot integrate and the result will be 0

    if len(x) == 3: # use simpsons 1/3 rule
        I = h[0]/3 * (fx[0] + 4 * fx[1] + fx[2])
    elif len(x) == 4: # use simpsons 3/8 rule
        I = ((3*h[0])/8) * ((fx[0] + 3 * fx[1] + 3 * fx[2] + fx[3]))

    return I


def fun(x):
    return [(i ** 2) * exp(i) for i in x]

# Example usage:
h15 = [0, 1.5, 3]
Ih15 = simpint(h15, fun(h15)) # done

h1 = [0, 1, 2, 3]
Ih1 = simpint(h1, fun(h1)) # done

h75 = [0, .75, 1.5, 2.25, 3.0]
Ih75 = (simpint(h75[0:3],fun(h75[0:3])) + 
        simpint(h75[2:], fun(h75[2:]))) # done

h05 = [0, .5, 1, 1.5, 2, 2.5, 3.0]
Ih05 = (simpint(h05[0:4], fun(h05[0:4])) + 
        simpint(h05[3:], fun(h05[3:]))) # done

h0375 = [0, .375, .75, 1.125, 1.5, 1.875, 2.25, 2.625, 3]
Ih0375 = (simpint(h0375[0:4], fun(h0375[0:4])) + 
          simpint(h0375[3:6], fun(h0375[3:6])) + 
          simpint(h0375[5:], fun(h0375[5:])))

h33_ = [0, 1/3, 2/3, 1, 4/3, 5/3, 2, 7/3, 8/3, 3]
Ih33_ = (simpint(h33_[0:4], fun(h33_[0:4])) + 
         simpint(h33_[3:7], fun(h33_[3:7])) + 
         simpint(h33_[6:], fun(h33_[6:])))

h03 = [0, .3, .6, .9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
Ih03 = (simpint(h03[0:3], fun(h03[0:3])) +
        simpint(h03[2:5], fun(h03[2:5])) +
        simpint(h03[4:7], fun(h03[4:7])) +
        simpint(h03[6:9], fun(h03[6:9])) +
        simpint(h03[8:], fun(h03[8:])))

h025 = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0]
Ih025 = (simpint(h025[0:3], fun(h025[0:3])) +
        simpint(h025[2:5], fun(h025[2:5])) +
        simpint(h025[4:7], fun(h025[4:7])) +
        simpint(h025[6:9], fun(h025[6:9])) +
        simpint(h025[8:11], fun(h025[8:11])) +
        simpint(h025[10:], fun(h025[10:])))

# Define the h values and corresponding Ih values
h_values = [1.5, 1, 0.75, 0.5, 0.375, 1/3, 0.3, 0.25]
Ih_values = [Ih15, Ih1, Ih75, Ih05, Ih0375, Ih33_, Ih03, Ih025]

# Plot the true solution as a horizontal line
plt.axhline(y=true_sol, color='r', linestyle='--', label='True Solution')

# Plot the integral approximations
plt.plot(h_values, Ih_values, marker='o', linestyle='-', label='Integral Approximations')

plt.xlabel('h')
plt.ylabel('Integral Value')
plt.title("Simpson's Integrals Approaching True Solution")
plt.legend()
plt.grid(True)
plt.show()


# task 3c below
I_task_c = simpint([0, .05, .1],[40, 37.5, 43]) + simpint([.1, .15, .2, .25], [43, 52, 60, 55])
print(I_task_c)

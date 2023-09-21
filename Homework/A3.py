from math import *
import numpy as np

true_sol = 98.42768461593835 # the true solution to the integral, what will be the base for comparison later


def simpint(x, fx):
    I = 0
    h = [x[i + 1] - x[i] for i in range(len(x) - 1)]

    if len(x) != len(fx):
        raise Exception('Ensure that the x and fx are the same length')
    elif len(x) not in (3, 4):
        raise Exception('There are less than 3 or more than 4 data point given, input only 3 or 4 values for input 1')
    elif not (isinstance(x, list) and isinstance(fx, list)):
        raise Exception('Ensure that your inputs are vectors/lists')
    elif all(seg_length != h[0] for seg_length in h):
        raise Exception('Ensure that h ()')
    elif len(x) == 1:
        return I # If there is only one data point, I cannot integrate and the result will be 0

    if len(x) == 3: # use simpsons 1/3 rule
        print(h[0])
        I = h[0]/3 * (fx[0] + 4 * fx[1] + fx[2])
    elif len(x) == 4: # use simpsons 3/8 rule
        print(h[0])
        I = (3*h[0])/8 * ((fx[0] + 3 * fx[1] + 3 * fx[2] + fx[3]))

    return I


def fun(x):
    return [(i ** 2) * exp(i) for i in x]

# # Example usage:
# h15 = [0, 1.5, 3]
# Ih15 = simpint(h15, fun(h15)) # done

# h1 = [0, 1, 2, 3]
# Ih1 = simpint(h1, fun(h1)) # done

# h75 = [0, .75, 1.5, 2.25, 3.0]
# print(h75[0:3])
# print(h75[2:])
# Ih75 = simpint(h75[0:3],fun(h75[0:3])) + simpint(h75[2:], fun(h75[2:])) # done

# h05 = [0, .5, 1, 1.5, 2, 2.5, 3.0]
# print(h05[0:4])
# print(h05[3:])
# Ih05 = simpint(h05[0:4], fun(h05[0:4])) + simpint(h05[3:], fun(h05[3:]))

# h0375 = [0, .375, .75, 1.125, 1.5, 1.875, 2.25, 2.625, 3]
# # Ih0375 = simpint() + simpint() + simpint()

# h33_ = [0, 1/3, 2/3, 1, 4/3, 5/3, 2, 7/3, 8/3, 3]
# h03 = [0, .3, .6, .9, 1.2,  1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
# h025 = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0]
# print(Ih15)
# print(Ih1)
# print(Ih75)
# print(Ih05)

# task 3c below
I_task_c = simpint([0, .05, .1],[40, 37.5, 43]) + simpint([.1, .15, .2, .25], [43, 52, 60, 55])
print(I_task_c)

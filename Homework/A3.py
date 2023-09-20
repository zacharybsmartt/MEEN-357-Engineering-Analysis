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
        I += (x[2] - x[0]) * ((fx[0] + 4 * fx[1] + fx[2]) / 6)
    elif len(x) == 4: # use simpsons 3/8 rule
        I = ((x[3] - x[0])) * ((fx[0] + 3 * fx[1] + 3 * fx[2] + 3 * fx[3]) / 8)

    return I


def fun(x):
    return [(i ** 2) * exp(i) for i in x]

def step_list(step_size):   
    num_steps = int(3 / step_size) + 1
    return [i * step_size for i in range(num_steps)]

# Example usage:
h15 = [0, 1.5, 3]


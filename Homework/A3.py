from math import *

true_sol = 98.42768461593835 # the true solution to the integral, what will be the base for comparison later


def fun(x):
    return (x ** 2) * exp(x)


def simpint(x, fx):
    I = 0
    h = [x[i + 1] - x[i] for i in range(len(x) - 1)]

    if len(x) != len(fx):
        raise Exception('Ensure that the x and fx are the same length')
    elif len(x) not in (3, 4):
        raise Exception('There are less than 3 or more than 4 data point given, input only 3 or 4 values for input 1')
    elif type(x) or type(fx) != list:
        raise Exception('Ensure that your inputs are vectors/lists')
    elif all(seg_length != h[0] for seg_length in h):
        raise Exception('Ensure that h ()')
    elif len(x) == 1:
        return I # If there is only one data point, I cannot integrate and the result will be 0

    if len(x) == 3: # use simpsons 1/3 rule
        
    elif len(x) == 4: # use simpsons 3/8 rule
        
    return I



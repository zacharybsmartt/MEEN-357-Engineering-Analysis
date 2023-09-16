from math import *

def fun(x):
    return x * 2

def simpint(x, fx):
    I = 0
    differences = [x[i + 1] - x[i] for i in range(len(x) - 1)]

    if len(x) != len(fx):
        raise Exception('Ensure that the x and fx are the same length')
    elif len(x) not in (3, 4):
        raise Exception('There are less than 3 or more than 4 data point given, input only 3 or 4 values for input 1')
    elif type(x) or type(fx) != list:
        raise Exception('Ensure that your inputs are vectors/lists')
    elif all(h != differences[0] for h in differences):
        raise Exception('Ensure that h ()')

    if len(x) == 3: # use simpsons 1/3 rule
        
    elif len(x) == 4: # use simpsons 3/8 rule
        
    return I



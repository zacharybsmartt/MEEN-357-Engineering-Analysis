from math import *

def bisection(fun,lb,ub,err_max = 1e-8, iter_max = 1000):
    global root, err, numIter, exitFlag
    done = False
    numIter = 0
    root = 0
    err = 0

    if type(lb) == int or type(lb) == float: pass
    else: raise Exception('The lower bound input type is incorrect. Input an integer value.')
    
    if type(ub) == int or type(ub) == float: pass
    else: raise Exception('The upper bound input type is incorrect. Input an integer value.')
    
    if (type(err_max) == int or type(err_max) == float) and err_max >= 0: pass
    else: raise Exception('The err_max input type is incorrect. Input an integer value.')
    
    if type(iter_max) == int or type(iter_max) == float: pass
    else: raise Exception('The iter_max input type is incorrect. Input an integer value.')

    if iter_max <=0: raise Exception('Maximum number of iterations is either zero or negative. Input a positive value.')

    if err_max <= 0: raise Exception('Maximum error is either zero or a negative value. Input a positive value.')
    
    if ub < lb: raise Exception("Make upper bound greater than lower bound.")
    
    if isinf(fun(lb)) or isinf(fun(ub)) or isnan(fun(lb)) or isnan(fun(ub)): exitFlag = -2 ; done = True
   
    if fun(lb)*fun(ub) > 0: exitFlag = -1 ; done = True


    while not done:
        numIter += 1
        root = (lb + ub) / 2
        
        if fun(lb)*fun(root) < 0: ub = root 
        else: lb = root

        err = abs((lb - ub) / (ub + lb)) * 100
        
        if err_max > err: exitFlag = 1 ; done = True

        if numIter == iter_max: exitFlag = 0 ; done = True

    return root, err, numIter, exitFlag

def secant(fun, ini_guess, err_max=1e-6, iter_max=10000):
    xold = 100000000000000000000000000000
    xnew = ini_guess
    numIter = 0

    if not callable(fun):
        raise ValueError("The function 'fun' must be callable.")

    if not isinstance(ini_guess, (int, float)):
        raise ValueError("The initial guess must be a scalar value.")

    if not (err_max > 0 and iter_max > 0):
        raise ValueError("err_max and iter_max must be positive scalar values.")

    while numIter < iter_max:
        m = (fun(xnew) - fun(xnew + 1e-6)) / 1e-6  # Compute the slope numerically
        if m == 0:
            raise ValueError("Slope (m) is zero. Cannot proceed with the secant method.")
        
        b = fun(xnew) - m * xnew
        xnew = -b / m
        numIter += 1
        err = abs((xnew - xold) / xnew) * 100
        xold = xnew

        if err < err_max:
            return xnew, err, numIter, 1  # Found a root

    raise ValueError("Maximum number of iterations reached without finding a root.")
# def secant(fun, ini_guess, err_max = 10e-6, iter_max = 10000):
#     global root, err, numIter, exitFlag
#     done = False
#     xold = 100000000000000000000000000000
#     xnew = ini_guess
#     numIter = 0
#     step = 0.001

#     if not callable(fun):
#         exitFlag = -2
#         return root, err, numIter, exitFlag
    
#     if isinstance(ini_guess,(int,float)) == False:
#         exitFlag = -1
#         raise ValueError('Exit Flag:', exitFlag, "The second inpt must be scaler values")
        
#     if ((err_max < 0) or (iter_max < 0) or (isinstance(err_max,(int,float))) or (isinstance(iter_max,(int,float))) ) == False:
#         exitFlag = -1
#         raise ValueError('Exit Flag:', exitFlag, "The third and fourth inputs must be positive scaler values")
    
#     while not done:
#         m = (fun(xnew)-fun((xnew + step)))/(-1 * step)
#         b = fun(xnew) - m * xnew
#         xnew = (-1 * b)/m
#         numIter += 1
#         err = abs((xnew - xold) / xnew) * 100
#         xold = xnew
        
#         if err < err_max:
#             exitFlag = 1
#             root = xnew
#             done = True
        
#         if numIter >= iter_max:
#             exitFlag = 0
#             root = xnew
#             done = True
   
#     return root, err, numIter, exitFlag


# Task 5
def funa(x):
    """graphing this function shows roots on -6 to 6 around
    x = -4.71, -3.209, and 1.5707, so I will call bisection
    around all three of these ranges for the solution.
    """
    return x * sin(x) + 3 * cos(x) - x

def funb(x):
    """graphing this function shows roots on -6 to 6 around
    x = -4.493, 0, and 4.493, so I will call bisection and
    secant around all three of these ranges for the solution
    """
    return x * (sin(x) - x * cos(x))

def func(x):
    """graphing this functions shows one root on -6 to 6
    at roughly x = 3.049, so I will call bisection around
    this range for the solution
    """
    return ((x ** 3 - 2 * x ** 2 + 5 * x - 25) / 40)

#Task 5 part A
r1_a = bisection(funa, -7, -4)
r2_a = bisection(funa, -4, -2)
r3_a = bisection(funa, 1, 2)
funa_roots = r1_a[0], r2_a[0], r3_a[0]

#Task 5 part B
r1_b = bisection(funb, -5, -3)
r2_b = secant(funb, .5)
r3_b = bisection(funb, 3, 5)
funb_roots = r1_b[0], r2_b[0], r3_b[0]

#Task 5 part C
r1_c = bisection(func, 0, 5)
func_roots = r1_c[0]

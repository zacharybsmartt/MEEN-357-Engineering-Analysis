from math import *

def bisection(fun,lb,ub,err_max = 1e-6, iter_max = 1000):
    done = False
    #Define global variable root and err and exitFlag
    global root
    global err
    global exitFlag
    
    #writing validations for the numerical inputs#
    if type(lb) == int:
        pass
    elif type(lb) == float:
        pass
    else:
        raise Exception('The lower bound input type is incorrect. Please input an integer value.')
    
    if type(ub) == int:
        pass
    elif type(ub) == float:
        pass
    else:
        raise Exception('The upper bound input type is incorrect. Please input an integer value.')
     
    if type(err_max) == int:
        pass
    elif type(err_max) == float:
        pass
    else:   
        raise Exception('The err_max input type is incorrect. Please input an integer value.')
    
    if type(iter_max) == int:
        pass
    elif type(iter_max) == float:
        pass
    else:
        raise Exception('The iter_max input type is incorrect. Please input an integer value.')
    
    if err_max <= 0:
        raise Exception('Maximum error is either zero or a negative value. Please input a positive value.')
    
    if iter_max <=0:
        raise Exception('Maximum number of iterations is either zero or negative. Please input a positive value.')
    
    ## Account for user input errors
    if lb > ub:
        raise Exception("User Error. Please make ub greater than lb.")
    
    if isnan(fun(lb)) or isnan(fun(ub)) or isinf(fun(lb)) or isinf(fun(ub)):
        exitFlag = -2
        done = True
        
    
    #Check for no roots
    #This would mean that one y value is pos and other is neg
   
    if fun(ub)*fun(lb) > 0:
        done = True
        exitFlag = -1
        
    #Create Xr function
    numIter = 0
    
    while not done:
        numIter+=1
        root = float(ub + lb)/2
        
        if fun(lb)*fun(root) < 0:
            ub = root
    
        else:
            lb = root
        
        #Account for the error estimate
        err = abs((lb-ub)/(ub+lb))*100
        
       
        #Check to make sure error isn't greater than the max error
        if err < err_max:
            done = True
            exitFlag = 1
            
        #Check if numIter are greater than the set number
        if numIter >= iter_max:
            done = True
            exitFlag = 0
            
        
    return root, err, numIter, exitFlag


def secant(fun, ini_guess, err_max = 10e-6, iter_max = 1000):
    
    if not callable(fun):
        exitFlag = -2
        return root, err, numIter, exitFlag

    
    if isinstance(ini_guess,(int,float)) == False:
        exitFlag = -1
        ###############################################
        raise ValueError('Exit Flag:', exitFlag, "The second inpt must be scaler values")#### Fix exit conditions
        ###############################################
        
    if ((err_max < 0) or (iter_max < 0) or (isinstance(err_max,(int,float))) or (isinstance(iter_max,(int,float))) ) == False:
        exitFlag = -1
        ###############################################
        raise ValueError('Exit Flag:', exitFlag, "The third and fourth inputs must be positive scaler values")#### Fix exit conditions
        ###############################################
        
    xold = 100000000000000000000000000000
    xnew = ini_guess
    numIter = 0
    step = 0.001
    
    
    for i in range(iter_max):
        
        m = (fun(xnew)-fun((xnew + step)))/(-1 * step)
        b = fun(xnew) - m * xnew
        xnew = (-1 * b)/m
        numIter += 1
        err = abs((xnew - xold)/xnew) *100
        xold = xnew
        
        if err < err_max:
            exitFlag = 1
            root = xnew
            break
        
        if numIter >= iter_max:
            exitFlag = 0
            root = xnew
            break
   
    
    
    return root, err, numIter, exitFlag

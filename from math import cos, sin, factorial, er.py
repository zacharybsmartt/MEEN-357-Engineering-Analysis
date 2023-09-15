from math import cos, sin, factorial, erf
from operator import xor
from os import error
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from scipy.optimize import root_scalar # for finsing zero 
import numpy as np
x=0


#Task 1
def bisection(f,L,R,toll = 0.001,n= 100):
    flag = -1
    #if L>R: # makse sure directions of var is right
    #    tmep = L
    #    L=R
    #    R=temp

    LR =L
    RR = R
    if callable(f) == False:
        raise TypeError("0 : algorithm terminated due to max iterations being reached-")


    from numpy import sign 
    # look at initla inteval
    flag = -1
    tol = toll*2
    count = 0
    a = True
    while a == True:
        #print(count)
        mid = (L + R)/2 
        #if mid == "null" or mid == "Nan" or mid = "Inf", or mid = null:
        #    l
            #return mid,erf(f(mid)) ,count, flag

        if (count > n):
            flag = 0 #raise Exception("0 : algorithm terminated due to max iterations being reached-")
            print("Exit with:",flag)
            return
        if np.sign(f(L)) == np.sign(f(R)):
            flag = 1 #raise Exception("1 : algorithm terminated due to invalid bracket specification (no root in bracket)")
            print("Exit with:",flag)
            return

        if abs(f(mid)) < tol:
            #print("-1 :algorithm terminated normally (due to error being sufficiently small)")

            return mid,erf(f(mid)) ,count, flag

        elif f(mid)*f(R) < 0: 
            L = mid 
            
        elif f(mid)*f(L) < 0:
            R = mid
        

        count += 1



# -1 :algorithm terminated normally (due to error being sufficiently small)
# 0 : algorithm terminated due to max iterations being reached-
# 1 : algorithm terminated due to invalid bracket specification (no root in bracket)-
# 2 : algorithm terminated due to invalidreturn value from function fun (e.g., NaN, Inf, empty bracket)
        #print((L+R)/2.0)


#Task 4
#Using yourPython functionsfrom Tasks 1-3, find allrootsfor eachofthe following functions on the range  −6≤𝑥≤6:

def y1(x):
    return x*sin(x)+3*cos(x)- x
def y2(x):
    return x - cos(x)
def y3(x):
    return(x**3-2**2+5*x-25)/40

print(bisection(y2, -100, 200))

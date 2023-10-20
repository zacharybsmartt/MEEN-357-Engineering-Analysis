import numpy as np
from math import *

def euler_integrate(fun, t0, y0, tStop, h):
  T, Y = [], []
  t, y = t0, y0
  T.append(x)
  Y.append(y)

  while t < tStop:
    h = min(h, tStop-t)
    phi = F(t, y)
    y = y + phi*h
    Y.append(y)
    t = t+h
    T.append(t)

    return T, Y 


def midpoint_integrate(fun, t0, y0, tStop, h):

    return T, Y 


def RK4_integrate(fun, t0, y0, tStop, h):

    return T, Y 


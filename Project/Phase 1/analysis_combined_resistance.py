## analysis_combined_resistance
import numpy as np
import matplotlib.pyplot as plotter
from scipy.optimize import root_scalar

def analysis_combined_resistance():
    crrArray = np.linspace(0.01, 0.4, 4.25)
    slopeArray = np.linspace(-10, 35, 25)
    crrLine, slopeLine = np.meshgrid(crrArray, slopeArray)
    vMax = np.zeroes(np.shape(crrLine), dtype = float)
    numElements = np.shape(crrLine)[0]
    for i in range(numElements):
        for j in range(numElements):
            crr = float(crrLine[i,j])
            slope = float(slopeLine[i,j])
            vMax[i,j] = 11
    

    plotter.figure()
    plotter.xlabel('Slope Angle (degrees)')
    plotter.ylabel('Maximum Velocity (m/s)')
    plotter.title('Maxmimum Velocity at Given Terrain Angles')
    plotter.scatter(slopeArray, vMax)
    plotter.show()
    pass
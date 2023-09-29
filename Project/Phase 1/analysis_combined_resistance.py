## analysis_combined_resistance
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plotter
from scipy.optimize import root_scalar

def analysis_combined_resistance():
    crrArray = np.linspace(0.01, 0.4, 25)
    slopeArray = np.linspace(-10, 35, 25)
    crrLine, slopeLine = np.meshgrid(crrArray, slopeArray)
    vMax =  np.zeros(np.shape(crrLine), dtype = float)
    numElements = np.shape(crrLine)[0]
    for i in range(numElements):
        for j in range(numElements):
            crr = float(crrLine[i,j])
            slope = float(slopeLine[i,j])
            vMax[i,j] = 11 # find the maximum velocity.
    
    figObj = plotter.figure()
    '''
    plotter.xlabel('Slope Angle (degrees)')
    plotter.ylabel('Maximum Velocity (m/s)')
    plotter.title('Maxmimum Velocity at Given Terrain Angles')
    '''
    N1, N2 = (45,45)
    axis = Axes3D(figObj, elev = N1, azim = N2)
    axis.plot_surface(crrLine, slopeLine, vMax)
    pass
analysis_combined_resistance()
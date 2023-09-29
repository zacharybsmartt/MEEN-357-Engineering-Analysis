####ANALYSIS_COMBINED_TERRAIN####
###IMPORTS###
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from subfunctions import*
import matplotlib.pyplot as plt
from define_rover import*
from scipy.optimize import root_scalar
#from basic_bisection_method import *

#retrieve rover and planet values
rover, planet = rover()
gear_ratio = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
wheel_radius = rover['wheel_assembly']['wheel']['radius']
#Create necessary Crr array
Crr_array = np.linspace(0.01,0.4,25)
#Create slope array of degrees
slope_array_deg = np.linspace(-10,35,25)
#initialize v_max by copying slope_array_deg
v_max = slope_array_deg.copy()
noload_speed = rover['wheel_assembly']['motor']['speed_noload']

Crr, Slope = np.meshgrid(Crr_array,slope_array_deg)

CRR, SLOPE = np.meshgrid (Crr_array, slope_array_deg)
VMAX = np.zeros(np.shape(CRR), dtype = float)
N = np.shape(CRR)[0]

for i in range(N):
    for j in range(N):
        Crr_sample = float(CRR[i,j])
        slope_sample = float(SLOPE[i,j])
        #create function using F_net
        function = lambda omega: F_net(omega,slope_sample,rover,planet,Crr_sample)
        solution = root_scalar(function,method = 'bisect', bracket = [0,noload_speed])
        VMAX[i,j] = (solution.root * wheel_radius) / gear_ratio

figure = plt.figure()
ax = Axes3D(figure, elev = 400, azim = 400)
ax.plot_surface(CRR, SLOPE, VMAX)
ax.set_title("Maximum Rover Speed vs. Crr vs. Terrain Angle")
ax.set_ylabel("Terrain Angle [deg]")
ax.set_zlabel("Maximum Rover Speed [m/s]")
ax.set_xlable("Crr")
plt.show()

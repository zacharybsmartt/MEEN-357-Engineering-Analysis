####ANALYSIS_COMBINED_TERRAIN####
###IMPORTS###
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from subfunctions import*
import matplotlib.pyplot as plt
from define_rover import*
#Need to account to different errors to graph using bisection
from scipy.optimize import root_scalar


#retrieve rover and planet values
rover, planet = rover()
gear_raio = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
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
#initialize VMAX
VMAX = np.zeros(np.shape(CRR), dtype = float)
N = np.shape(CRR)[0]

#LOOP#
for i in range(N):
    for j in range(N):
        Crr_sample = float(CRR[i,j])
        slope_sample = float(SLOPE[i,j])
        #create function using F_net
        function = lambda omega: F_net(omega,slope_sample,rover,planet,Crr_sample)
        try:
            solution = root_scalar(function,method = 'bisect', bracket = [0,noload_speed])
        except ValueError:
            VMAX[i,j] = np.nan
        #if VMAX[i,j] != np.nan:
            #VMAX[i,j] = (solution.root * wheel_radius) / gear_ratio
        VMAX[i,j] = (solution.root * wheel_radius) / gear_ratio
        
####GRAPHING####
fig, ax = plt.subplots(subplot_kw={'projection' : '3d'})
surf = ax.plot_surface(CRR, SLOPE, VMAX)
ax.set_xlabel('Resistance Crr')
ax.set_ylabel("Terrain ANgle [deg]")
ax.set_title("Max Rover Speed vs. Terrain angles vs. Rolling Resistances")
ax.set_zlabel('Max Rover Speed [m/s]')
plt.show()


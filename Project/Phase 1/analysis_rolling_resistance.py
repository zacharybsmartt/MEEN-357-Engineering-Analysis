#####ANALYSIS_ROLLING_RESISTANCE####

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar
from subfunctions import*
from define_rover import*

rover, planet = rover()
#Assume Horizontal Terrain Slope
terrain_slope = 0
#Access the wheel radius
wheel_radius = rover['wheel_assembly']['wheel']['radius']
#Determine the gear ratio
gear_ratio = get_gear_ratio((rover['wheel_assembly']['speed_reducer']))
#Intialize teh Crr_array with the provided values using linspace
Crr_array = np.linspace(0.01,0.4,25)
#Copy Crr_array to v_max in order to maintain same dimensional size for graphing, etc
v_max = Crr_array.copy()
#Access the noload speed of the rover
noload_speed = rover['wheel_assembly']['motor']['speed_noload']
#Loop through using the F_net function using labmda of omega, which the bisection method will be used to solve
#The solutions of the function will be used to compute v_max at each index of the list
for i in range(len(v_max)):
    function = lambda omega: F_net(omega,terrain_slope,rover,planet,float(Crr_array[i]))
    #Use Bisection root scalar method, assume we start from zero. Must have function, method, and bracket defined to use function
    solution = root_scalar(function,method = 'bisect',bracket = [0,noload_speed])
    v_max[i] = (solution.root * wheel_radius) / gear_ratio

#Graph
plt.plot(Crr_array,v_max)
plt.xlabel("Rolling Resistance Coeff")
plt.ylabel('Max Rover Speed [m/s]')
#plt.show()

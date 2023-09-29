############ANALYSIS TERRAIN SLOPE##########

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar
from subfunctions import*
from define_rover import*

rover, planet = rover() ####LOCATION#####
#Assumed Coefficient Crr#
Crr = 0.2
#Initialize slope angles in degress
slope_array_deg = np.linspace(-10,35,25)
#Determine gear ratios, wheel radius, rover, and planet
gear_ratio = get_gear_ratio(rover["wheel_assembly"]["speed_reducer"])
wheel_radius = rover["wheel_assembly"]["wheel"]["radius"]
#rover, planet = rover() ####LOCATION#####
#Make a maximum velocity list as the same size of teh slope angles list along with the no load speed
no_load_speed = rover['wheel_assembly']['motor']['speed_noload']
v_max = slope_array_deg.copy()
#Determine the Rover Maximum Velocity, Which Would Be When F_net is Equal to Zero
for i in range(len(v_max)):
    function = lambda omega: F_net(omega,float(slope_array_deg[i]),rover,planet,Crr)
    #Use Bisection root scalar method, assume we start from zero. Must have function, method, and bracket defined to use function
    solution = root_scalar(function,method='bisect',bracket = [0,no_load_speed])
    v_max[i] = (solution.root * wheel_radius)/ gear_ratio
    
#Produce Graphs#
plt.plot(slope_array_deg,v_max)
plt.xlabel("Terrain Angle [deg]")
plt.ylabel("Max Rover Speed [m/s]")
plt.title("Terrain Angle [deg] vs. Max Rover Speed [m/s]")
#plt.show()




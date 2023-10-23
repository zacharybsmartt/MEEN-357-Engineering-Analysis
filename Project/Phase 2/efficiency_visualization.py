import numpy as np
import matplotlib.pyplot as plt
from define_experiment import*
from define_rover import*
from scipy.interpolate import*


#retrieve rover and planet from the defined rover#
rover, planet = rover()
#Create the efficiency function#
effcy_fun = interp1d(rover["wheel_assembly"]["motor"]["effcy_tau"],rover["wheel_assembly"]["motor"]["effcy"], kind = "cubic")
#Determine the rover efficiency using linspace of amin and amax
rover_efficiency = np.linspace(np.amin(rover["wheel_assembly"]["motor"]["effcy_tau"]),np.amax(rover["wheel_assembly"]["motor"]["effcy_tau"]),100)
#Calculate the efficiency using the efficiency function of the rover efficiency
efficiency = effcy_fun(rover_efficiency)



#PLOTTING GRAPH#
#Plot the line along withe the knwon data points 
plt.plot(rover_efficiency,efficiency,color='black',label="Cubic Interpolation")
plt.plot(rover['wheel_assembly']['motor']['effcy_tau'],rover['wheel_assembly']['motor']['effcy'],'b*',label='known data')
plt.title("Motor Torque vs Efficiency")
plt.xlabel("Torque (N*m)")
plt.ylabel("Efficiency")
plt.legend(loc="upper right")
#show plot
plt.show()

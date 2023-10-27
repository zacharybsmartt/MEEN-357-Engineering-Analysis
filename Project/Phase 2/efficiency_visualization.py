import numpy as np
import matplotlib.pyplot as plt
from define_experiment import *
from define_rover import *
from scipy.interpolate import interp1d

# Retrieve rover and planet data from defined sources
rover, planet = rover()

# Create the efficiency function using cubic interpolation
effcy_fun = interp1d(
    rover["wheel_assembly"]["motor"]["effcy_tau"],
    rover["wheel_assembly"]["motor"]["effcy"],
    kind="cubic"
)

# Determine the rover efficiency using a linspace of amin and amax
rover_efficiency = np.linspace(
    np.amin(rover["wheel_assembly"]["motor"]["effcy_tau"]),
    np.amax(rover["wheel_assembly"]["motor"]["effcy_tau"]),
    100
)

# Calculate the efficiency using the efficiency function of the rover efficiency
efficiency = effcy_fun(rover_efficiency)

# PLOTTING GRAPH
# Plot the cubic interpolation line along with the known data points
plt.plot(rover_efficiency, efficiency, color='black', label="Cubic Interpolation")
plt.plot(
    rover['wheel_assembly']['motor']['effcy_tau'],
    rover['wheel_assembly']['motor']['effcy'],
    'b*',
    label='known data'
)
plt.title("Motor Torque vs Efficiency")
plt.xlabel("Torque (N*m)")
plt.ylabel("Efficiency")
plt.legend(loc="upper right")

# Show the plot
plt.show()

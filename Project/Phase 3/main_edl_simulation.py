"""###########################################################################
# This file is the top-level script for running a simulation of the EDL system.
# It loads the appropriate info into the workspace and calls simulate_edl.
# After returning from simulate_edl, it graphs some of the simulation
# trajectory data.
#
#   Created by: Former Marvin Numerical Methods Engineering Team
#   Last Modified: 28 October 2023
###########################################################################"""

import numpy as np
import matplotlib.pyplot as plt
from define_edl_system import *
from subfunctions_EDL import *
from define_planet import *
from define_mission_events import *

# *************************************
# load dicts that define the EDL system (includes rover), planet,
# mission events and so forth.
edl_system = define_edl_system_1()
mars = define_planet()
mission_events = define_mission_events()


# Overrides what might be in the loaded data to establish our desired
# initial conditions
edl_system['altitude'] = 11000    # [m] initial altitude
edl_system['velocity'] = -590     # [m/s] initial velocity
edl_system['parachute']['deployed'] = True   # our parachute is open
edl_system['parachute']['ejected'] = False   # and still attached
edl_system['rover']['on_ground'] = False # the rover has not yet landed
edl_system['parachute']['diameter'] = 18.0

tmax = 2000   # [s] maximum simulated time

# the simulation. changing last argument to false turns off message echo
[t, Y, edl_system] = simulate_edl(edl_system, mars, mission_events, tmax, True)

# visualize the simulation results
plot1 = plt.figure(0)
fig, axs = plt.subplots(7) 
plt.tight_layout()
axs[0].plot(t, Y[0, :])
axs[0].set_title('velocity vs. time', fontsize=10)
axs[0].grid()
axs[1].plot(t,Y[1, :])
axs[1].set_title('altitude vs. time', fontsize=10)
axs[1].grid()
axs[2].plot(t,Y[2, :])
axs[2].set_title('fuel mass vs. time', fontsize=10)
axs[2].grid()
axs[3].plot(t,Y[3, :])
axs[3].set_title('speed error integral vs. time', fontsize=10)
axs[3].grid()
axs[4].plot(t,Y[4, :])
axs[4].set_title('position error integral vs. time', fontsize=10)
axs[4].grid()
axs[5].plot(t,Y[5, :])
axs[5].set_title('velocity of rover relative to sky crane vs. time', fontsize=10)
axs[5].grid()
axs[6].plot(t,Y[6, :])
axs[6].set_title('position of rover relative to sky crane vs. time', fontsize=10)
axs[6].grid()

plot2 = plt.figure(1)
fig2, axs2 = plt.subplots(2)
plt.tight_layout()
sky_crane_hover_pos = Y[1, :]
sky_crane_speed = Y[0, :]
ignore_indices = sky_crane_hover_pos>2*20
sky_crane_hover_pos[ignore_indices] = np.NaN 
sky_crane_speed[ignore_indices] = np.NaN  
axs2[0].plot(t,sky_crane_speed)
axs2[0].set_title('speed of sky crane vs. time')
axs2[0].grid()
axs2[1].plot(t,sky_crane_hover_pos)
axs2[1].set_title('position of sky crane vs. time')
axs2[1].grid()


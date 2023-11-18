import numpy as np
import matplotlib.pyplot as plt
from define_edl_system import *
from subfunctions_EDL import *
from define_planet import *
from define_mission_events import *

mars = define_planet()
edl_system = define_edl_system_1()
mission_events = define_mission_events()

t_terminated = []
parachute_diameter = []
rover_speed = []
rover_landing_success = []

for i in range(28, 39):
    edl_system['altitude'] = 11000  # [m] initial altitude
    edl_system['velocity'] = -590  # [m/s] initial velocity
    edl_system['rocket']['on'] = False  # rockets off
    edl_system['parachute']['deployed'] = True  # our parachute is open
    edl_system['parachute']['ejected'] = False  # and not ejected
    edl_system['heat_shield']['ejected'] = False  # heat shield not ejected
    edl_system['sky_crane']['on'] = False  # skycrane inactive
    edl_system['speed_control']['on'] = False  # speed controller off
    edl_system['position_control']['on'] = False  # position controller off
    edl_system['rover']['on_ground'] = False  # the rover has not yet landed
    edl_system['parachute']['diameter'] = i/2 # range of parachute diameters we are simulating in 
    tmax = 2000 

    # simulate and turn off the annoying echo that makes your computer fans go brrr
    [t, Y, edl_system] = simulate_edl(edl_system, mars, mission_events, tmax, False)
    t_termination = np.array

    # graphed variables
    parachute_diameter += [i/2]                         
    t_terminated += [t[-1]]                             
    rover_speed += [Y[0, -1]]                          
    if edl_system['velocity'] < -1:
        rover_landing_success += [0]
    else:
        rover_landing_success += [1]


# Actualy graphing the solution
fig, axis = plt.subplots(3)
fig.subplots_adjust(hspace=1.2)
axis[0].plot(parachute_diameter, t_terminated)
axis[0].set_xlabel('Parachute diameter (m)')
axis[0].set_ylabel('t terminated (s)')
axis[0].set_title('Time Terminated vs. Parachute diameter')
axis[0].grid()

axis[1].plot(parachute_diameter, rover_speed)
axis[1].set_title('Rover speed vs Parachute diameter')
axis[1].set_xlabel('Parachute_diameter (m)')
axis[1].set_ylabel('rover Speed (m/s)')
axis[1].grid()

axis[2].plot(parachute_diameter, rover_landing_success)
axis[2].set_title('Landing Success vs Parachute diameter')
axis[2].set_xlabel('Parachute_diameter (m)')
axis[2].set_ylabel('Landing Success (Y-1/N-0)')
axis[2].grid()

plt.show()

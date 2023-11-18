import numpy as np
import matplotlib.pyplot as plt
from define_edl_system import *
from subfunctions_EDL import *
from define_planet import *
from define_mission_events import *

rover_speed = []
landing_success = []
simulation_time = []
diameter = []
mars = define_planet()
edl_system = define_edl_system_1()
mission_events = define_mission_events()


for i in range(28, 39):
    edl_system["altitude"] = 11000 # [m] initial altitude
    edl_system["velocity"] = -590 # [m/s] initial velocity
    edl_system["rocket"]["on"] = False # rockets off
    edl_system["parachute"]["deployed"] = True # our parachute is open
    edl_system["parachute"]["ejected"] = False # and not ejected
    edl_system["heat_shield"]["ejected"] = False # heat shield not ejected
    edl_system["sky_crane"]["on"] = False # skycrane inactive
    edl_system["speed_control"]["on"] = False # speed controller off
    edl_system["position_control"]["on"] = False # position controller off
    edl_system["rover"]["on_ground"] = False # the rover has not yet landed
    edl_system["parachute"]["diameter"] = i / 2 # range of parachute diameters we are simulating in 
    tmax = 2000 

    # simulate and turn off the annoying echo that makes your computer fans go brrr
    [t, Y, edl_system] = simulate_edl(edl_system, mars, mission_events, tmax, False)

    # things to graph
    diameter += [i / 2]                         
    simulation_time += [t[-1]]                             
    rover_speed += [Y[0, -1]]

    if edl_system["velocity"] < -1:
        landing_success += [0]
    else:
        landing_success += [1]


# Graphing time
fig, axis = plt.subplots(3)
axis[0].plot(diameter, simulation_time)
axis[0].set_title("Simulated time vs. Parachute diameter")
axis[0].set_xlabel("parachute diameter (m)")
axis[0].set_ylabel("t simulated (s)")
axis[0].grid()

axis[1].plot(diameter, rover_speed)
axis[1].set_title("Rover speed vs Parachute diameter")
axis[1].set_xlabel("diameter (m)")
axis[1].set_ylabel("rover speed (m/s)")
axis[1].grid()

axis[2].plot(diameter, landing_success)
axis[2].set_title("Landing success vs Parachute diameter")
axis[2].set_xlabel("diameter (m)")
axis[2].set_ylabel("landing success (Y: 1/N: 0)")
axis[2].grid()

plt.show()

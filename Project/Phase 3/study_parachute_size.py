import matplotlib.pyplot as plt
import numpy as np
from define_rovers import *
from define_edl_system import *
from define_planet import *
from define_mission_events import *
from subfunctions_EDL import *
from redefine_edl_system import *

edl_system = define_edl_system_1()
rocket = define_edl_system_1()
heat_shield = define_edl_system_1()
sky_crane = define_edl_system_1()
speed_controller = define_edl_system_1()
position_controller = define_edl_system_1()
planet = define_planet()
mission_events = define_mission_events()
tmax = 2000

diameters = np.arange(14,19.5,0.5)
times = np.array([])
v = np.array([])
fail = np.array([])

for d in diameters:
    edl_system = redefine_edl_system(edl_system)
    edl_system['parachute']['diameter'] = d
    [t, Y, edl_system] = simulate_edl(edl_system, planet, mission_events, tmax, True)
    times = np.append(times,t[len(t)-1]) 
    v = np.append(np.abs(v),Y[0][len(Y[0])-1])

    if Y[0][len(Y[0])-1] <= edl_system['sky_crane']['danger_speed']:
        fail = np.append(fail,0)
    
    else:
        fail = np.append(fail,1)

plt.subplot(3,1,1)
plt.plot(diameters, times)
plt.xlabel('Diameter (m)')
plt.ylabel('Time (s)')
plt.title('Simulated Time vs. Diameter')


plt.subplot(3,1,2)
plt.plot(diameters, v)
plt.xlabel('Diameter (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Landing Velocity vs. Diameter')


plt.subplot(3,1,3)
plt.plot(diameters,fail)
plt.xlabel('Diameter (m)')
plt.ylabel('Success/Failure')
plt.title('Success and Failure vs. Diameter')


plt.show()

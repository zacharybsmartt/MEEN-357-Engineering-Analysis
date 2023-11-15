"""###########################################################################
#   EDL system dictionary definitions for project Phase 3.  Run this file to 
# instantiate an edl_system dict.
#
#   Created by: Former Marvin Numerical Methods Engineering Team
#   Last Modified: 08 March 2022
###########################################################################"""

import numpy as np
from define_rovers import *

def define_edl_system_1():
           
    # parachute dict.  Includes physical definition and state information.
    # proper use is to set 'deployed' to False when parachute is ejected,
    # (as well as 'ejected' to True) in case someone checks deployed but not
    # ejected.
    # Note: we are starting the simulation from the point at which the 
    # parachute is first deployed to keep things simpler.  
    parachute = {'deployed' : True,  # true means it has been deployed but not ejected
                 'ejected' : False,  # true means parachute no longer is attached to system
                 'diameter' : 16.25, # [m] (MSL is about 16 m)
                 'Cd' : 0.615,       # [-] (0.615 is nominal for subsonic)
                 'mass' : 185.0}     # [kg] (this is a wild guess -- no data found)
        
    # Rocket dict.  This defines a SINGLE rocket.
    rocket = {'on' : False,
              'structure_mass' : 8.0,                 # [kg] everything not fuel
              'initial_fuel_mass' : 230.0,            # [kg]  230.0
              'fuel_mass' : 230.0,                    # [kg] current fuel mass (<= initial)
              'effective_exhaust_velocity' : 4500.0,  # [m/s]
              'max_thrust' : 3100.0,                  # [N]  
              'min_thrust' : 40.0}                    # [N]
        
    speed_control = {'on' : False,             # indicates whether control mode is activated
                     'Kp' : 2000,              # proportional gain term
                     'Kd' : 20,                # derivative gain term
                     'Ki' : 50,                # integral gain term
                     'target_velocity' : -3.0} # [m/s] desired descent speed
        
    position_control = {'on' : False,            # indicates whether control mode is activated
                        'Kp' : 2000,             # proportional gain term
                        'Kd' : 1000,             # derivative gain term
                        'Ki' : 50,               # integral gain term
                        'target_altitude' : 7.6} # [m] needs to reflect the sky crane cable length
        
    # This is the part that lowers the rover onto the surface
    sky_crane = {'on' : False,            # true means lowering rover mode
                 'danger_altitude' : 4.5, # [m] altitude at which considered too low for safe rover touch down
                 'danger_speed' : -1.0,   # [m/s] speed at which rover would impact to hard on surface
                 'mass' : 35.0,           # [kg]
                 'area' : 16.0,           # [m^2] frontal area for drag calculations
                 'Cd' : 0.9,              # [-] coefficient of drag
                 'max_cable' : 7.6,       # [m] max length of cable for lowering rover
                 'velocity' : -0.1}       # [m] speed at which sky crane lowers rover
        
    # Heat shield dict
    heat_shield = {'ejected' : False,  # true means heat shield has been ejected from system
                   'mass' : 225.0,     # [kg] mass of heat shield
                   'diameter' : 4.5,   # [m]
                   'Cd' : 0.35}        # [-]
        
    rover = define_rover_4()
        
    # pack everything together and clean up subdicts.
    edl_system = {'altitude' : np.NaN,   # system state variable that is updated throughout simulation
                  'velocity' : np.NaN,   # system state variable that is updated throughout simulation
                  'num_rockets' : 8,     # system level parameter
                  'volume' :150,         # system level parameter
                  'parachute' : parachute,
                  'heat_shield' : heat_shield,
                  'rocket' : rocket,
                  'speed_control' : speed_control,
                  'position_control' : position_control,
                  'sky_crane' : sky_crane,
                  'rover' : rover}
        
    #del parachute, rocket, speed_control, position_control, sky_crane
    #del heat_shield, rover
    return edl_system
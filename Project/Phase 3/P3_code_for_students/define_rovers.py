"""###########################################################################
#   This file initializes a rover structure for testing/grading
#
#   Created by: Former Marvin Numerical Methods Engineering Team
#   Last Modified: 28 October 2023
###########################################################################"""

import numpy as np

def define_rover_1():
    # Initialize Rover dict for testing
    wheel = {'radius':0.30,
             'mass':1}
    speed_reducer = {'type':'reverted',
                     'diam_pinion':0.04,
                     'diam_gear':0.07,
                     'mass':1.5}
    motor = {'torque_stall':170,
             'torque_noload':0,
             'speed_noload':3.80,
             'mass':5.0}
    
    # phase 2 add ##############################
    motor['effcy_tau'] = np.array([0, 10, 20, 40, 70, 165])
    motor['effcy']     = np.array([0,.55,.75,.71,.5, .05])
    #############################################
        
    chassis = {'mass':659}
    science_payload = {'mass':75}
    power_subsys = {'mass':90}
    
    wheel_assembly = {'wheel':wheel,
                      'speed_reducer':speed_reducer,
                      'motor':motor}
    
    rover = {'wheel_assembly':wheel_assembly,
             'chassis':chassis,
             'science_payload':science_payload,
             'power_subsys':power_subsys}
    
    # planet = {'g':3.72}
    
    # return only the rover now
    return rover #, planet

def define_rover_2():
    # Initialize Rover dict for testing
    wheel = {'radius':0.30,
             'mass':2} 
    speed_reducer = {'type':'reverted',
                     'diam_pinion':0.04,
                     'diam_gear':0.06,
                     'mass':1.5}
    motor = {'torque_stall':180,
             'torque_noload':0,
             'speed_noload':3.70,
             'mass':5.0}
    
    
    # phase 2 add ##############################
    motor['effcy_tau'] = [0, 10, 20, 40, 75, 165]
    motor['effcy']     = [0,.60,.75,.73,.55, .05]
    #############################################
    
    
    chassis = {'mass':659}
    science_payload = {'mass':75}
    power_subsys = {'mass':90}
    
    wheel_assembly = {'wheel':wheel,
                      'speed_reducer':speed_reducer,
                      'motor':motor}
    
    rover = {'wheel_assembly':wheel_assembly,
             'chassis':chassis,
             'science_payload':science_payload,
             'power_subsys':power_subsys}
    
    # planet = {'g':3.72}
    
    # return only the rover now
    return rover #, planet

def define_rover_3():
    # Initialize Rover dict for testing
    wheel = {'radius':0.30,
             'mass':2} 
    speed_reducer = {'type':'standard',
                     'diam_pinion':0.04,
                     'diam_gear':0.06,
                     'mass':1.5}
    motor = {'torque_stall':180,
             'torque_noload':0,
             'speed_noload':3.70,
             'mass':5.0}
    
    
    # phase 2 add ##############################
    motor['effcy_tau'] = [0, 10, 20, 40, 75, 165]
    motor['effcy']     = [0,.60,.75,.73,.55, .05]
    #############################################
    
    
    chassis = {'mass':659}
    science_payload = {'mass':75}
    power_subsys = {'mass':90}
    
    wheel_assembly = {'wheel':wheel,
                      'speed_reducer':speed_reducer,
                      'motor':motor}
    
    rover = {'wheel_assembly':wheel_assembly,
             'chassis':chassis,
             'science_payload':science_payload,
             'power_subsys':power_subsys}
    
    # planet = {'g':3.72}
    
    # return only the rover now
    return rover #, planet


def define_rover_4():
    # Initialize Rover dict for testing
    wheel = {'radius':0.20,
             'mass':2} 
    speed_reducer = {'type':'reverted',
                     'diam_pinion':0.04,
                     'diam_gear':0.06,
                     'mass':1.5}
    motor = {'torque_stall':165,
             'torque_noload':0,
             'speed_noload':3.85,
             'mass':5.0}
    
    # phase 2 add ##############################
    motor['effcy_tau'] = np.array([0, 10, 20, 40, 75, 170])
    motor['effcy']     = np.array([0,.60,.75,.73,.55, .05])
    #############################################
    
    
    chassis = {'mass':674}
    science_payload = {'mass':80}
    power_subsys = {'mass':100}
    
    wheel_assembly = {'wheel':wheel,
                      'speed_reducer':speed_reducer,
                      'motor':motor}
    
    rover = {'wheel_assembly':wheel_assembly,
             'chassis':chassis,
             'science_payload':science_payload,
             'power_subsys':power_subsys}
    
    # planet = {'g':3.72}
    
    # return only the rover now
    return rover #, planet
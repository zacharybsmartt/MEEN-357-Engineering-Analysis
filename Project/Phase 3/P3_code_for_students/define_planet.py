"""###########################################################################
#   This file contains planet dict definitions
#
#   Created by: Former Marvin Numerical Methods Engineering Team
#   Last Modified: 22 October 2021
###########################################################################"""

import numpy as np

def define_planet():

    high_altitude = {'temperature' : lambda altitude: -23.4 - 0.00222*altitude, # [C]
                     'pressure' : lambda altitude: 0.699*np.exp(-0.00009*altitude)} # [KPa]
                                                                
    low_altitude = {'temperature' : lambda altitude: -31 - 0.000998*altitude, # [C]
                    'pressure' : lambda altitude: 0.699*np.exp(-0.00009*altitude)} # [KPa]
    
    density = lambda temperature, pressure: pressure/(0.1921*(temperature+273.15)) # [kg/m^3]
    
    mars = {'g' : -3.72,   # m/s^2]
            'altitude_threshold' : 7000, # [m]
            'low_altitude' : low_altitude, 
            'high_altitude' : high_altitude,
            'density' : density}
    
    #del high_altitude, low_altitude, density
    return mars
                                                            
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 23:18:09 2021

@author: dallaire
"""

def redefine_edl_system(edl_system):
    
    edl_system['altitude'] = 11000
    edl_system['velocity'] = -590

    edl_system['num_rockets'] = 8 # number of rockets in our system
    edl_system['volume'] = 150    # [m^3] volume of air displaced by EDL system
     
    edl_system['parachute']['deployed'] = True
    edl_system['parachute']['ejected'] = False
    
    edl_system['rocket']['on'] = False

    edl_system['sky_crane']['on'] = False
    
    edl_system['heat_shield']['ejected'] = False
    
    edl_system['position_control']['on'] = False
    
    edl_system['rover']['on_ground'] = False
    
    rover = edl_system['rover']
    
    rover.pop('velocity', None)
    rover.pop('position', None)
    rover.pop('telemetry', None)
    
    edl_system['rover'] = rover
    del rover

    
    rocket = edl_system['rocket']
    rocket.pop('control', None)
    
    edl_system['rocket'] = rocket
    del rocket
    
    return edl_system







    


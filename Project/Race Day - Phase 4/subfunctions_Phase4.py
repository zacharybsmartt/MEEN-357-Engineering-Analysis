#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 21:06:00 2021

@author: Marvin Mechanical Engineering Team
Last Modified: 30 March 2022
"""

import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from statistics import mean

def get_mass_rover(rover):

    # Computes the mass of the rover defined in rover field of the edl system 
    # struct. Assumes that the rover is defined as a dict corresponding to 
    # the specification of project Phase 1.
    
    m = 6*(rover['wheel_assembly']['motor']['mass'] + 
           rover['wheel_assembly']['speed_reducer']['mass'] + 
           rover['wheel_assembly']['wheel']['mass']) + rover['chassis']['mass'] + rover['science_payload']['mass'] + rover['power_subsys']['mass'] + rover['power_subsys']['battery']['mass']
    
    return m

def get_mass_rockets(edl_system):

    # Returns the curret total mass of all rockets on the edl system. 

    m = edl_system['num_rockets']*(edl_system['rocket']['structure_mass'] + edl_system['rocket']['fuel_mass'])

    return m

def get_mass_edl(edl_system):

    # Returns the total current mass of the edl system 
    
    m = int(not(edl_system['parachute']['ejected']))*edl_system['parachute']['mass'] + \
        int(not(edl_system['heat_shield']['ejected']))*edl_system['heat_shield']['mass'] + \
            get_mass_rockets(edl_system) + edl_system['sky_crane']['mass'] + get_mass_rover(edl_system['rover'])
        
    return m

def get_local_atm_properties(planet, altitude):
    
    # get_local_atm_properties
    #
    # Returns local atmospheric properties at a given altitude. 
    #
    # Usage:
    #  density = get_local_atm_properties(planet, altitude) returns the
    #  atmospheric density in kg/m^3. Assumed altitude is specified in meters.
    #
    #  [density, temperature] = get_local_atm_properties(planet, altitude) also
    #  returns the local temperature in C.
    #
    #  [density, temperature, pressure] = get_local_atm_properties(planet, altitude)
    #  also returns the local pressure in KPa.
    #
    # Note: this function is NOT vectorized. It will not accept a vector of
    # altitudes.
    
    if altitude > planet['altitude_threshold']:
       temperature = planet['high_altitude']['temperature'](altitude) 
       pressure = planet['high_altitude']['pressure'](altitude)
    else:
       temperature = planet['low_altitude']['temperature'](altitude) 
       pressure = planet['low_altitude']['pressure'](altitude)    
    
    density = planet['density'](temperature, pressure) 
    
    return density, temperature, pressure

def get_gear_ratio(speed_reducer):
    """
    Inputs:  speed_reducer:  dict      Data dictionary specifying speed
                                        reducer parameters
    Outputs:            Ng:  scalar    Speed ratio from input pinion shaft
                                        to output gear shaft. Unitless.
    """
    
    # Check that the input is a dict
    if type(speed_reducer) != dict:
        raise Exception('Input must be a dict')
    
    # Check 'type' field (not case sensitive)
    if speed_reducer['type'].lower() != 'reverted':
        raise Exception('The speed reducer type is not recognized.')
    
    # Main code
    d1 = speed_reducer['diam_pinion']
    d2 = speed_reducer['diam_gear']
    
    Ng = (d2/d1)**2
    
    return Ng

def get_cost_rover(rover):
    # get_cost_rover
    #
    # Computes the cost of the rover. Requires a valid rover struct as input.
    #
    #
    
    # Wheel cost as a function of wheel radius.
    # accounts for the fact that there are six wheels.
    wheel_radius = rover['wheel_assembly']['wheel']['radius']
    if wheel_radius > 0.5:
        cost_wheels = 90000
    else:
        cost_wheels = 0.5e7*(wheel_radius**3-0.1**3)+1e4

    
    cost_battery = rover['power_subsys']['battery']['cost']
    
    # Speed reducer cost as a function of the gear diameter.
    # Accounts for the fact that there are six speed reducers.
    d2 = rover['wheel_assembly']['speed_reducer']['diam_gear']
    cost_d2      = 5e7*(d2**2-0.04**2)
    
    # remember that there are six motors
    cost_motor = 6*rover['wheel_assembly']['motor']['cost']
    
    # Chassis cost as a function of strength and material quantity
    
    cost_chassis=(rover['chassis']['specific_strength']/100)**2*1000*rover['chassis']['mass']+50000
    
    
    total_cost = cost_wheels+cost_battery+cost_d2+cost_motor+cost_chassis
    
    return total_cost

def get_cost_edl(edl_system):
    # get_cost_edl
    #
    # Computes the cost of the edl system in $US. Takes a valid edl_system
    # struct as input.
    
    
    cost_rover = get_cost_rover(edl_system['rover'])
    
    # A simlistic model based on mass of solid rocket fuel. Not terrible to a
    # first order I suppose.
    cost_fuel = edl_system['rocket']['initial_fuel_mass']*edl_system['rocket']['cost_per_kg']
    
    # A simplistic model based on cost proportional to the area defined by the
    # parachute diameter. The real area of material used is much greater than
    # this area, so it isn't really a material-proportional model. 
    cost_parachute = edl_system['parachute']['cost_per_A']*np.pi*(edl_system['parachute']['diameter']/2)**2
    
    # add up everything and get out of here
    total_cost = cost_rover + cost_fuel + cost_parachute
    
    return total_cost

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

def define_rover():
    # Initialize Rover dict 
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
    motor['effcy_tau'] = np.array([0, 10, 20, 40, 75, 165])
    motor['effcy']     = np.array([0,.60,.75,.73,.55, .05])
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
    
    # need to add Phase 4 information still
    
    return rover

def define_edl_system():
           
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
                 'cost_per_A' : 1e3, # [$US/m^2] cost proportional to parachute area
                 'mass' : 185.0}     # [kg] (this is a wild guess -- no data found)
        
    # Rocket dict.  This defines a SINGLE rocket.
    rocket = {'on' : False,
              'structure_mass' : 8.0,                 # [kg] everything not fuel
              'initial_fuel_mass' : 230.0,            # [kg]  230.0
              'cost_per_kg' : 1500,                   # [$US/kg] rocket cost per kg of fueld
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
        
    rover = define_rover()
        
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

def define_mission_events():
        
    mission_events = {'alt_heatshield_eject' : 8000,
                      'alt_parachute_eject' : 900,
                      'alt_rockets_on' : 1800,
                      'alt_skycrane_on' : 7.6}
    
    return mission_events

def define_batt_pack(edl_system, battery_type, num_modules):
# define_batt_pack
#
# Determines the attributes of the battery pack and returns a
# properly define battery struct.
#
# INPUTS:
#    edl_system
#    type    A string defining the type of battery technology being used.
#           Valid choices are: 
#               NiCd - Nickel Cadmium - module from AA cells
#               NiMH - Nickel Metal Hydride - module from sub-C cells
#               LiFePO4 - Lithium Iron Phosphate - Tenergy 31382
#               PbAcid-1 - Lead Acid - UPG D5722
#               PbAcid-2 - Lead Acid - UPG UB-GC2
#               PbAcid-3 - Lead Acid - Powersonic PS-12180NB
#
#   num_modules   A positive scalar integer indicating the number of
#               individual 36V modules used of the specified type of
#               battery. (Batteries come in discrete cells that you connect
#               in series to achieve a desired voltage level. Then you
#               bundle several of these in parallel to achieve the desired
#               energy storage capacity. So in our case, one "pack"
#               consists of several "modules", which itself consists of
#               several "cells".) 
#
# OUTPUTS
#   battery     A struct defining the battery pack. Includes fields for
#               cost [$], mass [kg], capacity [J], type [string], and
#               num_modules [integer>0].
#
# 
# Notes: 
#   1) I am using the term 'module' in this context to refer to several
#   cells in series as required to create 36V across the terminals. For
#   example, one would require 18 lead-acid cells and 36 NiMH cells to
#   achieve 36V. I could have defined this in terms of the individual
#   battery cells. However, rather than having you guys (ya'all?) worry
#   about volatage requirements and all that noise, I've simplified things
#   to be based in 36V increments. 
#
#   2) The 36V assumption is somewhat arbitrary on my part. Many DC motors
#   of the size we'll need do run at 36V, but they often also will run at
#   24V or 48V (and higher). To add voltage requirements into the mix would
#   complicate the problem without really adding to your learning as it
#   pertains to MEEN 357.
#


    if num_modules % 1 != 0 or num_modules <= 0:
        raise Exception('define_batt_pack: num_modules must be a positive integer')
    
    if battery_type.lower() == 'LiFePO4'.lower():
        mass_per_module = 3.4860        # kg
        Joules_per_module = 0.9072e5   # Joules (7 AHr @ 36 V)
        cost_per_module = 2.25e5          # $ (retail)
    elif battery_type.lower() == 'NiMH'.lower():
        # 3x Tenergy 12V module made from sub-C cell NiMH batteries
        # 5000 mAhr per module
        mass_per_module = 2.1630        # kg           
        Joules_per_module = 0.6480e5    # Joules (5000 mAhr @ 36V)
        cost_per_module = 125000        # $ (retail reflecting quantity discount)
    elif battery_type.lower() == 'NiCD'.lower():
        # 3x 12V module made from AA cell NiCD batteries
        # 700 mAHr capacity per module
        mass_per_module = 0.669        # kg
        Joules_per_module = 0.0906e5   # Joules (700 mAhr @ 36V)
        cost_per_module = 25000           # $ (retail reflecting quantity discount)
    elif battery_type.lower() == 'PbAcid-1'.lower():
        # 3x UPG D5722 Sealed Lead Acid
        mass_per_module = 30        # kg
        Joules_per_module = 4.38e5    # Joules (35 AHr @ 36V)
        cost_per_module = 150000           # $ (full retail)
    elif battery_type.lower() == 'PbAcid-2'.lower():
        # UPG UB-GC2 Golf Cart/AGM Battery - Sealed Lead Acid
        # factored below at x6 since each of the above is only 6V DC
        mass_per_module   = 60         # kg
        Joules_per_module = 8.76e5     # Joules (200 AHr @ 36V)
        cost_per_module   = 21000          # $ (full retail)
    elif battery_type.lower() == 'PbAcid-3'.lower():
        # 3x Powersonic PS-12180NB 12V 18Ah Sealed Lead Acid Battery
        mass_per_module   = 45         # kg
        Joules_per_module = 6.57e5    # Joules (18 AHr @ 36V)
        cost_per_module   = 170000           # $ (full retail)
    else:
        raise Exception('define_batt_pack: unknown battery battery_type')

    
    battery = {'battery_type' : battery_type,
               'num_modules' : num_modules,
               'mass' : mass_per_module*num_modules,
               'cost' : cost_per_module*num_modules,
               'capacity' : Joules_per_module*num_modules}
    
    edl_system['rover']['power_subsys']['battery'] = battery
    
    return edl_system

def define_chassis(edl_system, chassis_type):
    # define_batt_pack
    #
    # Determines the attributes of the rover motor. Returns edl_system with
    # properly modified motor.
    #
    # INPUTS:
    #    edl_system and type of chassis:
    #       steel, magnesium, carbon fiber
    #
    # OUTPUTS
    #   edl_system
    # 
    # Notes: 
    #   

    
    chassis=edl_system['rover']['chassis']
    
    if chassis_type.lower() == 'steel'.lower():
        chassis['type'] = 'steel'
        chassis['specific_strength'] = 100
        
    elif chassis_type.lower() == 'magnesium'.lower():
        chassis['type'] = 'magnesium'
        chassis['specific_strength'] = 250    
        
    elif chassis_type.lower() == 'carbon'.lower():
        chassis['type'] = 'carbon'
        chassis['specific_strength'] = 1000
        
    else:
        raise Exception('input not recognized') 

      
    chassis['strength'] = chassis['mass']* chassis['specific_strength']
    
    edl_system['rover']['chassis'] = chassis
    
    return edl_system

def define_motor(edl_system, motor_type):

    # define_motor
    #
    # Determines the attributes of the rover motor. Returns edl_system with
    # properly modified motor.
    #
    # INPUTS:
    #    edl_system and type of motor:
    #       base, base_he, torque, torque_he, speed, speed_he
    #
    # OUTPUTS
    #   edl_system
    #   type : string = {'base', 'base_he', 'torque', 'torque_he', 'speed',
    #   'speed_he'}

    
    motor = edl_system['rover']['wheel_assembly']['motor']
    
    
    if motor_type.lower() == 'base'.lower():
        motor['type'] = 'base'
        motor['torque_stall'] = 165
        motor['speed_noload'] = 3.85
        motor['cost'] = 2.5e5
        
    elif motor_type.lower() == 'base_he'.lower():
        motor['type'] = 'base_he'
        motor['torque_stall'] = 165
        motor['speed_noload'] = 3.85
        motor['cost'] = 2.8e5
        motor['effcy'] = motor['effcy']*1.15
        motor['effcy_tau'] = motor['effcy_tau']*1.15
        
    elif motor_type.lower() == 'torque'.lower():
        motor['type'] = 'torque'
        motor['torque_stall'] = 165*1.25
        motor['speed_noload'] = 3.85
        motor['cost'] = 3e5
        
    elif motor_type.lower() == 'torque_he'.lower():
        motor['type'] = 'torque_he'
        motor['torque_stall'] = 165*1.25
        motor['speed_noload'] = 3.85
        motor['cost'] = 3.36e5
        motor['effcy'] = motor['effcy']*1.15
        motor['effcy_tau'] = motor['effcy_tau']*1.15
        
    elif motor_type.lower() == 'speed'.lower():
        motor['type'] = 'speed'
        motor['torque_stall'] = 165*.75
        motor['speed_noload'] = 3.85*2
        motor['cost'] = 3e5
        motor['effcy_tau'] = motor['effcy_tau']*.75
        
    elif motor_type.lower() == 'speed_he'.lower():
        motor['type'] = 'speed_he'
        motor['torque_stall'] = 165*.75
        motor['speed_noload'] = 3.85*2
        motor['cost'] = 3.36e5
        motor['effcy'] = motor['effcy']*1.15
        motor['effcy_tau'] = motor['effcy_tau']*.75
    else:
       raise Exception('input not recognized')

     
    edl_system['rover']['wheel_assembly']['motor'] = motor
    
    return edl_system

def tau_dcmotor(omega, motor):
    """
    Inputs:  omega:  numpy array      Motor shaft speed [rad/s]
             motor:  dict             Data dictionary specifying motor parameters
    Outputs:   tau:  numpy array      Torque at motor shaft [Nm].  Return argument
                                      is same size as first input argument.
    """
    
    # Check that 2 inputs have been given
    #   IS THIS NECESSARY ANYMORE????
    
    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega],dtype=float) # make the scalar a numpy array
    elif len(np.shape(omega)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')

    # Check that the second input is a dict
    if type(motor) != dict:
        raise Exception('Second input must be a dict')
        
    # Main code
    tau_s    = motor['torque_stall']
    tau_nl   = motor['torque_noload']
    omega_nl = motor['speed_noload']
    
    # initialize
    tau = np.zeros(len(omega),dtype = float)
    for ii in range(len(omega)):
        if omega[ii] >= 0 and omega[ii] <= omega_nl:
            tau[ii] = tau_s - (tau_s-tau_nl)/omega_nl *omega[ii]
        elif omega[ii] < 0:
            tau[ii] = tau_s
        elif omega[ii] > omega_nl:
            tau[ii] = 0
        
    return tau

def F_buoyancy_descent(edl_system,planet,altitude):
    
    # Compute the net buoyancy force. 
    
    density, _, _ = get_local_atm_properties(planet, altitude)
    
    F = np.sign(planet['g'])*planet['g']*density*edl_system['volume']
    
    return F

def F_drag_descent(edl_system,planet,altitude,velocity):
    
    # Compute the net drag force. 
    
    
    # compute the density of planetary atmosphere at current altitude
    density, _, _ = get_local_atm_properties(planet, altitude)
    
    # This is the (1/2)*density*velocity^2 part of the drag model. The missing
    # bit is area*Cd, which we'll figure out below.
    rhov2=0.5*density*velocity**2
    
    
    # *************************************
    # Determine which part(s) of the EDL system are contributing to drag
    
    # If the heat shield has not been ejected, use that as our drag
    # contributor. Otherwise, use the sky crane.
    if not edl_system['heat_shield']['ejected']:
        ACd_body = np.pi*(edl_system['heat_shield']['diameter']/2.0)**2*edl_system['heat_shield']['Cd']
    else:
        ACd_body = edl_system['sky_crane']['area']*edl_system['sky_crane']['Cd']

    
    # if the parachute is in the deployed state, need to account for its area
    # in the drag calculation
    if edl_system['parachute']['deployed'] and not edl_system['parachute']['ejected']:
        ACd_parachute = np.pi*(edl_system['parachute']['diameter']/2.0)**2*edl_system['parachute']['Cd']
    else:
        ACd_parachute = 0.0
    
    
    # This computes the ultimate drag force
    F=rhov2*(ACd_body+ACd_parachute)
    
    return F

def F_gravity_descent(edl_system,planet):
    
    # Compute the gravitational force acting on the EDL system

    F = get_mass_edl(edl_system)*planet['g']

    return F

def F_gravity(terrain_angle, rover, planet):
    """
    Inputs:  terrain_angle:  numpy array   Array of terrain angles [deg]
                     rover:  dict          Data structure specifying rover 
                                            parameters
                    planet:  dict          Data dictionary specifying planetary 
                                            parameters
    
    Outputs:           Fgt:  numpy array   Array of forces [N]
    """
    
    # Check that the first input is a scalar or a vector
    if (type(terrain_angle) != int) and (type(terrain_angle) != float) and (not isinstance(terrain_angle, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(terrain_angle, np.ndarray):
        terrain_angle = np.array([terrain_angle],dtype=float) # make the scalar a numpy array
    elif len(np.shape(terrain_angle)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that values of the first input are within the feasible range  
    if max([abs(x) for x in terrain_angle]) > 75:    
        raise Exception('All elements of the first input must be between -75 degrees and +75 degrees')

    # Check that the second input is a dict
    if type(rover) != dict:
        raise Exception('Second input must be a dict')
    
    # Check that the third input is a dict
    if type(planet) != dict:
        raise Exception('Third input must be a dict')
        
    # Main Code
    m = get_mass_rover(rover)
    g = planet['g']
    
    Fgt = np.array([-m*g*math.sin(math.radians(x)) for x in terrain_angle], dtype = float)
        
    return Fgt

def F_drive(omega, rover):
    """
    Inputs:  omega:  numpy array   Array of motor shaft speeds [rad/s]
             rover:  dict          Data dictionary specifying rover parameters
    
    Outputs:    Fd:  numpy array   Array of drive forces [N]
    """

    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega],dtype=float) # make the scalar a numpy array
    elif len(np.shape(omega)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')

    # Check that the second input is a dict
    if type(rover) != dict:
        raise Exception('Second input must be a dict')
    
    # Main code
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    
    tau = tau_dcmotor(omega, rover['wheel_assembly']['motor'])
    tau_out = tau*Ng
    
    r = rover['wheel_assembly']['wheel']['radius']
    
    # Drive force for one wheel
    Fd_wheel = tau_out/r 
    
    # Drive force for all six wheels
    Fd = 6*Fd_wheel
    
    return Fd

def F_rollingCorr(omega, terrain_angle, rover, planet, Crr):
    """
    Inputs:           omega:  numpy array     Motor shaft speed [rad/s]
              terrain_angle:  numpy array     Array of terrain angles [deg]
                      rover:  dict            Data structure specifying rover 
                                              parameters
                    planet:   dict            Data dictionary specifying planetary 
                                              parameters
                        Crr:  scalar          Value of rolling resistance coefficient
                                              [-]
    
    Outputs:           Frr:  numpy array     Array of forces [N]
    """
    
    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega],dtype=float) # make the scalar a numpy array
    elif len(np.shape(omega)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the second input is a scalar or a vector
    if (type(terrain_angle) != int) and (type(terrain_angle) != float) and (not isinstance(terrain_angle, np.ndarray)):
        raise Exception('Second input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(terrain_angle, np.ndarray):
        terrain_angle = np.array([terrain_angle],dtype=float) # make the scalar a numpy array
    elif len(np.shape(terrain_angle)) != 1:
        raise Exception('Second input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the first two inputs are of the same size
    if len(omega) != len(terrain_angle):
        raise Exception('First two inputs must be the same size')
    
    # Check that values of the second input are within the feasible range  
    if max([abs(x) for x in terrain_angle]) > 75:    
        raise Exception('All elements of the second input must be between -75 degrees and +75 degrees')
        
    # Check that the third input is a dict
    if type(rover) != dict:
        raise Exception('Third input must be a dict')
        
    # Check that the fourth input is a dict
    if type(planet) != dict:
        raise Exception('Fourth input must be a dict')
        
    # Check that the fifth input is a scalar and positive
    if (type(Crr) != int) and (type(Crr) != float):
        raise Exception('Fifth input must be a scalar')
    if Crr <= 0:
        raise Exception('Fifth input must be a positive number')
        
    # Main Code
    m = get_mass_rover(rover)
    g = planet['g']
    r = rover['wheel_assembly']['wheel']['radius']
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    
    v_rover = r*omega/Ng
    
    # compute rolling resistance
    Crr = np.sqrt(0.0005/r) + 0.05
    
    Fn = np.array([m*g*math.cos(math.radians(x)) for x in terrain_angle],dtype=float) # normal force
    
    Frr_simple = -Crr*Fn # simple rolling resistance
    
    Frr = np.array([math.erf(40*v_rover[ii]) * Frr_simple[ii] for ii in range(len(v_rover))], dtype = float)
    
    return Frr

def F_net(omega, terrain_angle, rover, planet, Crr):
    
    """
    Inputs:           omega:  numpy array     Motor shaft speed [rad/s]
              terrain_angle:  numpy array     Array of terrain angles [deg]
                      rover:  dict            Data structure specifying rover 
                                              parameters
                     planet:  dict            Data dictionary specifying planetary 
                                              parameters
                        Crr:  scalar          Value of rolling resistance 
                                              coefficient [-]
    
    Outputs:           Fnet:  numpy array     Array of forces [N]
    """
    
    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega],dtype=float) # make the scalar a numpy array
    elif len(np.shape(omega)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the second input is a scalar or a vector
    if (type(terrain_angle) != int) and (type(terrain_angle) != float) and (not isinstance(terrain_angle, np.ndarray)):
        raise Exception('Second input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(terrain_angle, np.ndarray):
        terrain_angle = np.array([terrain_angle],dtype=float) # make the scalar a numpy array
    elif len(np.shape(terrain_angle)) != 1:
        raise Exception('Second input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the first two inputs are of the same size
    if len(omega) != len(terrain_angle):
        raise Exception('First two inputs must be the same size')
    
    # Check that values of the second input are within the feasible range  
    if max([abs(x) for x in terrain_angle]) > 75:    
        raise Exception('All elements of the second input must be between -75 degrees and +75 degrees')
        
    # Check that the third input is a dict
    if type(rover) != dict:
        raise Exception('Third input must be a dict')
        
    # Check that the fourth input is a dict
    if type(planet) != dict:
        raise Exception('Fourth input must be a dict')
        
    # Check that the fifth input is a scalar and positive
    if (type(Crr) != int) and (type(Crr) != float):
        raise Exception('Fifth input must be a scalar')
    if Crr <= 0:
        raise Exception('Fifth input must be a positive number')
    
    # Main Code
    Fd = F_drive(omega, rover)
    Frr = F_rollingCorr(omega, terrain_angle, rover, planet, Crr)
    Fg = F_gravity(terrain_angle, rover, planet)
    
    Fnet = Fd - Frr - Fg 
    
    return Fnet

def motorW(v, rover):
    """
    Inputs:               v:  numpy array     Array of velocities [m/s]
                      rover:  dict            Data structure specifying rover 
                                              parameters
    
    Outputs:              w:  numpy array     Array of motor speeds [rad/s]
    """
    
    # Check that the v input is a scalar or a vector
    if (type(v) != int) and (type(v) != float) and (not isinstance(v, np.ndarray)):
        raise Exception('v input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(v, np.ndarray):
        v = np.array([v],dtype=float) # make the scalar a numpy array
    elif len(np.shape(v)) != 1:
        raise Exception('v input must be a scalar or a vector. Matrices are not allowed.')
    
    # Check that the rover input is a dict
    if type(rover) != dict:
        raise Exception('rover input must be a dict')
        
        
    # Main Code
    r = rover['wheel_assembly']['wheel']['radius']
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    
    w = v*Ng/r
    
    return w

def mechpower(v, rover):
    """
    Inputs:               v:  numpy array     Array of velocities [m/s]
                      rover:  dict            Data structure specifying rover 
                                              parameters
    
    Outputs:              P:  numpy array     Array of instantaneous power 
                                              output of a single motor 
                                              corresponding to each element in 
                                              array v [W]
    """
    # Check that the v input is a scalar or a vector
    if (type(v) != int) and (type(v) != float) and (not isinstance(v, np.ndarray)):
        raise Exception('v input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(v, np.ndarray):
        v = np.array([v],dtype=float) # make the scalar a numpy array
    elif len(np.shape(v)) != 1:
        raise Exception('v input must be a scalar or a vector. Matrices are not allowed.')
    
    # Check that the rover input is a dict
    if type(rover) != dict:
        raise Exception('rover input must be a dict')
    
    omega = motorW(v, rover)  
    tau = tau_dcmotor(omega, rover['wheel_assembly']['motor']) 
    
    P = tau*omega
    
    return P

def battenergy(t,v,rover):
    """
    Inputs:               t:  numpy array     Array of time samples from a 
                                              rover simulation [s]
                          v:  numpy array     Array of velocities from a rover 
                                              simulation [m/s]
                      rover:  dict            Data structure specifying rover 
                                              parameters
    
    Outputs:              E:  scalar          Total electrical energy consumed 
                                              from the rover battery pack over
                                              the input simulation profile [J]
    """
    # Check that the t input is a scalar or a vector
    if (not isinstance(t, np.ndarray)):
        raise Exception('t input must be a scalar or a vector. If t is a vector, it should be defined as a numpy array.')
    elif len(np.shape(t)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the v input is a scalar or a vector
    if (not isinstance(v, np.ndarray)):
        raise Exception('v input must be a scalar or a vector. If v is a vector, it should be defined as a numpy array.')
    elif len(np.shape(v)) != 1:
        raise Exception('v input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the first two inputs are of the same size
    if len(t) != len(v):
        raise Exception('First two inputs must be the same size')
    
    # Main code
    P = mechpower(v, rover) # calculate power at each time/velocity
    omega = motorW(v, rover) # calculate motor speed (used in next line)
    tau = tau_dcmotor(omega, rover['wheel_assembly']['motor']) # calculate torque (used for efficiency info)
    
    # Determine efficiency for each time/velocity
    effcy_tau = rover['wheel_assembly']['motor']['effcy_tau'].ravel() # change to 1D array
    effcy = rover['wheel_assembly']['motor']['effcy'].ravel()
    effcy_fun = interp1d(effcy_tau, effcy, kind = 'cubic', fill_value = 'extrapolate') # fit the cubic spline
    effcy_dat = effcy_fun(tau)
    
    
    validIndices = np.where(effcy_dat > 0)
    P_batt = np.zeros(P.shape)
    P_batt[validIndices] = P[validIndices] / effcy_dat[validIndices]

    
    # Calculate battery power for each time/velocity
    #print(effcy_dat)
    #P_batt = P/effcy_dat
    
    # integrate to calculate energy    
    #E_motor = simpson(P_batt, t)
    E_motor = np.trapz(P_batt,t)
    E = 6*E_motor # 6 wheels, each with a dedicated motor
    
    return E

def rover_dynamics(t, y, rover, planet, experiment):
    """
    Inputs:         t:  scalar            Time sample [s]
                    y:  numpy array       Two element array of dependent variables 
                                          (i.e., state vector). First element is 
                                          rover velocity [m/s] and second 
                                          element is rover position [m]
                rover:  dict              Data structure specifying rover 
                                          parameters
               planet:  dict              Data dictionary specifying planetary 
                                          parameters
           experiment:  dict              Data dictionary specifying experiment 
                                          definition
    
    Outputs:     dydt:  numpy array       First derivatives of state vector. 
                                          First element is rover acceleration 
                                          [m/s^2] and second element is rover 
                                          velocity [m/s]
    """
    
    # Check that the t input is a scalar
    if (type(t) != int) and (type(t) != float) and (not isinstance(t, np.ndarray)) and (not isinstance(t,np.float64)):
        raise Exception('t input must be a scalar.')
    elif isinstance(t, np.ndarray):
        if len(t) == 1:
            t = float(t) # make a scalar
        else:
            raise Exception('t input must be a scalar.')
    
    # Check that y input is a 2-entry numpy array
    if (not isinstance(y, np.ndarray)) or (len(y) != 2):
        raise Exception('y must be a 2x1 numpy array.')
    elif isinstance(y[0], np.ndarray):
        y = np.array([float(y[0]),float(y[1])]) # this will turn the column vector into a row
    
    # Check that the rover input is a dict
    if type(rover) != dict:
        raise Exception('rover input must be a dict')
    
    # Check that the planet input is a dict
    if type(planet) != dict:
        raise Exception('planet input must be a dict')
    
    # Check that the experiment input is a dict
    if type(experiment) != dict:
        raise Exception('experiment input must be a dict')
    
    # Main code
    v = float(y[0]) # velocity
    pos = float(y[1]) # position
    
    omega = motorW(v, rover)   
    alpha_fun = interp1d(experiment['alpha_dist'].ravel(), experiment['alpha_deg'].ravel(), kind = 'cubic', fill_value="extrapolate")
    terrain_angle = float(alpha_fun(pos))
    F = F_net(omega, terrain_angle, rover, planet, experiment['Crr'])
    
    m = get_mass_rover(rover)
    accel = float(F/m)
    dydt = np.array([accel, v], dtype = float)
    
    return dydt

def end_of_mission_event(end_event):
    """
    Defines an event that terminates the mission simulation. Mission is over
    when rover reaches a certain distance, has moved for a maximum simulation 
    time or has reached a minimum velocity.            
    """
    
    mission_distance = end_event['max_distance']
    mission_max_time = end_event['max_time']
    mission_min_velocity = end_event['min_velocity']
    
    # Assume that y[1] is the distance traveled
    distance_left = lambda t,y: mission_distance - y[1]
    distance_left.terminal = True
    
    time_left = lambda t,y: mission_max_time - t
    time_left.terminal = True
    time_left.direction = -1
    
    velocity_threshold = lambda t,y: y[0] - mission_min_velocity;
    velocity_threshold.terminal = True
    
    # terminal indicates whether any of the conditions can lead to the
    # termination of the ODE solver. In this case all conditions can terminate
    # the simulation independently.
    
    # direction indicates whether the direction along which the different
    # conditions is reached matter or does not matter. In this case, only
    # the direction in which the velocity treshold is arrived at matters
    # (negative)
    
    events = [distance_left, time_left, velocity_threshold]
    
    return events

def simulate_rover(rover,planet,experiment,end_event):
    """
    Inputs:     rover:  dict              Data structure specifying rover 
                                          parameters
               planet:  dict              Data dictionary specifying planetary 
                                          parameters
           experiment:  dict              Data dictionary specifying experiment 
                                          definition
            end_event:  dict              Data dictionary containing the 
                                          conditions necessary and sufficient 
                                          to terminate simulation of rover 
                                          dynamics                 
    
    Outputs:    rover:  dict              Updated rover structure including 
                                          telemetry information
    """
    # Check that the rover input is a dict
    if type(rover) != dict:
        raise Exception('rover input must be a dict')
    
    # Check that the planet input is a dict
    if type(planet) != dict:
        raise Exception('planet input must be a dict')
    
    # Check that the experiment input is a dict
    if type(experiment) != dict:
        raise Exception('experiment input must be a dict')
        
    # Check that the end_event input is a dict
    if type(end_event) != dict:
        raise Exception('end_event input must be a dict')
    
    # Main Code
    fun = lambda t,y: rover_dynamics(t, y, rover, planet, experiment) # differential equation
    t_span = experiment['time_range'] # time span
    y0 = experiment['initial_conditions'].ravel() # initial conditions
    events = end_of_mission_event(end_event) # stopping criteria
    sol = solve_ivp(fun, t_span, y0, method = 'BDF', events=events, max_step=1.0) #t_eval=(np.linspace(0, 3000, 1000)))  # need a stiff solver like BDF
    
    # extract necessary data
    v_max = max(sol.y[0,:])
    v_avg = mean(sol.y[0,:])
    P = mechpower(sol.y[0,:], rover)
    E = battenergy(sol.t,sol.y[0,:],rover)
    
    # Add telemetry info to rover dict
    telemetry = {'Time' : sol.t,
                 'completion_time' : sol.t[-1],
                 'velocity' : sol.y[0,:],
                 'position' : sol.y[1,:],
                 'distance_traveled' : sol.y[1,-1],  # Matlab version had an integration of velocity over time, but if velocity must be positive (defined by min_velocity), then the final position is the distance traveled
                 'max_velocity' : v_max,
                 'average_velocity' : v_avg,
                 'power' : P,
                 'battery_energy' : E,
                 'energy_per_distance' : E/sol.y[1,-1]}
    
    rover['telemetry'] = telemetry
    return rover

def edl_events(edl_system, mission_events):

    # Defines events that occur in EDL System simulation.
    #
    # y = [ velocity, altitude, fuel_mass] and more
    #
    #
    # 0. Reached altitude to eject heat shield 
    # 1. Reached altitude to eject parachute 
    # 2. Reached altitude to turn on rockets 
    # 3. Reached altitude to turn on crane & altitude control
    # 4. Out of fuel --> y(3)<=0. Terminal. Direction: -1.
    # 5. EDL System crashed at zero altitude
    # 6. Reached speed at which speed-controlled descent is required
    # 7. Reached position at which altitude control is required
    # 8. Rover has touched down on surface of Mars

    
    event0 = lambda t, y: y[1] - mission_events['alt_heatshield_eject'] - int(edl_system["heat_shield"]["ejected"])*999999
    event0.terminal = True
    event0.direction = -1
    
    event1 = lambda t, y: y[1] - mission_events['alt_parachute_eject'] - int(edl_system["parachute"]["ejected"])*999999
    event1.terminal = True
    event1.direction = -1
    
    event2 = lambda t, y: y[1] - mission_events['alt_rockets_on'] - int(edl_system["rocket"]["on"])*999999
    event2.terminal = True
    event2.direction = -1

    event3 = lambda t, y: y[1] - mission_events['alt_skycrane_on'] - int(edl_system["sky_crane"]["on"])*999999
    event3.terminal = True
    event3.direction = -1
    
    event4 = lambda t, y: y[2] 
    event4.terminal = True
    event4.direction = -1
    
    event5 = lambda t, y: y[1]
    event5.terminal = True
    event5.direction = -1
    
    event6 = lambda t, y: y[0] - 3*edl_system['speed_control']['target_velocity'] + int(edl_system["speed_control"]["on"])*999999
    event6.terminal = True
    event6.direction = 1
    
    event7 = lambda t, y: y[1] - 1.2*mission_events['alt_skycrane_on'] - int(edl_system["position_control"]["on"])*999999
    event7.terminal = True
    event7.direction = -1
    
    event8 = lambda t, y: y[1] + y[6]
    event8.terminal = True
    event8.direction = -1

    
    events = [event0, event1, event2, event3, event4, event5, event6, event7, event8]
    
    return events

def edl_dynamics(t, y, edl_system, planet):

    # Dynamics of EDL as it descends and lowers the rover to the surface. 
    # State vector: 
    #   y=[vel_edl;pos_edl;fuel_mass;ei_vel;ei_pos;vel_rov;pos_rov]
    #   ydot=[accel_edl;vel_edl;dmdt;e_vel;e_pos;accel_rov;vel_rov]
    # 
    # edl altitude, velocity and acceleration are absolute
    # rov is relative to edl
    # fuel_mass is total over all rockets
    #
    # 
    # Note: this is a VARIABLE MASS SYSTEM, which means Newton's second law
    # expressed as F=ma cannot be used. The more fundamental relationship is 
    # F = dp/dt, where p is the momentum of the system. (For a particle, p=mv
    # and if m is constant you recover the F=ma form easily.) It is important
    # to consider the momentum contributions of the EDL system and propellant 
    # being expelled from the rocket. Ultimately you end up with something that
    # roughly resembles Newton's second law: F_ext + v_rel*(dm/dt) = ma where m
    # is the mass of the EDL, dm/dt is the rate at which mass is changing (due
    # to propellant being expelled), v_rel is the speed at which propellant is
    # being expelled, a is the acceleration of the EDL and F_ext is the sum of
    # other forces acting on the EDL (drag, bouyancy, gravity). Since
    # v_rel*(dm/dt) is a force, we can write this as F_ext + F_thrust = ma,
    # which is very Newton-like.
    #
    #
    #


    # ********************************************
    # unpack the input state vector into variables with more readable names
    # 
    vel_edl = y[0]       # [m/s] velocity of EDL system
    altitude_edl = y[1]  # [m] altitude of EDL system
    fuel_mass = y[2]     # [kg] total mass of fuel in EDL system 
    ei_vel = y[3]        # [m/s] error integral for velocity error 
    ei_pos = y[4]        # [m] error integral for position (altitude) error 
    vel_rov = y[5]       # [m/s] velocity of rover relative to sky crane
    pos_rov = y[6]       # [m] position of rover relative to sky crane
    
    # ***
    # Record the current mass of the system. Since the edl_system being passed
    # is the initial state system, need to update the fuel mass as we go. This
    # is an artefact of how ode45 and its cousins are implemented.
    edl_system['rocket']['fuel_mass'] = fuel_mass/edl_system['num_rockets'] 
    edl_mass = get_mass_edl(edl_system)
    
    
    # Forces EXCEPT THRUST acting on EDL System
    F_ext = F_gravity_descent(edl_system,planet) + F_buoyancy_descent(edl_system,planet,altitude_edl)+ F_drag_descent(edl_system,planet,altitude_edl,vel_edl)

    # Remove comment if you want some extra debugging display
    #print('\n')  
    #print(F_gravity_descent(edl_system,planet))
    #print(F_buoyancy_descent(edl_system,planet,altitude_edl))
    #print(F_drag_descent(edl_system,planet,altitude_edl,vel_edl))



    # ***********************************************************************
    # Two if statements here for the dynamical state. First one determines the
    # state of the EDL system. The second determines the state of the sky crane
    # operation in terms of lowering the rover. Somewhere else, it should be
    # enforced that the sky crane cannot begin lowering the rover until it is
    # at an appropriate and stable altitude. Here we just will do the dynamics
    # of each without regard for whether we should.
    # ****
    
    # ****************
    # EDL System Dynamics
    if edl_system['rocket']['on'] and not(edl_system['speed_control']['on']) and not(edl_system['position_control']['on']):
    
        # ** Uncontrolled (0.95*max) rocket firing
        F_thrust = 0.9*edl_system['rocket']['max_thrust']*edl_system['num_rockets']  # Thrust from rockets

        dy1dt = (F_ext+F_thrust)/edl_mass   # acceleration
        dy2dt = vel_edl                     # velocity

        # Change in total mass of rockets due to propellant being expelled to
        # produce thrust. Calculate this as F_thrust/v_rel, where v_rel is the
        # effective exhaust velocity of the propellant
        dmdt = -(F_thrust/edl_system['rocket']['effective_exhaust_velocity'])
        
        # error signals
        e_vel = 0
        e_pos = 0

    
    elif edl_system['rocket']['on'] and edl_system['speed_control']['on']:
    
        # ** This is the dynamical regime for when the rockets are firing 
        # ** with a speed controller    
        
        # PID gains
        Kp = edl_system['speed_control']['Kp']
        Kd = edl_system['speed_control']['Kd']
        Ki = edl_system['speed_control']['Ki']

    
        # error and error integral -- can't compute error derivative explicitly
        # due to an implicit relationship. Error derivative, dedt, is equal to
        # dy1dt (acceleration). However, we need dedt to compute F_thrust and
        # F_thrust to compute dy1dt. So the solution is to rearrange thing
        # symbolically so that we eliminate the error derivative term.
        e_vel = edl_system['speed_control']['target_velocity']-vel_edl
    
    
        num = (Kp*e_vel + Kd*(F_ext/edl_mass) + Ki*ei_vel) - edl_mass*planet['g']
        den = (1-Kd/edl_mass)
        F_thrust = num/den
    
        # this ensures we never try to reverse thrust (which is impossible in
        # this system)
        F_thrust = max(edl_system['num_rockets']*edl_system['rocket']['min_thrust'], F_thrust)
    
        # this checks for saturation of thrust (more than 100# of what rockets
        # can deliver)
        F_thrust = min(F_thrust, edl_system['num_rockets']*edl_system['rocket']['max_thrust'])
        
      
        # acceleration and velocity, respectively 
        dy1dt = (F_ext + F_thrust)/edl_mass
        dy2dt = vel_edl
    

        # Change in total mass of rockets due to propellant being expelled to
        # produce thrust. Calculate this as F_thrust/v_rel, where v_rel is the
        # effective exhaust velocity of the propellant
        dmdt = -(F_thrust/edl_system['rocket']['effective_exhaust_velocity'])
    
        # position error
        e_pos = 0
    
    elif edl_system['rocket']['on'] and edl_system['position_control']['on']:
    
        # ** This is the dynamical regime for when the rockets are firing 
        # ** with an altitude controller    
        
        Kp = edl_system['position_control']['Kp']
        Kd = edl_system['position_control']['Kd']
        Ki = edl_system['position_control']['Ki']

    
        # position error and change in that error. note sign convention. 
        e_pos = edl_system['position_control']['target_altitude'] - altitude_edl
        dedt_pos = -vel_edl
        
        # note: planet['g'] is <0 due to sign convention, so negating here gives a
        # positive valued thrust. 
        F_thrust = edl_system['num_rockets']*(Kp*e_pos + Kd*dedt_pos + Ki*ei_pos) - planet['g']*edl_mass
        
        # enforces a minimum thrust level since we cannot thrust downward
        F_thrust = max(edl_system['rocket']['min_thrust']*edl_system['num_rockets'], F_thrust)
           
        # enforces a maximum thrust level (saturation condition)
        F_thrust = min(F_thrust, edl_system['num_rockets']*edl_system['rocket']['max_thrust'])
        
        # velocity and acceleration 
        dy2dt = vel_edl     
        dy1dt = (F_ext+F_thrust)/edl_mass
    
        # Change in total mass of rockets due to propellant being expelled to
        # produce thrust. Calculate this as F_thrust/v_rel, where v_rel is the
        # effective exhaust velocity of the propellant
        dmdt = -(F_thrust/edl_system['rocket']['effective_exhaust_velocity'])
    
         # velocity error 
        e_vel = 0
        
    else:
        
        # ** If we get here, we either have not yet fired up the rockets or we
        # ** have run out of fuel (so thrust and change in mass are zero).
        
        # update state of EDL dynamics
        dy1dt = F_ext/edl_mass
        dy2dt = vel_edl
          
        # since rockets are off in this dynamical regime, we simply can set dmdt=0
        dmdt = 0
        
        # error signals
        e_vel = 0
        e_pos = 0
        
    # Sky Crane dynamics (lowering the rover)
    if edl_system['sky_crane']['on']:
        
        # this is a 1st order model. We instantaneously jump to constant
        # velocity and then stay there (the jump is handled by an initial
        # condition).
        
        dy6dt = 0 # model as constant velocity (zero accel) process
        dy7dt = edl_system['sky_crane']['velocity']
        
    #     print('sky crane platform: h = #f\tv = #f\ta = #f\n',altitude_edl,vel_edl,dy1dt)
    #     print('rover status (rel): h = #f\tv = #f\ta = #f\n',pos_rov,vel_rov,dy6dt)
    #     print('rover status (abs): h = #f\tv = #f\ta = #f\n\n',altitude_edl+pos_rov,vel_edl+vel_rov,dy1dt-dy6dt)
        
    else:
        # rover relative to sky crane
        dy6dt = 0 # rover acceleration
        dy7dt = 0 # rover velocity
    
    
    # the return vector (note that e is deidt since ei is integral of e)
    dydt = np.array([dy1dt, dy2dt, dmdt, e_vel, e_pos, dy6dt, dy7dt])
    
    return dydt

def update_edl_state(edl_system, TE, YE, Y, ITER_INFO):
    # update_edl
    #
    # update status of EDL System based on simulation events
    #
    # 0. Reached altitude to eject heat shield
    # 1. Reached altitude to eject parachute
    # 2. Reached altitude to turn on rockets
    # 3. Reached altitude to turn on crane
    # 4. Out of fuel --> y(3)<=0. Terminal. Direction: -1.
    # 5. EDL System crashed at zero altitude
    # 6. Reached speed at which controlled descent is required
    # 7. Reached altitude at which position control is required
    # 8. Rover has touched down on surface of Mars
    #
    # This also updates the rocket mass (due to fuel having been expelled).

    # default initial conditions are final conditions of prior time interval.
    y0 = Y[:, -1]
    # this updates the per rocket fuel mass in the edl_system struct
    edl_system["rocket"]["fuel_mass"] = y0[2] / edl_system["num_rockets"]
    edl_system["altitude"] = y0[1]
    edl_system["velocity"] = y0[0]

    TERMINATE_SIM = False
    # num_events = length(IE);
    # num_events = len(TE)

    for i in range(9):  

        event = i

        if event == 0:  # heat shield eject
            if TE[i].size != 0:
                time = TE[i][0]
                altitude = YE[i][0, 1]
                speed = YE[i][0, 0]
                rover_rel_pos = YE[i][0, 6]
                rover_rel_vel = YE[i][0, 5]
                if not (edl_system["heat_shield"]["ejected"]):
                    edl_system["heat_shield"]["ejected"] = True
                    if ITER_INFO:
                        print("{:<30} {:<3} {:<8.4f} [s], {:<10} {:<9.4f} [m], {:<7} {:<9.4f} [m/s]".format('Ejecting heat shield at', 't =', time, 'altitude =', altitude, 'speed =', speed))
                    y0 = Y[:, -1]
                    #y0[1] = y0[1]  # - 0.001

        if event == 1:  # parachute shield eject
            if TE[i].size != 0:
                time = TE[i][0]
                altitude = YE[i][0, 1]
                speed = YE[i][0, 0]
                rover_rel_pos = YE[i][0, 6]
                rover_rel_vel = YE[i][0, 5]
                if not (edl_system["parachute"]["ejected"]):
                    edl_system["parachute"]["ejected"] = True
                    if ITER_INFO:
                        print("{:<30} {:<3} {:<8.4f} [s], {:<10} {:<9.4f} [m], {:<7} {:<9.4f} [m/s]".format('Ejecting parachute at', 't =', time, 'altitude =', altitude, 'speed =', speed))
                    y0 = Y[:, -1]
                    #y0[1] = y0[1] #- 0.001

        if event == 2:  # turning on rockets, but w/o control
            if TE[i].size != 0:
                time = TE[i][0]
                altitude = YE[i][0, 1]
                speed = YE[i][0, 0]
                rover_rel_pos = YE[i][0, 6]
                rover_rel_vel = YE[i][0, 5]
                if not (edl_system["rocket"]["on"]):
                    edl_system["rocket"]["on"] = True
                    # edl_system['rocket_control']['on'] = False
                    if ITER_INFO:
                        print("{:<30} {:<3} {:<8.4f} [s], {:<10} {:<9.4f} [m], {:<7} {:<9.4f} [m/s]".format('Turning on rockets at', 't =', time, 'altitude =', altitude, 'speed =', speed))
                    y0 = Y[:, -1]
                    #y0[1] = y0[1] # - 0.001

        if event == 3:  # turn on sky crane if we're low enough (triggers event 4) and we're under a position-controlled regime
            if TE[i].size != 0:
                time = TE[i][0]
                altitude = YE[i][0, 1]
                speed = YE[i][0, 0]
                rover_rel_pos = YE[i][0, 6]
                rover_rel_vel = YE[i][0, 5]
                if not (edl_system["sky_crane"]["on"]) and edl_system["position_control"]["on"]:
                    edl_system["sky_crane"]["on"] = True
                    if ITER_INFO:
                        print("{:<30} {:<3} {:<8.4f} [s], {:<10} {:<9.4f} [m], {:<7} {:<9.4f} [m/s]".format('Turning on sky crane at', 't =', time, 'altitude =', altitude, 'speed =', speed))
                y0 = Y[:, -1]
                y0[5] = edl_system["sky_crane"]["velocity"]
                #y0[1] = y0[1]  - 0.0001



        if event == 4:  # we are out of rocket fuel!
            if TE[i].size != 0:
                time = TE[i][0]
                altitude = YE[i][0, 1]
                speed = YE[i][0, 0]
                rover_rel_pos = YE[i][0, 6]
                rover_rel_vel = YE[i][0, 5]
                if edl_system["rocket"]["on"]:
                    edl_system["rocket"]["on"] = False
                    if ITER_INFO:
                        print("{:<30} {:<3} {:<8.4f} [s], {:<10} {:<9.4f} [m], {:<7} {:<9.4f} [m/s]".format('Ran out of rocket fuel at', 't =', time, 'altitude =', altitude, 'speed =', speed))
                    #y0 = Y[:, -1]
                    #y0[2] = 0 - .01  # no more fuel
                    y0 = Y[:, -1]
                    y0[2] = 0.0 
                    TERMINATE_SIM = True
                    

        if event == 5:  # edl crashed before sky crane is activated
            if TE[i].size != 0:
                time = TE[i][0]
                altitude = YE[i][0, 1]
                speed = YE[i][0, 0]
                rover_rel_pos = YE[i][0, 6]
                rover_rel_vel = YE[i][0, 5]
                if ITER_INFO:
                    print("{:<30} {:<3} {:<8.4f} [s], {:<10} {:<9.4f} [m], {:<7} {:<9.4f} [m/s]".format('EDL SYSTEM CRASHED INTO MARS AT', 't =', time, 'altitude =', altitude, 'speed =', speed))
                y0 = []
                TERMINATE_SIM = True



        if event == 6:  # still descending, but slow enough to turn on speed controller
            if TE[i].size != 0:
                time = TE[i][0]
                altitude = YE[i][0, 1]
                speed = YE[i][0, 0]
                rover_rel_pos = YE[i][0, 6]
                rover_rel_vel = YE[i][0, 5]
                if not (edl_system["speed_control"]["on"]) and not (edl_system["position_control"]["on"]):
                    edl_system["speed_control"]["on"] = True
                    if ITER_INFO:
                        print("{:<30} {:<3} {:<8.4f} [s], {:<10} {:<9.4f} [m], {:<7} {:<9.4f} [m/s]".format('Turning on speed control at', 't =', time, 'altitude =', altitude, 'speed =', speed))
                else:
                    edl_system['speed_control']['on'] = True
                    if ITER_INFO:
                        print("{:<30} {:<3} {:<8.4f} [s], {:<10} {:<9.4f} [m], {:<7} {:<9.4f} [m/s]".format('Trouble at', 't =', time, 'altitude =', altitude, 'speed =', speed))
                y0 = Y[:, -1]
                y0[3] = 0
                y0[4] = 0
                #y0[0] = y0[0] #+ 0.01


        if event == 7:  # now we're low enough to activate the altitude control (and turn off speed control)
            if TE[i].size != 0:
                time = TE[i][0]
                altitude = YE[i][0, 1]
                speed = YE[i][0, 0]
                rover_rel_pos = YE[i][0, 6]
                rover_rel_vel = YE[i][0, 5]
                if not (edl_system["position_control"]["on"]) and edl_system['speed_control']['on']:
                    edl_system["speed_control"]["on"] = False
                    edl_system["position_control"]["on"] = True
                    if ITER_INFO:
                        print("{:<30} {:<3} {:<8.4f} [s], {:<10} {:<9.4f} [m], {:<7} {:<9.4f} [m/s]".format('Turning on altitude control at', 't =', time, 'altitude =', altitude, 'speed =', speed))
                        y0 = Y[:, -1]
                        y0[3] = 0
                        y0[4] = 0
                        #y0[1] = y0[1] # - 0.00001
                elif not (edl_system['position_control']['on']):
                    if ITER_INFO:
                        print("SYSTEM FAIL: SPEED CONTROL DID NOT ACTIVATE PRIOR TO ALTITUDE CONTROL")
                    TERMINATE_SIM = True
                    y0 = []
                    #edl_system = redefine_edl_system(edl_system)


        if event == 8:  # we've determined the rover position is at 0 altitude (on the ground)
            if TE[i].size != 0:
                time = TE[i][0]
                altitude = YE[i][0, 1]
                speed = YE[i][0, 0]
                rover_rel_pos = YE[i][0, 6]
                rover_rel_vel = YE[i][0, 5]
                rover_touchdown_speed = speed + rover_rel_vel
                edl_system['rover_touchdown_speed'] = rover_touchdown_speed

                if altitude >= edl_system["sky_crane"]["danger_altitude"] and abs(
                    rover_touchdown_speed
                ) <= abs(edl_system["sky_crane"]["danger_speed"]):
                    if ITER_INFO:
                        print(
                            "The rover has landed!\n   t={:.4f} [s], rover pos = {:.4f} [m], rover speed = {:.4f} [m/s] (sky crane at h={:.4f}, v={:.6f})\n".format(
                                time,
                                altitude + rover_rel_pos,
                                speed + rover_rel_vel,
                                altitude,
                                speed,
                            )
                        )
                    y0 = []
                    TERMINATE_SIM = True
                    edl_system["sky_crane"]["on"] = False
                    edl_system["rover"]["on_ground"] = True

                elif abs(rover_touchdown_speed) > abs(
                    edl_system["sky_crane"]["danger_speed"]
                ):
                    if ITER_INFO:
                        print(
                            "EDL SYSTEM FAIL. Rover has landed, but possible damage due to touch down speed.\n >>> t={:.4f} [s], rover pos = {:10.4f} [m], rover speed = {:10.4f} [m/s] (sky crane at h={:10.4f}, v={:10.4f}\n".format(
                                time,
                                altitude + rover_rel_pos,
                                speed + rover_rel_vel,
                                altitude,
                                speed,
                            )
                        )
                    y0 = []
                    TERMINATE_SIM = True
                    edl_system['sky_crane']['on'] = False
                    edl_system["rover"]["on_ground"] = True
                   

                elif altitude < edl_system["sky_crane"]["danger_altitude"]:
                    if ITER_INFO:
                        print(
                            "EDL SYSTEM FAIL. Rover has landed, but possible damage due to sky crane low altitude.\n >>> t={:.4f} [s], rover pos = {:10.4f} [m], rover speed = {:10.4f} [m/s] (sky crane at h={:10.4f}, v={:10.4f}\n".format(
                                time,
                                altitude + rover_rel_pos,
                                speed + rover_rel_vel,
                                altitude,
                                speed,
                            )
                        )
                    y0 = []
                    TERMINATE_SIM = True
                    edl_system["sky_crane"]["on"] = False
                    edl_system["rover"]["on_ground"] = True


    return edl_system, y0, TERMINATE_SIM

def simulate_edl(edl_system, planet, mission_events, tmax, ITER_INFO):
    # simulate_edl
    #
    # This simulates the EDL system. It requires a definition of the
    # edl_system, the planet, the mission events, a maximum simulation time and
    # has an optional flag to display detailed iteration information.
    
    # handle to events function for edl simulation
    #h_edl_events = lambda t, y: edl_events(t, y, edl_system, mission_events)
    events = edl_events(edl_system, mission_events)
        
    # simulation time span
    tspan = (0, tmax)
    
    # initial state of system
    y0 = np.array([edl_system['velocity'], 
                   edl_system['altitude'], 
                   edl_system['rocket']['initial_fuel_mass'] * edl_system['num_rockets'],
                   0,
                   0,
                   0,
                   0])
    
    
    # *** NOTE: This does not yet check for whether the fuel runs out...
    if ITER_INFO:
        print('Commencing simulation run...\n')
    
    
    # declare our variables (just empty arrays)
    T = np.array([])
    Y = np.array([[], [], [], [], [], [], []])
    TERMINATE_SIM = False
    while not(TERMINATE_SIM):
        
        # run simulation until an event occurs 
        fun = lambda t, y: edl_dynamics(t, y, edl_system, planet)
        sol = solve_ivp(fun, tspan, y0, method='DOP853', events=events, max_step=0.1)
        t_part = sol.t
        Y_part = sol.y
        TE = sol.t_events
        YE = sol.y_events

        # process the event and update the edl_system accordingly. Also sets
        # the initial conditions for the next stage (in y0) and the
        # TERMINATE_SIM flag.
        #print(YE)
        [edl_system, y0, TERMINATE_SIM] = update_edl_state(edl_system, TE, YE, Y_part, ITER_INFO)
        
        # update the simulation time span for the next stage
        tspan = (t_part[-1], tmax)
        
        # appending to grow a list is inefficient, but there is no way to
        # know in advance how many elements we'll need due to the adaptive step
        # size in ode45
        
        T = np.append(T, t_part)
        Y = np.hstack((Y, Y_part))
        #T.append(t_part)
        #Y.append(Y_part)

        
        # This looks for whether we're out of time. other termination
        # conditions checked in update_edl_state
        if tspan[0] >= tspan[1]:
            TERMINATE_SIM = True
    
    return T, Y, edl_system
    
def obj_fun_time(x,edl_system,planet,mission_events,tmax,experiment,end_event):
    # OBJ_FUN_TIME
    # 
    # This function runs both simulations -- edl and rover -- to get a total
    # time to land and travel the specified terrain. 
    #
    #
    
    
    # Note: Although edl_system is modified in this function, the modifications
    # are lost after the function terminates because the struct is not a
    # returned argument and is not a global variable. Thus, we only ever are
    # modifying a local copy.
    #
    
    edl_system = redefine_edl_system(edl_system)
    
    # edl_system=define_chassis(edl_system,'steel');
    # edl_system=define_motor(edl_system,'speed');
    # edl_system=define_batt_pack(edl_system,'LiFePO4',10);
    
    # ****************
    # RUNNING THE EDL SIMULATION
    # **
    #
    # Unpack the edl-related design variables and update the struct
    edl_system['parachute']['diameter'] = x[0]
    edl_system['rocket']['fuel_mass'] = x[4]
    edl_system['rocket']['initial_fuel_mass'] = x[4]
    edl_system['rover']['wheel_assembly']['wheel']['radius'] = x[1]
    edl_system['rover']['chassis']['mass'] = x[2]
    edl_system['rover']['wheel_assembly']['speed_reducer']['diam_gear'] = x[3]
    #
    [time_edl_run,_,edl_system] = simulate_edl(edl_system,planet,mission_events,tmax,False)
    time_edl = time_edl_run[-1]
    #
    # *****************

    
    # *****************
    # RUNNING THE ROVER SIMULATION
    #
    edl_system['rover'] = simulate_rover(edl_system['rover'],planet,experiment,end_event)
    time_rover = edl_system['rover']['telemetry']['completion_time']
    #
    # ****************
    
    
    # ******************
    # CALCULATE TOTAL TIME
    # **
    total_time = time_edl + time_rover
    
    return total_time  
        
def constraints_edl_system(x,edl_system,planet,mission_events,tmax,experiment,end_event,min_strength,max_rover_velocity,max_cost,max_batt_energy_per_meter):
    # constraints_edl_system
    #
    # This function evaluates the nonlinear constraints for the optimization
    # problem to maximize speed (minimize time)
    #
    # To evaluate the constraints entails simulating both the edl system and the
    # rover. Thus, this function calls simulate_edl and simulate_rover.
    #

    edl_system = redefine_edl_system(edl_system)
    
    # CONSTRAINTS_EDL_SYSTEM
    # 
    
    # edl_system=define_chassis(edl_system,'steel');
    # edl_system=define_motor(edl_system,'speed');
    # edl_system=define_batt_pack(edl_system,'LiFePO4',10);
    
    
    # ****************
    # RUNNING THE EDL SIMULATION
    # **
    #
    # Unpack the edl-related design variables and update the struct
    edl_system['parachute']['diameter'] = x[0]
    edl_system['rocket']['fuel_mass'] = x[4]
    edl_system['rocket']['initial_fuel_mass'] = x[4]
    edl_system['rover']['wheel_assembly']['wheel']['radius'] = x[1]
    edl_system['rover']['chassis']['mass'] = x[2]
    edl_system['rover']['wheel_assembly']['speed_reducer']['diam_gear'] = x[3]
    

    #
    # run the edl simulation
    _, _, edl_system = simulate_edl(edl_system,planet,mission_events,tmax,False)
    # time_edl = time_edl_run(end);
    #
    # *****************
    
    
    # *****************
    # RUNNING THE ROVER SIMULATION
    # **
    #
    # run the rover simulation
    edl_system['rover'] = simulate_rover(edl_system['rover'],planet,experiment,end_event)
    # time_rover = edl_system.rover.telemetry.completion_time;
    #
    
    # *****************
    # EVALUATE THE CONSTRAINT FUNCTIONS
    # **
    # Note: some of the following simply normalizes the constraints to be on
    # similar orders of magnitude.
    #
    #
    # The rover must travel the complete distance
    constraint_distance = (end_event['max_distance']-edl_system['rover']['telemetry']['distance_traveled'])/end_event['max_distance']
    #
    # The chassis must be strong enough to survive the landing
    chassis_strength = edl_system['rover']['chassis']['mass']*edl_system['rover']['chassis']['specific_strength']
    constraint_strength = -(chassis_strength-min_strength)/min_strength
    #
    # The battery must not run out of charge
    constraint_battery  = (edl_system['rover']['telemetry']['energy_per_distance']- max_batt_energy_per_meter)/max_batt_energy_per_meter
    #
    # The touchdown speed of the rover must not be too much (or else damage may occur) 
    constraint_velocity = (abs(edl_system['velocity'])-abs(max_rover_velocity))/abs(max_rover_velocity)
    #
    # The total cost cannot exceed our budget
    constraint_cost = (get_cost_edl(edl_system)-max_cost)/max_cost
    
    
    # *****************
    # CONSTRUCT THE CONSTRAINT LIST
    # **
    c=[constraint_distance, constraint_strength, constraint_velocity, constraint_cost, constraint_battery]
    
    return np.array(c)

def redefine_edl_system(edl_system):
    
    edl_system['altitude'] = 11000
    edl_system['velocity'] = -587

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


def obj_fun_plot(x,edl_system,planet,mission_events,tmax,experiment,end_event):
    # OBJ_FUN_TIME
    # 
    # This function runs both simulations -- edl and rover -- to get a total
    # time to land and travel the specified terrain. 
    #
    #
    
    
    # Note: Although edl_system is modified in this function, the modifications
    # are lost after the function terminates because the struct is not a
    # returned argument and is not a global variable. Thus, we only ever are
    # modifying a local copy.
    #
    
    edl_system = redefine_edl_system(edl_system)
    
    # edl_system=define_chassis(edl_system,'steel');
    # edl_system=define_motor(edl_system,'speed');
    # edl_system=define_batt_pack(edl_system,'LiFePO4',10);
    
    # ****************
    # RUNNING THE EDL SIMULATION
    # **
    #
    # Unpack the edl-related design variables and update the struct
    edl_system['parachute']['diameter'] = x[0]
    edl_system['rocket']['fuel_mass'] = x[4]
    edl_system['rocket']['initial_fuel_mass'] = x[4]
    edl_system['rover']['wheel_assembly']['wheel']['radius'] = x[1]
    edl_system['rover']['chassis']['mass'] = x[2]
    edl_system['rover']['wheel_assembly']['speed_reducer']['diam_gear'] = x[3]
    #
    [time_edl_run, Y_edl_run, edl_system] = simulate_edl(edl_system,planet,mission_events,tmax,False)
    time_edl = time_edl_run[-1]
    #
    # *****************

    
    # *****************
    # RUNNING THE ROVER SIMULATION
    #
    edl_system['rover'] = simulate_rover(edl_system['rover'],planet,experiment,end_event)
    rover_position = edl_system['rover']['telemetry']['position']
    rover_time = edl_system['rover']['telemetry']['Time']
    #
    # ****************
    
    
    # ******************
    # CALCULATE TOTAL TIME
    # **

    
    return time_edl, rover_position, rover_time

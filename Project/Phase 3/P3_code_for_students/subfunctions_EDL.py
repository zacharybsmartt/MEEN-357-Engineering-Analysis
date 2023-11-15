"""###########################################################################
#   This file contains subfunctions for Phase 3 of the TAMU MEEN 357 project
#
#   Created by: Former Marvin Numerical Methods Engineering Team
#   Last Modified: 28 October 2023
###########################################################################"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator as pchip
from scipy.integrate import solve_ivp


def get_mass_rover(edl_system):

    # Computes the mass of the rover defined in rover field of the edl system 
    # struct. Assumes that the rover is defined as a dict corresponding to 
    # the specification of project Phase 1.
    
    m = 6*(edl_system['rover']['wheel_assembly']['motor']['mass'] + 
           edl_system['rover']['wheel_assembly']['speed_reducer']['mass'] + 
           edl_system['rover']['wheel_assembly']['wheel']['mass']) + \
        edl_system['rover']['chassis']['mass'] + \
        edl_system['rover']['science_payload']['mass'] + \
        edl_system['rover']['power_subsys']['mass']
    
    return m

def get_mass_rockets(edl_system):

    # Returns the curret total mass of all rockets on the edl system. 

    m = edl_system['num_rockets']*(edl_system['rocket']['structure_mass'] + edl_system['rocket']['fuel_mass'])

    return m

def get_mass_edl(edl_system):

    # Returns the total current mass of the edl system 
    
    m = int(not(edl_system['parachute']['ejected']))*edl_system['parachute']['mass'] + \
        int(not(edl_system['heat_shield']['ejected']))*edl_system['heat_shield']['mass'] + \
        get_mass_rockets(edl_system) + edl_system['sky_crane']['mass'] + get_mass_rover(edl_system)
        
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

def v2M_Mars(v, a):
    # Converts descent speed, v [m/s], to Mach number on Mars as a function of 
    # altitude, a [m].
    
    # Returns only the absolute value Mach number (i.e., uses model
    # M = abs(v)/v_sound))).
    
    # This table defines the speed of sound vs. altitude on Mars
    SPD_data = np.array([[0, 244.4], 
                         [1000, 243.7], 
                         [2000, 243.2],
                         [3000, 242.7], 
                         [4000, 242.2], 
                         [5000, 241.7],
                         [6000, 241.2], 
                         [7000, 240.7], 
                         [8000, 239.6],
                         [9000, 238.4], 
                         [10000, 237.3], 
                         [11000, 236.1],
                         [12000, 235.0], 
                         [13000, 233.8], 
                         [14000, 232.6]])
    
    v_sound_fit = pchip(SPD_data[:, 0], SPD_data[:, 1])
    v_sound = v_sound_fit(a)
    
    M = abs(v) / v_sound
    
    return M


def thrust_controller(edl_system, planet):
    # thrust_controller
    #
    # This function implements a PID Controller for the EDL system. Uses
    # edl_system and planet structs to create a modified edl_system struct.
    # Modifies fields in rocket and telemetry substructs.
    #
    # Calling sequence: edl_system = thrust_controller(edl_system,planet)
    # Inputs:  edl_system - struct
    #          planet     - struct
    # Outputs: edl_system

    
    if type(edl_system) != dict:
        raise Exception('THRUST CONTROLLER: first input must be a dict')

    if type(planet) != dict:
        raise Exception('THRUST CONTROLLER: second input must be a dict')
    
    
    # First, check if the rocket is on and that the control is activated:
    if (edl_system['rocket']['control']['on']) and (edl_system['rocket']['on']):
    
        # If edl_system is within altitude, set the target velocity to zero  
        if edl_system['altitude'] < edl_system['sky_crane']['max_rope'] + 10:
             edl_system['rocket']['control']['target_velocity'] = 0.0
      
      
        # Calculate the error (difference between actual and target velocity)
        e = edl_system['control']['target_velocity'] - edl_system['velocity']
      
        # Update the error in telemetry array:
        edl_system['telemetry']['error'].append(e)  # = [edl_system['telemetry']['error'], e]  # check this
           
        # Set the parameters of the PID controller (Kp: proportional, 
        # Kd: derivative, Ki: integral terms)
        Kp = edl_system['control']['Kp']
        Kd = edl_system['control']['Kd']
        Ki = edl_system['control']['Ki']    
       
        #e     = edl_system.telemetry.error(end);
        dt    = edl_system['telemetry']['time'][-1] - edl_system['telemetry']['time'][-2]
        ei    = edl_system['telemetry']['error'][-1]
        eim1  = edl_system['telemetry']['error'][-2]
        eim2  = edl_system['telemetry']['error'][-3]
        # Calculate the derivative of the error wrt time
        dedt  = (3*ei-4*eim1+eim2)/2/dt
      
        # Integrate the error:
        ie    = np.trapz(edl_system['telemetry']['time'], edl_system['telemetry']['error'])
      
        # *****************
        # Compute Needed Thrust
        # Two-step process: (1) compute thrust using model, (2) apply corrections
        # as needed due to saturating rocket capabilities.
      
        # Calculate the required thrust from model
        edl_system['rocket']['thrust'] = Kp*e + Kd*dedt + Ki*ie + \
            edl_system['total_mass']*np.sign(planet['gravity'])*planet['gravity']/edl_system['rocket']['number_of_rockets']
       
        # check to see if we're over or under the limits of the rocket motors
        if abs(edl_system['rocket']['thrust']) > abs(edl_system['rocket']['max_thrust']):
            # got to here? Asking for more thrust than rocket can give, so model
            # it as though we've hit the motor max thrust.
            edl_system['rocket']['thrust'] = \
               np.sign(edl_system['rocket']['thrust'])*edl_system['rocket']['max_thrust']
      
        elif abs(edl_system['rocket']['thrust']) < abs(edl_system['rocket']['min_thrust']):
            # got to here? Asking for less thrust than we can deliver (e.g.,
            # cannot thrust in both directions; can't turn rockets on then off
            # then back on. So knock it down to min thrust.
            edl_system.rocket.thrust = \
                np.sign(edl_system['rocket']['thrust'])*edl_system['rocket']['min_thrust']

        
    elif edl_system['rocket']['on'] and (edl_system['rocket']['control']['on'] == False):
        # if we get to this clause, it means the rockets are on but the
        # controller is not. In this case, we go with a     
        edl_system['rocket']['thrust'] = edl_system['rocket']['fixed_thrust']
        edl_system['telemetry']['error'].append(0) # = [edl_system.telemetry.error 0];
        #edl_system.telemetry.thrust =[edl_system.telemetry.thrust edl_system.rocket.thrust];
    else:
        edl_system['telemetry']['error'].append(0) # = [edl_system.telemetry.error 0];
        #edl_system.rocket.thrust = 0.0;
        #edl_system.telemetry.thrust =[edl_system.telemetry.thrust edl_system.rocket.thrust];  
    
    if edl_system['rocket']['on']:
        edl_system['telemetry']['thrust'].append(edl_system['rocket']['thrust']) # =[edl_system.telemetry.thrust edl_system.rocket.thrust];
    
    else:
        edl_system['telemetry']['thrust'].append(0) # =[edl_system.telemetry.thrust 0];
       
    return edl_system 

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
    F_ext = F_gravity_descent(edl_system,planet) + \
            F_buoyancy_descent(edl_system,planet,altitude_edl) + \
            F_drag_descent(edl_system,planet,altitude_edl,vel_edl)

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
    
        # this checks for saturation of thrust (more than 100% of what rockets
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
                    y0[1] = y0[1]  # - 0.001

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
                    y0[1] = y0[1] #- 0.001

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
                    y0[1] = y0[1] # - 0.001

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
                y0[1] = y0[1] # - 0.00001

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
                    y0 = []
                    TERMINATE_SIM = True

        if event == 5:  # edl crashed before sky crane is activated
            if TE[i].size != 0:
                time = TE[i][0]
                altitude = YE[i][0, 1]
                speed = YE[i][0, 0]
                rover_rel_pos = YE[i][0, 6]
                rover_rel_vel = YE[i][0, 5]
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
                y0 = Y[:, -1]
                y0[3] = 0
                y0[4] = 0
                y0[0] = y0[0] #+ 0.01

        if event == 7:  # now we're low enough to activate the altitude control (and turn off speed control)
            if TE[i].size != 0:
                time = TE[i][0]
                altitude = YE[i][0, 1]
                speed = YE[i][0, 0]
                rover_rel_pos = YE[i][0, 6]
                rover_rel_vel = YE[i][0, 5]
                if not (edl_system["position_control"]["on"]):
                    edl_system["speed_control"]["on"] = False
                    edl_system["position_control"]["on"] = True
                    if ITER_INFO:
                        print("{:<30} {:<3} {:<8.4f} [s], {:<10} {:<9.4f} [m], {:<7} {:<9.4f} [m/s]".format('Turning on altitude control at', 't =', time, 'altitude =', altitude, 'speed =', speed))
                        y0 = Y[:, -1]
                        y0[3] = 0
                        y0[4] = 0
                        y0[1] = y0[1] # - 0.00001

        if event == 8:  # we've determined the rover position is at 0 altitude (on the ground)
            if TE[i].size != 0:
                time = TE[i][0]
                altitude = YE[i][0, 1]
                speed = YE[i][0, 0]
                rover_rel_pos = YE[i][0, 6]
                rover_rel_vel = YE[i][0, 5]
                rover_touchdown_speed = speed + rover_rel_vel

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
                    edl_system["sky_crane"]["on"] = False
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
    
    
        
    
    
    
    



from math import *
from define_rover import *
import numpy as np
from define_rover import rover
from scipy.interpolate import interp1d
from define_experiment import *

# Define the rover and planet based on predefined parameters
rover, planet = rover()
experiment, end_event = experiment1()

# Lambda function to convert degrees to radians
degToRad = lambda deg: deg * np.pi / 180

# Function to calculate the total mass of the rover
def get_mass(rover):
    """
    Inputs:  rover:  dict      Data structure containing rover parameters
    
    Outputs:     m:  scalar    Rover mass [kg].
    """
    
    # Check that the input is a dict
    if type(rover) != dict:
        raise Exception('Input must be a dict')
    
    # add up mass of chassis, power subsystem, science payload, 
    # and components from all six wheel assemblies
    m = rover['chassis']['mass'] \
        + rover['power_subsys']['mass'] \
        + rover['science_payload']['mass'] \
        + 6*rover['wheel_assembly']['motor']['mass'] \
        + 6*rover['wheel_assembly']['speed_reducer']['mass'] \
        + 6*rover['wheel_assembly']['wheel']['mass'] \
    
    return m


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


def tau_dcmotor(omega, motor):
    """
    Inputs:  omega:  numpy array      Motor shaft speed [rad/s]
             motor:  dict             Data dictionary specifying motor parameters
    Outputs:   tau:  numpy array      Torque at motor shaft [Nm].  Return argument
                                      is same size as first input argument.
    """
    
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
    
    


def F_rolling(omega, terrain_angle, rover, planet, Crr):
    """
    Inputs:           omega:  numpy array     Motor shaft speed [rad/s]
              terrain_angle:  numpy array     Array of terrain angles [deg]
                      rover:  dict            Data structure specifying rover 
                                              parameters
                    planet:  dict            Data dictionary specifying planetary 
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
    m = get_mass(rover)
    g = planet['g']
    r = rover['wheel_assembly']['wheel']['radius']
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    
    v_rover = r*omega/Ng
    
    Fn = np.array([m*g*math.cos(math.radians(x)) for x in terrain_angle],dtype=float) # normal force
    Frr_simple = -Crr*Fn # simple rolling resistance
    
    Frr = np.array([math.erf(40*v_rover[ii]) * Frr_simple[ii] for ii in range(len(v_rover))], dtype = float)
    
    return Frr


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
    m = get_mass(rover)
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


def F_net(omega, terrain_angle, rover, planet, Crr):
    """
    Inputs:           omega:  list     Motor shaft speed [rad/s]
              terrain_angle:  list     Array of terrain angles [deg]
                      rover:  dict     Data structure specifying rover 
                                      parameters
                     planet:  dict     Data dictionary specifying planetary 
                                      parameters
                        Crr:  scalar   Value of rolling resistance coefficient
                                      [-]
    
    Outputs:           Fnet:  list     Array of forces [N]
    """
    
    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
    # if (not isinstance(omega, np.ndarray)):
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
    Frr = F_rolling(omega, terrain_angle, rover, planet, Crr)
    Fg = F_gravity(terrain_angle, rover, planet)
    
    Fnet = Fd + Frr + Fg # signs are handled in individual functions
    
    return Fnet


def motorW(v, rover):
    """
    Calculate the motor shaft's rotational speed [rad/s] based on the
    rover's translational velocity and the rover's parameters.
    This function is designed to be vectorized to handle velocity vectors.

    :param v: Scalar or vector numpy array of rover velocities.
    :param rover: Dictionary containing rover parameters.
    :return: Vector of motor speeds with the same size as the input velocity.
    """

    # Check the type of input velocity
    if (type(v) != int) and (type(v) != float) and (not isinstance(v, (np.ndarray, np.floating, np.integer))):
        raise Exception('Input velocity must be a scalar or a vector numpy array.')

    # Convert a scalar input to a numpy array
    elif not isinstance(v, np.ndarray):
        v = np.array([v], dtype=float)

    # Check if the input is a vector
    elif len(np.shape(v)) != 1:
        raise Exception('Input velocity must be a scalar or a vector. Matrices are not allowed.')

    # Get gear ratio and wheel radius from rover data
    gearRatio = rover['wheel_assembly']['speed_reducer']
    wheelRadius = rover['wheel_assembly']['wheel']['radius']

    # Initialize a zero omega array with the same length as the velocity vector
    w = np.zeros(len(v))

    # Calculate the motor speed for each element in the velocity vector
    for i in range(len(v)):
        w[i] = ((v[i] * get_gear_ratio(gearRatio)) / wheelRadius)

    return w


def rover_dynamics(t, y, rover, planet, experiment):
    """
    This function computes the derivative of the state vector for the rover given its
    current state. The state vector is [velocity, position]. It requires rover and experiment dictionary inputs
    and is intended to be used with an ODE solver.

    Parameters:
    t (float or int): The current time.
    y (np.ndarray): A 1D numpy array representing the current state vector.
    rover (dict): A dictionary containing rover parameters.
    planet (dict): A dictionary containing planet parameters.
    experiment (dict): A dictionary containing experiment parameters.

    Returns:
    np.ndarray: A 1D numpy array representing the derivative of the state vector.
    """

    # Check input parameter types
    if not isinstance(rover, dict):
        raise Exception("'rover' must be a dictionary")
    if not (isinstance(t, (int, float)) and isinstance(y, np.ndarray)):
        raise Exception("'t' must be a scalar value, and 'y' must be a 1D numpy array")
    if not isinstance(planet, dict):
        raise Exception("'planet' must be a dictionary")
    if not isinstance(experiment, dict):
        raise Exception("'experiment' must be a dictionary")

    # Get the terrain profile from the experiment
    alpha_dist = experiment['alpha_dist']

    # Extract current velocity and position from the state vector
    velocity = float(y[0])

    # Interpolate terrain angle based on the current position
    alpha_fun = interp1d(alpha_dist, experiment['alpha_deg'], kind='cubic', fill_value='extrapolate')
    terrain_angle = float(alpha_fun(y[1]))

    # Calculate the motor speed
    o = motorW(velocity, rover)

    # Calculate acceleration using F_net function
    accel = F_net(o, terrain_angle, rover, planet, 0.1) / get_mass(rover)

    # Create a new state vector with velocity derivative and position
    dydt = np.array([float(np.round(accel, 2)), y[0]])

    return dydt


def mechpower(v, rover):
    """
    This function computes the instantaneous mechanical power output by a single DC motor at each point in a given velocity profile.

    Parameters:
    v (float, int, or np.ndarray): The velocity profile for which mechanical power is calculated.
    rover (dict): A dictionary containing rover parameters.

    Returns:
    float or np.ndarray: The instantaneous mechanical power output corresponding to the input velocity profile.
    """

    # Check the type of input velocity
    if not np.isscalar(v) and not (isinstance(v, np.ndarray) and v.ndim == 1):
        raise Exception('Velocity parameter `v` must be a scalar or 1d array.')
    if isinstance(v, np.ndarray) and not all([np.isscalar(i) for i in v]):
        raise Exception('Velocity parameter `v` must contain scalars only.')
    if not isinstance(rover, dict):
        raise Exception('The parameter `rover` is not a dictionary type.')

    # Calculate the motor and wheel speeds
    omegaWheel = v / rover['wheel_assembly']['wheel']['radius']
    omegaMotor = get_gear_ratio(rover['wheel_assembly']['speed_reducer']) * omegaWheel

    # Calculate the motor torque and mechanical power
    torqueMotor = tau_dcmotor(omegaMotor, rover['wheel_assembly']['motor'])
    P = torqueMotor * omegaMotor

    return P


def battenergy(t, v, rover):
    """
    This function computes the total electrical energy consumed from the rover battery pack over a simulation profile defined as time-velocity pairs. It accounts for energy consumed by all motors in the rover and considers the inefficiencies of transforming electrical energy to mechanical energy using a DC motor. The result represents a lower bound on energy requirements and does not consider other losses not modeled in this project.

    Parameters:
    t (np.ndarray): A numpy array of time samples.
    v (np.ndarray): A numpy array of velocity samples corresponding to the time samples.
    rover (dict): A dictionary containing rover parameters.

    Returns:
    float: The total electrical energy consumed by all motors in the rover's battery pack over the simulation profile.
    """

    # Check the input data types and dimensions
    if not isinstance(t, np.ndarray) or not isinstance(v, np.ndarray):
        raise Exception('The time samples and/or velocity samples parameters must be numpy arrays.')
    if len(t) != len(v):
        raise Exception('The time samples vector, `t`, is not equal in length to the velocity samples vector, `v`.')
    if not isinstance(rover, dict):
        raise Exception('The parameter `rover` is not a dictionary type.')

    # Calculate motor and shaft parameters
    wheelAssembly = 'wheel_assembly'
    speedReducer = 'speed_reducer'
    powerMotor = mechpower(v, rover)
    omegaWheel = v / rover[wheelAssembly]['wheel']['radius']
    omegaShaft = omegaWheel * get_gear_ratio(rover[wheelAssembly][speedReducer])
    torqueShaft = tau_dcmotor(omegaShaft, rover[wheelAssembly]['motor'])

    # Calculate efficiency for torque shafts
    efficiencyForTorqueShafts = interp1d(
        rover[wheelAssembly]['motor']['effcy_tau'],
        rover[wheelAssembly]['motor']['effcy'],
        kind='cubic'
    )(torqueShaft)

    # Integrate to calculate total electrical energy consumption
    area = 0.0
    for i in range(1, len(t)):
        deltaT = t[i] - t[i-1]
        area += 6 * (powerMotor[i] / efficiencyForTorqueShafts[i] + powerMotor[i - 1] / efficiencyForTorqueShafts[i - 1]) * deltaT / 2  # Multiply by 6 to account for all motors
    E = area

    return E


def end_of_mission_event(end_event):
    """
    Defines an event that terminates the mission simulation. The mission can end when the rover reaches a certain distance, moves for a maximum simulation time, or reaches a minimum velocity.

    Parameters:
    end_event (dict): A dictionary containing mission termination criteria, including 'max_distance' (maximum distance traveled), 'max_time' (maximum simulation time), and 'min_velocity' (minimum velocity).

    Returns:
    list: A list of events that can terminate the simulation when any of the specified conditions are met.
    """

    # Extract mission termination criteria
    mission_distance = end_event['max_distance']
    mission_max_time = end_event['max_time']
    mission_min_velocity = end_event['min_velocity']
    
    # Define conditions that can terminate the simulation
    # based on distance, time, and velocity thresholds
    distance_left = lambda t, y: mission_distance - y[1]
    distance_left.terminal = True
    
    time_left = lambda t, y: mission_max_time - t
    time_left.terminal = True
    
    velocity_threshold = lambda t, y: y[0] - mission_min_velocity
    velocity_threshold.terminal = True
    velocity_threshold.direction = -1

    # Define a list of events that can independently terminate the simulation
    events = [distance_left, time_left, velocity_threshold]
    
    # terminal indicates whether any of the conditions can lead to the
    # termination of the ODE solver. In this case all conditions can terminate
    # the simulation independently.
    
    # direction indicates whether the direction along which the different
    # conditions is reached matters or does not matter. In this case, only
    # the direction in which the velocity treshold is arrived at matters
    # (negative)
    
    return events


def simulate_rover(rover, planet, experiment, end_event):
    """
    This function integrates the trajectory of a rover using an ODE solver. It simulates the rover's motion based on the provided input parameters and mission termination criteria.

    Parameters:
    rover (dict): A dictionary containing rover specifications and characteristics.
    planet (dict): A dictionary with planet-specific information.
    experiment (dict): A dictionary defining the simulation experiment and initial conditions.
    end_event (dict): A dictionary specifying mission termination criteria, including maximum distance, maximum time, and minimum velocity.

    Returns:
    dict: The rover dictionary updated with telemetry data from the simulation.
    """

    from scipy import integrate

    # Input parameter validation
    if not isinstance(rover, dict):
        raise Exception("The first input must be a dictionary.")
    if not isinstance(planet, dict):
        raise Exception("The second input must be a dictionary.")
    if not isinstance(experiment, dict):
        raise Exception("The third input must be a dictionary.")
    if not isinstance(end_event, dict):
        raise Exception("The fourth input must be a dictionary.")
    
    # Calls the rover_dynamics function to compute derivatives
    terrain_function = lambda t , y : rover_dynamics(t,y,rover,planet,experiment)

    # Solve the ODE and capture the solution
    solution = integrate.solve_ivp(terrain_function,np.array([experiment['time_range'][0],end_event['max_time']]),experiment['initial_conditions'],method = 'BDF', events = end_of_mission_event(end_event))
    #solution = integrate.solve_ivp(terrain_function, t, y, method='BDF', events=events)

    # Calculate various telemetry metrics
    vel_avg = np.average(solution.y[0])
    distance = solution.y[1][len(solution.y[1]) - 1]
    inst_pwr = mechpower(solution.y[0], rover)
    battery_energy_sol = battenergy(solution.t, solution.y[0], rover)
    energy_per_dist = battery_energy_sol / distance
    T = solution.t
    total_distance = solution.y[1][len(solution.y[1])-1]
    

    # Update the rover dictionary with telemetry data
    rover["telemetry"] = {
        "Time": T,
        "completion_time": T[len(T)-1],
        "velocity": solution.y[0],
        "position": solution.y[1],
        "distance_traveled": total_distance,
        "max_velocity": np.max(solution.y[0]),
        "average_velocity": vel_avg,
        "power": inst_pwr,
        "battery_energy": battery_energy_sol,
        "energy_per_distance": energy_per_dist,
    }

    return rover


# Test code below
"""
# Check Outputs
# print(F_gravity(5,rover,planet)) ###SHOULD EQUAL -282
# print(F_drive(1,rover)) ###SHOULD EQUAL 7672
# print(F_net(1,5,rover,planet,0.1)) ###SHOULD EQUAL 7069###
# #Below Function Run correctly
# print(F_rolling(1,5,rover,planet,0.1)) ###SHOULD EQUAL -322###
# print(get_mass(rover)) #check step
# print(get_gear_ratio(rover['wheel_assembly']['speed_reducer'])) # check step


# #check for vector output
# arN = np.array([5,5,5])
# arS = np.array([1,1,1])
# print(F_gravity(arN,rover,planet)) ###SHOULD EQUAL -282
# print(F_drive(arS,rover)) ###SHOULD EQUAL 7672
# print(F_net(arS,arN,rover,planet,0.1)) ###SHOULD EQUAL 7069###
# #Below Function Run correctly
# print(F_rolling(arS,arN,rover,planet,0.1)) ###SHOULD EQUAL -322###
# print(get_mass(rover)) #check step
# print(get_gear_ratio(rover['wheel_assembly']['speed_reducer'])) # check step
# # test for zachary, edit

#check functions
#print(motorW(0.3,rover))  #should return 3.0625 (correct)
#print(mechpower(0.3,rover))  #should return 101.04 (correct)
#print(battenergy(np.array([0,1,2,3,4]),np.array([0.33,0.32,0.33,0.2,0.25]),rover))  #should return around 4000, returned 4280.99644 (correct)
#print(rover_dynamics(20,np.array([0.25,500]),rover,planet,experiment)) #should return np.array([2.86,0.25]) (correct)
"""

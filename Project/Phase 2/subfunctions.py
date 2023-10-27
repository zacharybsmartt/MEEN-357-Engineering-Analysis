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
    Compute the total mass of the rover, including chassis, power subsystem,
    science payload, and all wheel assemblies.

    Parameters:
    rover (dict): A dictionary containing rover specifications, including the masses of different components.

    Returns:
    float: Total mass of the rover.

    This function calculates the total mass of the rover by summing the masses of various components, including the chassis, power subsystem, science payload, and all wheel assemblies.

    Parameters:
    - rover: A dictionary with specifications for the rover, including the masses of components like the chassis, wheel assemblies, motors, and more.

    The function computes the total mass using a formula that sums the masses of the chassis, power subsystem, science payload, and all six wheel assemblies (wheels, speed reducers, and motors).

    The result is returned as a float representing the total mass of the rover.
    """
    m = (
        6 * (rover['wheel_assembly']['wheel']['mass'] +
             rover['wheel_assembly']['speed_reducer']['mass'] +
             rover['wheel_assembly']['motor']['mass']) +
        rover['chassis']['mass'] +
        rover['science_payload']['mass'] +
        rover['power_subsys']['mass']
    )
    return m

# Function to calculate the gear ratio of the speed reducer
def get_gear_ratio(speed_reducer):
    """
    Compute the gear ratio of the speed reducer.

    Parameters:
    speed_reducer (dict): A dictionary containing speed reducer specifications.

    Returns:
    float: Gear ratio of the speed reducer.

    This function calculates the gear ratio of a speed reducer based on the specifications provided in the input dictionary.

    Parameters:
    - speed_reducer: A dictionary with specifications for the speed reducer, including the gear diameter and pinion diameter.

    The function calculates the gear ratio by dividing the square of the gear diameter by the square of the pinion diameter and returns the result as a float.

    This function raises exceptions for invalid input types and for cases where the 'type' of the speed reducer is not 'reverted'.
    """
    if type(speed_reducer) is not dict:
        raise Exception("Invalid input: 'speed_reducer' must be a dictionary.")
    elif speed_reducer['type'].casefold() != "reverted":
        raise Exception("Invalid input: Invalid 'type' for 'speed_reducer'.")
    else:
        Ng = (speed_reducer['diam_gear'] / speed_reducer['diam_pinion']) ** 2
    return Ng

# Function to calculate motor shaft torque
def tau_dcmotor(omega, motor):
    """
    Calculate motor shaft torque in Nm given shaft speed in rad/s and motor specifications.

    Parameters:
    omega (scalar or numpy.ndarray): Motor shaft speed in radians per second.
    motor (dict): Dictionary containing motor specifications.

    Returns:
    float or numpy.ndarray: Motor shaft torque in Newton-meters (Nm).

    This function calculates the motor shaft torque in Newton-meters (Nm) based on the motor's shaft speed and specifications. It handles both scalar and array inputs for the shaft speed.

    Parameters:
    - omega: Motor shaft speed in radians per second, either a scalar or an array.
    - motor: Dictionary with motor specifications, including stall torque, no-load torque, and no-load speed.

    The function calculates the torque according to the motor specifications, which involves linear interpolation between stall and no-load torque. It observes the physical relationships to compute the torque values and returns the result as a scalar or numpy array, depending on the input type.

    This function raises exceptions for various cases of invalid input types and values and handles the torque calculations for different motor shaft speeds.
    """
    # Check input types
    if np.ndim(omega) != 0 and np.ndim(omega) != 1:
        raise Exception('Invalid input: Motor shaft speed (omega) must be a scalar or 1D numpy array.')

    if not isinstance(motor, dict):
        raise Exception('Invalid input: Motor properties must be a dictionary.')

    tau_stall = motor['torque_stall']
    tau_noload = motor['torque_noload']
    omega_noload = motor['speed_noload']

    if np.ndim(omega) == 0:
        tau = (tau_stall - ((tau_stall - tau_noload) / omega_noload) * omega)
        return tau

    tau = np.zeros(len(omega))

    for w in range(len(omega)):
        if omega[w] > omega_noload:
            return 0
        elif omega[w] < 0:
            return tau_stall
        else:
            tau[w] = (tau_stall - ((tau_stall - tau_noload) / omega_noload) * omega[w])

    return tau

# Function to calculate the drive force
def F_drive(omega, rover):
    """
    Calculate the drive force exerted by the wheels.

    Parameters:
    omega (scalar or numpy.ndarray): Motor shaft speed in radians per second.
    rover (dict): Dictionary containing rover specifications.

    Returns:
    numpy.ndarray: Drive force exerted by the wheels.

    This function computes the drive force exerted by the wheels of a rover. It validates the input parameters and performs the necessary calculations to derive the drive force. The function handles both scalar and array inputs for the motor shaft speed.

    Parameters:
    - omega: Motor shaft speed in radians per second, either a scalar or an array.
    - rover: Dictionary with rover specifications.

    The function calculates the gear ratio, input torque, and drive force for the wheels. The result is returned as a numpy array, and element-wise calculations are performed if omega is an array.

    This function raises exceptions for various cases of invalid input types and values and observes the physical relationships for calculating the drive force.
    """
    # Check input types
    if np.ndim(omega) != 0 and np.ndim(omega) != 1:
        raise Exception('Invalid input: Motor shaft speed (omega) must be a scalar or 1D numpy array.')

    if not isinstance(omega, (np.float64, np.intc, np.double, np.ndarray, int, float, list)):
        raise Exception('Invalid input: The argument `omega` must be a scalar value or a vector of scalars.')

    if isinstance(omega, (list, np.ndarray)):
        omegaList = np.array(omega)  # omega as an np.ndarray. Handles if omega is a list.

    if not isinstance(rover, dict):
        raise Exception('Invalid input: The argument `rover` must be a dictionary type.')

    # Calculate gear ratio and input torque
    wheelAssembly = rover['wheel_assembly']
    gearRatio = get_gear_ratio(wheelAssembly['speed_reducer'])
    torqueInput = 0

    if isinstance(omega, (np.ndarray)):
        torqueInput = np.array([tau_dcmotor(OM, wheelAssembly['motor']) for OM in omega], dtype=float)
    elif np.isscalar(omega):
        torqueInput = tau_dcmotor(omega, wheelAssembly['motor'])

    torqueOutput = torqueInput * gearRatio
    Fd = 6 * torqueOutput / wheelAssembly['wheel']['radius']

    return Fd

def F_gravity(terrain_angle, rover, planet):
    """
    Calculate the gravitational force component acting on the rover due to the terrain angle.

    Parameters:
    terrain_angle (scalar or numpy.ndarray): Terrain angle in degrees.
    rover (dict): Dictionary containing rover specifications.
    planet (dict): Dictionary containing planet-specific information.

    Returns:
    numpy.ndarray: Gravitational force component acting on the rover.

    This function calculates the gravitational force component acting on the rover as it moves across a terrain with a given angle. The function validates input parameters, ensuring they meet the expected types and criteria, and performs necessary conversions and calculations to derive the force.

    Parameters:
    - terrain_angle: Terrain angle in degrees, either a scalar or an array.
    - rover: Dictionary with rover specifications.
    - planet: Dictionary with planet-specific information.

    The function calculates the gravitational force component based on the planet's gravity, the terrain angle, and the rover's mass. The result is returned as a numpy array, and element-wise calculations are performed if the terrain angle is an array.

    This function raises exceptions for various cases of invalid input types and values and observes sign conventions for the resulting force.
    """
    # Check the parameters
    if np.ndim(terrain_angle) != 0 and np.ndim(terrain_angle) != 1:
        raise Exception('Terrain angle must be a scalar or 1D numpy array. No matrices are allowed.')
    if not isinstance(terrain_angle, (int, float, np.float64, np.intc, np.double, np.ndarray, list)):
        raise Exception('The argument `terrain_angle` must be a scalar value or a vector of scalars.')
    if isinstance(terrain_angle, (list, np.ndarray)) and not all([float(ang) >= -75 and float(ang) <= 75 for ang in terrain_angle]):
        raise Exception('The argument `terrain_angle` as a vector list must contain values between -75 and 75 degrees, inclusive.')
    if not isinstance(rover, dict) or not isinstance(planet, dict):
        raise Exception('The arguments `rover` and `planet` must be of dictionary types.')

    # Convert a scalar to a 'vector' to match the correct return argument.
    listify = terrain_angle
    if not isinstance(terrain_angle, (list, np.ndarray)):
        listify = np.array([terrain_angle])
    elif isinstance(terrain_angle, list):
        listify = np.array(terrain_angle)

    rMass = get_mass(rover)
    
    # Function to calculate acceleration due to gravity for a given angle
    accelFunc = lambda deg: planet['g'] * np.sin(degToRad(deg))

    # Calculate gravitational force for each angle in the list
    Fgt = np.array([-rMass * accelFunc(ang) for ang in listify], dtype=float)

    if np.isscalar(terrain_angle):
        return float(Fgt[0])

    return Fgt  # Observe the sign conventions.


def F_rolling(omega, terrain_angle, rover, planet, Crr):
    """
    Calculate the rolling resistance force acting on the rover.

    Parameters:
    omega (scalar or numpy.ndarray): Motor shaft speed (rad/s).
    terrain_angle (scalar or numpy.ndarray): Terrain angle in degrees.
    rover (dict): Dictionary containing rover specifications.
    planet (dict): Dictionary containing planet-specific information.
    Crr (scalar): Coefficient of rolling resistance.

    Returns:
    numpy.ndarray: Rolling resistance force acting on the rover.

    This function computes the rolling resistance force, which opposes the motion of the rover on the terrain. It validates the types and values of input parameters and ensures they meet the required criteria. The function then calculates the rolling resistance force based on the rover's characteristics, terrain angle, and the coefficient of rolling resistance.

    Parameters:
    - omega: Motor shaft speed, either a scalar or an array.
    - terrain_angle: Terrain angle in degrees, either a scalar or an array.
    - rover: Dictionary with rover specifications.
    - planet: Dictionary with planet-specific information.
    - Crr: Coefficient of rolling resistance, a positive scalar.

    The function calculates the rolling resistance force, accounting for rover mass, wheel properties, and planet gravity, while considering the terrain angle. The result is returned as a numpy array, and element-wise calculations are performed if the inputs are vectors.

    This function raises exceptions if input types, dimensions, or values do not meet the expected criteria.
    """
    # Type validation of omega and terrain_angle
    if not (isNumeric := isinstance(omega, (np.float64, np.intc, int, float))) and not isinstance(omega, np.ndarray):
        raise Exception('The parameter `omega` must be a scalar value or array.')
    if np.ndim(terrain_angle) != 0 and np.ndim(terrain_angle) != 1:
        raise Exception('omega (Motor shaft speed) must be a scalar or 1D numpy array. No matrices are allowed.')
    if (isNumeric and not isinstance(terrain_angle, (np.float64, np.intc, int, float))) or not isNumeric and not (isinstance(omega, np.ndarray) and isinstance(terrain_angle, np.ndarray)):
        raise Exception('The parameter `terrain_angle` must match the type of omega.')

    if not isNumeric:
        if len(terrain_angle) != len(omega):
            raise Exception('The parameters `terrain_angle` and `omega` must either be vectors of the same length or scalars.')
        if not all([float(ang) >= -75 and float(ang) <= 75 for ang in terrain_angle]):
            raise Exception('The argument `terrain_angle` as a vector list must contain values between -75 and 75 degrees, inclusive.')

    if not (isinstance(rover, dict) and isinstance(planet, dict)):
        raise Exception('The arguments `rover` and `planet` must be of dictionary types.')
    if Crr <= 0:
        raise Exception('The parameter `Crr` must be a positive scalar.')

    roverMass = get_mass(rover)
    wheelAssembly = 'wheel_assembly'
    speedReducer = 'speed_reducer'
    omegaWheel = omega / get_gear_ratio(rover[wheelAssembly][speedReducer])
    roverVelocity = rover[wheelAssembly]['wheel']['radius'] * omegaWheel
    planetGravity = planet['g']

    if isinstance(roverVelocity, np.ndarray):
        erfValue = np.array([erf(40 * v) for v in roverVelocity])
    elif np.isscalar(roverVelocity):
        erfValue = erf(40 * roverVelocity)

    Frr = -erfValue * Crr * roverMass * planetGravity * np.cos(degToRad(terrain_angle))

    return Frr


def F_net(omega, terrain_angle, rover, planet, Crr):
    """
    Calculate the total force (N) acting on the rover in the direction of its motion.
    
    Parameters:
    omega (scalar or numpy.ndarray): Motor shaft speed (rad/s).
    terrain_angle (scalar or numpy.ndarray): Terrain angle in degrees.
    rover (dict): Dictionary containing rover specifications.
    planet (dict): Dictionary containing planet-specific information.
    Crr (scalar): Coefficient of rolling resistance.

    Returns:
    numpy.ndarray: Total force acting on the rover.
    
    This function computes the net force on the rover, which includes contributions from rolling resistance, gravity, and driving force. It ensures that the inputs are valid and within specified limits. The total force is returned as a numpy array. If the inputs are vectors, it performs element-wise calculations; if they are scalars, it calculates the force for the single value.
    """
    # Checking conditions and evaluating for F_net...
    if np.ndim(omega) != 0 and np.ndim(omega) != 1:
        raise Exception('omega (Motor shaft speed) must be a scalar or 1D numpy array. No matrices are allowed.')
    if np.ndim(terrain_angle) != 0 and np.ndim(terrain_angle) != 1:
        raise Exception('terrain_angle must be a scalar or 1D numpy array. No matrices are allowed.')
    if not isinstance(planet, dict) or not isinstance(rover, dict):
        raise Exception("The third or fourth inputs are not dictionaries.")
    if not np.isscalar(Crr) or Crr < 0:
        raise Exception("The fifth input is not a scalar or is not positive.")
    if not isinstance(omega, np.ndarray) and not np.isscalar(omega):
        raise Exception("The first input is not a scalar or a vector.")
    else:
        if isinstance(terrain_angle, np.ndarray):  # Evaluate if the given terrain_angle is an array
            if (terrain_angle > 75).any() or (terrain_angle < -75).any():  # Degrees input
                raise Exception("The second input contains values greater than 75 or less than -75 degrees.")
            Fnet = F_rolling(omega, terrain_angle, rover, planet, Crr) + F_gravity(terrain_angle, rover, planet) + F_drive(omega, rover)
        elif np.isscalar(terrain_angle):
            if terrain_angle > 75 or terrain_angle < -75:
                raise Exception("The second input is either greater than 75 or less than -75 degrees.")
            Fnet = F_rolling(omega, terrain_angle, rover, planet, Crr) + F_gravity(terrain_angle, rover, planet) + F_drive(omega, rover)
        else:
            raise Exception("The second input is not a scalar or a vector.")

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
    dydt = np.array([round(accel, 4), y[0]])

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
    if not np.isscalar(v) and not (isinstance(v, np.ndarray) and not v.ndim == 1):
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

    def terrain_function(t, y):
        # Calls the rover_dynamics function to compute derivatives
        rover_dynamics(t, y, rover, planet, experiment)

    # Solve the ODE and capture the solution
    solution = integrate.solve_ivp(terrain_function, t, y, method='BDF', events=events)

    # Calculate various telemetry metrics
    vel_avg = np.average(solution.y[0])
    distance = solution.y[1][len(solution.y[1]) - 1]
    inst_pwr = mechpower(solution.y[0], rover)
    battery_energy_sol = battenergy(solution.t, solution.y[0], rover)
    energy_per_dist = battery_energy_sol / distance
    T = solution.t
    total_distance = np.average(solution.y[0, :]) * T[-1]

    # Update the rover dictionary with telemetry data
    rover["telemetry"] = {
        "Time": T,
        "completion_time": T[-1],
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
"""

from math import *
from define_rover import *
import numpy as np
from scipy.interpolate import interp1d

#-----------------------------------------------------
#UPDATES FOR TASK 2: CODE MUST NOT BE THOROUGHLY COMMENTED!!
# ALL FUNCTIONS MUST HAVE A DOCSTRING!!
#-----------------------------------------------------


# save space above for imports if need be
rover, planet = rover() # rover call to define all our variables
degToRad = lambda deg: deg * np.pi / 180

def get_mass(rover):
    """
    This function computes rover mass in kilograms. It accounts for
the chassis, power subsystem, science payload,
    and six wheel assemblies, which itself is comprised of a motor,
speed reducer, and the wheel itself.
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


def get_gear_ratio(speed_reducer): # should be good, just follows the formula given to us
    """
    This function computes the gear ratio of the speed reducer.
    In later project phases, you will extend this to work for various
    types of speed reducers. For now, it needs to work
    only with the simple reverted gear set described in Section 2.2
    """
    if type(speed_reducer) is not dict:
        raise Exception("Invalid input: get_gear_ratio")

    elif speed_reducer['type'].casefold() != "reverted":
        raise Exception("Invalid input: invalid type for speed_reducer")

    else:
        Ng = (speed_reducer['diam_gear'] / speed_reducer['diam_pinion']) ** 2

    return Ng


def tau_dcmotor(omega, motor):
    """
    This function returns the motor shaft torque in Nm given the shaft
    speed in rad/s and the motor specifications
    structure (which defines the no-load speed, no-load torque, and
    stall speed, among other things.
    This function must operate in a “vectorized” manner, meaning that
    if given a vector of motor shaft speeds, it
    returns a vector of the same size consisting of the corresponding
    motor shaft torques.
    """
    # Check all inputs!!!
    if np.ndim(omega) != 0 and np.ndim(omega) != 1:
        raise Exception('omega (Motor shaft speed) must be a scalar or 1D numpy array. No matricies are allowed')

    elif type(rover) != dict:
        raise Exception('Rover properties must be a dictionary')

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
            tau[w] = (tau_stall - ((tau_stall - tau_noload) /
omega_noload) * omega[w])

    return tau


def F_drive(omega, rover):
    if np.ndim(omega) != 0 and np.ndim(omega) != 1:
        raise Exception('omega (Motor shaft speed) must be a scalar or 1D numpy array. No matricies are allowed')
    if not isinstance(omega, (np.float64, np.intc, np.double,
np.ndarray, int, float, list)):
        raise Exception('The argument `omega` must be a scalar value or a vector of scalars.')
    if isinstance(omega, (list, np.ndarray)):
        omegaList = np.array(omega) # omega as an np.ndarray. Handles if omega is a list.
    if not isinstance(rover, dict):
        raise Exception('The argument `rover` must be a dictionary type.')

    #must call tau_dcmotor and get_gear_ratio
    wheelAssembly = rover['wheel_assembly']
    gearRatio = get_gear_ratio(wheelAssembly['speed_reducer'])
    torqueInput = 0
    if isinstance(omega, (np.ndarray)):
        torqueInput = np.array([tau_dcmotor(OM,wheelAssembly['motor']) for OM in omega], dtype = float) # get thetorque inputs from the motor
    elif np.isscalar(omega):
       torqueInput = tau_dcmotor(omega, wheelAssembly['motor'])
    torqueOutput = torqueInput*gearRatio #perform a transformation over the speed reducer given by the gear ratio.
    #print('torque:',torqueInput, 'gear:', gearRatio)
    Fd = 6*torqueOutput / wheelAssembly['wheel']['radius'] #find the drive force of the wheel by taking the output torque and applying it to the wheel.
    return Fd

def F_gravity(terrain_angle, rover, planet):
    #check the parameters
    if np.ndim(terrain_angle) != 0 and np.ndim(terrain_angle) != 1:
        raise Exception('omega (Motor shaft speed) must be a scalar or 1D numpy array. No matricies are allowed')
    if not isinstance(terrain_angle, (int, float, np.float64, np.intc, np.double, np.ndarray, list)):
        raise Exception('The argument `terrain_angle` must be a scalar value or a vector of scalars.')
    if isinstance(terrain_angle, (list, np.ndarray)) and not all([float(ang) >= -75 and float(ang) <= 75 for ang in terrain_angle]): # confirm that all angles are between -75 and 75 degrees
        raise Exception('The argument `terrain_angle` as a vector list must contain values between -75 and 75 degrees, inclusive.')
    if not isinstance(rover, dict) or not isinstance(planet, dict):
        raise Exception('The arguments `rover` and `planet` must be of dictionary types.')

    #convert a scalar to a 'vector' so that it matches the correct return argument.
    listify = terrain_angle
    if not isinstance(terrain_angle, (list, np.ndarray)):
        listify = np.array([terrain_angle])
    elif isinstance(terrain_angle, list):
        listify = np.array(terrain_angle)

    rMass = get_mass(rover)
    accelFunc = lambda deg: planet['g'] * np.sin(degToRad(deg)) #get planet gravity and apply a terrain angle transform to get the acceleration along the path of travel.
    Fgt = np.array([-1 * rMass*accelFunc(ang) for ang in listify],dtype = float) # apply a list transformation. Like C# .Select(). Negative to account for the true direction of the vector.

    if np.isscalar(terrain_angle):
        return float(Fgt[0])

    return Fgt #observe the sign conventions.

def F_rolling(omega, terrain_angle, rover, planet, Crr):
    # type validation of omega and terrain_angle
    if not (isNumeric := isinstance(omega, (np.float64, np.intc, int, float))) and not isinstance(omega, np.ndarray):
        raise Exception('The parameter `omega` must be a scalar value or array.')
    if np.ndim(terrain_angle) != 0 and np.ndim(terrain_angle) != 1:
         raise Exception('omega (Motor shaft speed) must be a scalar or 1D numpy array. No matricies are allowed')
    if (isNumeric and not isinstance(terrain_angle, (np.float64,np.intc, int, float))) or not isNumeric and not (isinstance(omega, np.ndarray) and isinstance(terrain_angle, np.ndarray)):
        raise Exception('The parameter `terrain_angle` must match the type of omega.')

    if not isNumeric:
        if len(terrain_angle) != len(omega):
            raise Exception('The parameters `terrain_angle` and `omega` must either be vectors of the same length or scalars.')
        if not all([float(ang) >= -75 and float(ang) <= 75 for ang in terrain_angle]):
            raise Exception('The argument `terrain_angle` as a vector list must contain values between -75 and 75 degrees, inclusive.')

    if not (isinstance(rover, dict) and isinstance(planet, dict)):
        raise Exception('The arguments `rover` and `planet` must be a dictionary.')
    if Crr <= 0:
        raise Exception('The parameter `Crr` must be a positive scalar.')

    roverMass = get_mass(rover)
    wheelAssembly = 'wheel_assembly'
    speedReducer = 'speed_reducer'
    omegaWheel = omega / get_gear_ratio(rover[wheelAssembly][speedReducer])
    roverVelocity = rover[wheelAssembly]['wheel']['radius'] * omegaWheel
    planetGravity = planet['g']
    if isinstance(roverVelocity, np.ndarray):
        erfValue = np.array([erf(40*v) for v in roverVelocity])
    elif np.isscalar(roverVelocity):
        erfValue = erf(40*roverVelocity)
    Frr =  -erfValue * Crr * roverMass * planetGravity *np.cos(degToRad(terrain_angle))
    return Frr

def F_net(omega, terrain_angle, rover, planet, Crr):
    #This function computes the total force (N) acting on the rover in the direction of its motion
    ####WHAT ABOUT SAME SIZE?####
    ####CHECKING CONDITIONS & EVALUATING FOR Fnet####
    if np.ndim(omega) != 0 and np.ndim(omega) != 1:
        raise Exception('omega (Motor shaft speed) must be a scalar or 1D numpy array. No matricies are allowed')
    if np.ndim(terrain_angle) != 0 and np.ndim(terrain_angle) != 1:
        raise Exception('omega (Motor shaft speed) must be a scalar or 1D numpy array. No matricies are allowed')
    if not isinstance(planet,dict) or not isinstance(rover,dict):
        raise Exception("The third or fourth inputs are not dictionaries.")
    if not np.isscalar(Crr) or Crr < 0:
        raise Exception("The fifth input is not a scalar or is not positive")
    if not isinstance(omega,np.ndarray) and not np.isscalar(omega):
        raise Exception("The first input is not a scalar or a vector")
    else:
        if isinstance(terrain_angle,np.ndarray): #Evaluate for if the given terrain_angle is an array
            if (terrain_angle > 75).any() or (terrain_angle < -75).any(): #degrees input
                raise Exception("The second input is more than 75 or less than -75 degrees.")
            Fnet = F_rolling(omega,terrain_angle,rover,planet,Crr) + F_gravity(terrain_angle,rover,planet) + F_drive(omega,rover)
        elif np.isscalar(terrain_angle):
            if terrain_angle > 75 or terrain_angle < -75:
                raise Exception("The second input is either greater than 75 or less than -75 degrees.")
            Fnet = F_rolling(omega,terrain_angle,rover,planet,Crr) + F_gravity(terrain_angle,rover,planet) + F_drive(omega,rover)
        else:
            raise Exception("The second input is not a scalar or a vector")

    return Fnet


def motorW(v, rover):
    """
    Compute the rotational speed of the motor shaft [rad/s] given the
    translational velocity of the rover and the rover
    dictionary.
    This function should be “vectorized” such that if given a vector
    of rover velocities it returns a vector the same size
    containing the corresponding motor speeds.
    """
    is_not_float = False
    radius = float(rover['wheel_assembly']['wheel']['radius'])
    omega = []

    if type(rover) != dict:
        raise Exception(" 'rover' must be a dictionary")
    if type(v) != int and type(v) != float:
        is_not_float = True
    if isinstance(v, np.ndarray) == False and is_not_float == False:
        raise Exception("Your first input must be a real number or a numpy array of real numbers")
    
    if type(v) == float:
        w = v / radius

    else:
        for i in v:
            b = i / radius
            omega.append(b)

        w = np.array(omega)

    return w


def rover_dynamics(t, y, rover, planet, experiment):
    """
    This function computes the derivative of the state vector (state
    vector is: [velocity, position]) for the rover given its
    current state. It requires rover and experiment dictionary input
    parameters. It is intended to be passed to an ODE
    solver.
    """
    if not isinstance(rover, dict):
        raise Exception("'rover' must be a dictionary")
    if not (isinstance(t, (int, float)) and isinstance(y, np.ndarray)):
        raise Exception("'t' must be a scalar value, and 'y' must be a 1D numpy array")
    if not isinstance(planet, dict):
        raise Exception("'planet' must be a dictionary")
    if not isinstance(experiment, dict):
        raise Exception("'experiment' must be a dictionary")

    alpha_dist = experiment['alpha_dist']
    velocity = float(y[0])
    alpha_fun = interp1d(alpha_dist, experiment['alpha_deg'], kind='cubic', fill_value='extrapolate')
    terrain_angle = float(alpha_fun(y[1]))
    o = motorW(velocity, rover)
    accel = F_net(o, terrain_angle, rover, planet, 0.1) / get_mass(rover)
    dydt = np.array([round(accel, 4), y[0]])

    return dydt


def mechpower(v, rover):
    """
    This function computes the instantaneous mechanical power output
    by a single DC motor at each point in a given
    velocity profile.
    """
    
    return P


def battenergy(t, v, rover):
    """
    This function computes the total electrical energy consumed from
    the rover battery pack over a simulation profile,
    defined as time-velocity pairs. This function assumes all 6 motors
    are driven from the same battery pack (i.e., this
    function accounts for energy consumed by all motors).
    This function accounts for the inefficiencies of transforming
    electrical energy to mechanical energy using a DC
    motor.
    In practice, the result given by this function will be a lower
    bound on energy requirements since it is undesirable to
    run batteries to zero capacity and other losses exist that are not
    modeled in this project.
    """
    
    return E


def simulate_rover(rover, planet, experiment, end_event):
    """
    This function integrates the trajectory of a rover.
    """

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

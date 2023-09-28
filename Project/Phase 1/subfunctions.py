from math import *
from define_rover import *
import numpy as np

# save space above for imports if need be
rover, planet = rover() # rover call to define all our variables

def get_mass(rover):
    """
    This function computes rover mass in kilograms. It accounts for the chassis, power subsystem, science payload,
    and six wheel assemblies, which itself is comprised of a motor, speed reducer, and the wheel itself.
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
    In later project phases, you will extend this to work for various types of speed reducers. For now, it needs to work
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
    This function returns the motor shaft torque in Nm given the shaft speed in rad/s and the motor specifications
    structure (which defines the no-load speed, no-load torque, and stall speed, among other things.
    This function must operate in a “vectorized” manner, meaning that if given a vector of motor shaft speeds, it
    returns a vector of the same size consisting of the corresponding motor shaft torques.
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
        return (tau_stall - ((tau_stall - tau_noload) / omega_noload) * omega)
    
    tau = np.zeros(len(omega))

    for w in range(len(omega)):
        if omega[w] > omega_noload:
            return 0
        elif omega[w] < 0:
            return tau_stall
        else:
            tau[w] = (tau_stall - ((tau_stall - tau_noload) / omega_noload) * omega[w])

    return tau


def F_drive(omega, rover):
    """
    take in the radius of the wheel from the rover file
    determine the power for each wheel (6 are said to be identical) by multiplying tau and w that are retured from the speed reducer in the wheel, 
    which can be obtained by inputting the omega values given
    calculate the rpm of the wheel using the same w value
    determine the drive force using the equation 30*power/(rpm*r*pi)
    multiply this driving force by 6 to account for all six wheels
    for each value in the omega list append to the array then at the end return these driving forces
    """
    if not isinstance(omega, (np.float64, np.intc, np.double, np.ndarray, list)):
        raise Exception('The argument `omega` must be a scalar value or a vector of scalars.')
    omegaList = np.ndarray([omega])
    if isinstance(omega, list):
        omegaList = np.ndarray(omega) # omega as an np.ndarray. Handles if omega is a scalar.
    if not isinstance(rover, dict):
        raise Exception('The argument `rover` must be a dictionary type.')
    
    #must call tau_dcmotor and get_gear_ratio
    wheelAssembly = rover['wheel_assembly']
    gearRatio = get_gear_ratio(wheelAssembly['speed_reducer'])
    
    torqueInput = np.ndarray([tau_dcmotor(OM, wheelAssembly['motor']) for OM in omegaList]) # get the torque inputs from the motor
    torqueOutput = torqueInput*gearRatio #perform a transformation over the speed reducer given by the gear ratio.
    
    Fd = 6*torqueOutput / wheelAssembly['wheel']['radius'] #find the drive force of the wheel by taking the output torque and applying it to the wheel.
    return Fd


def F_gravity(terrain_angle, rover, planet):
    """
    given the terrain angle acquire the mass of the mover from the rover dict along with the gravity of the planet
    the force due to gravity will be m*g*sin(terrain_angle) for the translational force due to gravity
    determine if this force is going in the same direction of the rover such as up an incline or down an incline
    if the force opposes the translational motion of the rover make negative, otherwise: positive
    """
    
    #check the parameters
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
    accelFunc = lambda deg: planet['g'] * np.sin(deg * np.pi / 180) #get planet gravity and apply a terrain angle transform to get the acceleration along the path of travel.
    Fgt = [-rMass*accelFunc(ang) for ang in listify] # apply a list transformation. Like C# .Select(). Negative to account for the true direction of the vector.
    return Fgt #observe the sign conventions.


def F_rolling(omega, terrain_angle, rover, planet, Crr):
    #This function computes the component of the force due to the rolling resistance (N) in the direction of translation
    ####WHAT ABOUT SAME SIZE?####
    ###### CHECKING CONDITIONS ########
    if not isinstance(planet,dict) or not isinstance(rover,dict):
        raise Exception("The third or fourth inputs are not dictionaries.")
    if not isinstance(omega,np.ndarray) and not np.isscalar(omega):
        raise Exception("The first input is not a scalar or a vector")
    else:
        if isinstance(terrain_angle,np.ndarray): #Evaluate for if the given terrain_angle is an array
            if (terrain_angle > 75).any() or (terrain_angle < -75).any(): #degrees input
                raise Exception("The second input is more than 75 or less than -75 degrees.")
            F_normal = get_mass(rover) * planet['g'] * np.cos((np.pi/180) * terrain_angle)
        elif np.isscalar(terrain_angle):
            if terrain_angle > 75 or terrain_angle < -75:
                raise Exception("The second input is either greater than 75 or less than -75 degrees.")
            F_normal = get_mass(rover) * planet['g'] * np.cos((np.pi/180)*terrain_angle)
        else:
            raise Exception("The second input is not a scalar or a vector")
    if not np.isscalar(Crr) or Crr < 0:
        raise Exception("The fifth input is not a scalar or is not positive")

    ####Evaluation Rolling Force#########
    omega_output = omega / get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    F_rolling_resistance = -1 * F_normal * Crr
    rover_tan_velocity = rover['wheel_assembly']['wheel']['radius'] * omega_output
    
    if np.isscalar(omega):
        Frr = F_rolling_resistance * erf(40 * rover_tan_velocity)
    else: #Then omega is an array
        Frr = np.copy(omega) #Initialize a same-size array for Frr
        for i in range(len(omega)):
            Frr[i] = F_rolling_resistance[i] * erf(40 * rover_tan_velocity[i])
    #erf(40 * roverVelocity) * Crr * roverMass * planetGravity * np.cos(terrainAngle)
    return Frr

def F_net(omega, terrain_angle, rover, planet, Crr):
    #This function computes the total force (N) acting on the rover in the direction of its motion
    ####WHAT ABOUT SAME SIZE?####
    ####CHECKING CONDITIONS & EVALUATING FOR Fnet####
    if not isinstance(planet,dict) or not isinstance(rover,dict):
        raise Exception("The third or fourth inputs are not dictionaries.")
    if not np.isscalar(Crr) or Crr < 0:
        raise Exception("The fifth input is not a scalar or is not positive")
    if not isinstance(omega,np.ndarray) or not np.isscalar(omega):
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

#DOESNT WORK DUE TO ERRORS FOUND WITHIN F_drive & F_gravity and possibly tau_dcmotor
print(F_gravity(5,rover,planet)) ###SHOULD EQUAL -282
print(F_drive(1,rover)) ###SHOULD EQUAL 7672
print(F_net(1,5,rover,planet,0.1)) ###SHOULD EQUAL 7069###
#Below Function Run correctly
print(F_rolling(1,5,rover,planet,0.1)) ###SHOULD EQUAL -322###
print(get_mass(rover)) #check step
print(get_gear_ratio(rover['wheel_assembly']['speed_reducer'])) # check step
# test for zachary, edit

from math import *
from define_rover import *

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
    return Fd


def F_gravity(terrain_angle, rover, planet):
    """
    given the terrain angle acquire the mass of the mover from the rover dict along with the gravity of the planet
    the force due to gravity will be m*g*sin(terrain_angle) for the translational force due to gravity
    determine if this force is going in the same direction of the rover such as up an incline or down an incline
    if the force opposes the translational motion of the rover make negative, otherwise: positive
    """
    return Fgt


def F_rolling(omega, terrain_angle, rover, planet, Crr):
    return Frr


def F_net(omega, terrain_angle, rover, planet, Crr):
    return Fnet

print(get_mass(rover)) #check step
print(get_gear_ratio(rover['wheel_assembly']['speed_reducer'])) # check step
# test for zachary, edit

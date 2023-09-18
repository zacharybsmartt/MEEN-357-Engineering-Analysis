from math import *
from define_rover import rover

# save space above for imports if need be
rover, planet = rover() # rover call to define all our variables

def get_mass(rover):
    m = (
        6 * (rover['wheel_assembly']['wheel']['mass'] +
        rover['wheel_assembly']['speed_reducer']['mass'] +
        rover['wheel_assembly']['motor']['mass']) +
        rover['chassis']['mass'] +
        rover['science_payload']['mass'] + 
        rover['power_subsys']['mass']
    )
    return m


def get_gear_ratio(speed_reducer):
    return Ng


def tau_dcmotor(omega, motor):
    return tau


def F_drive(omega, rover):
    return Fd


def F_gravity(terrain_angle, rover, planet):
    return Fgt


def F_rolling(omega, terrain_angle, rover, planet, Crr):
    return Frr


def F_net(omega, terrain_angle, rover, planet, Crr):
    return Fnet

print(get_mass(rover)) #check step

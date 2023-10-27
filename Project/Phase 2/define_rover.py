# we are using functions here to define our rovers in case there ends up being more than one rover with
# different specs that we need to consider, eg rover_1, rover_2, rover_3 etc


def rover():
    from numpy import array
    
    # Define the properties of the rover's wheel
    wheel = {
        'radius': 0.3,  # Radius of the wheel in meters
        'mass': 1.0    # Mass of the wheel in kilograms
    }

    # Define the properties of the speed reducer
    speed_reducer = {
        'type': 'reverted',
        'diam_pinion': 0.04,  # Diameter of the pinion in meters
        'diam_gear': 0.07,    # Diameter of the gear in meters
        'mass': 1.5           # Mass of the speed reducer in kilograms
    }

    # Define the properties of the motor
    motor = {
        'torque_stall': 170,              # Stall torque of the motor in Nm
        'torque_noload': 0,              # No-load torque of the motor in Nm
        'speed_noload': 3.8,            # No-load speed of the motor in rad/s
        'mass': 5,                      # Mass of the motor in kilograms
        'effcy_tau': array([0, 10, 20, 40, 70, 165]),  # Array of torque values for efficiency data
        'effcy': array([0, 0.55, 0.75, 0.71, 0.50, 0.05])  # Array of corresponding efficiency values
    }
    
    # Define the wheel assembly, which includes the wheel, speed reducer, and motor
    wheel_assembly = {
        'wheel': wheel,
        'speed_reducer': speed_reducer,
        'motor': motor
    }

    # Define the chassis properties, such as mass
    chassis = {'mass': 659}  # Mass of the chassis in kilograms

    # Define the mass of the science payload
    science_payload = {'mass': 75}  # Mass of the science payload in kilograms

    # Define the mass of the power subsystem
    power_subsys = {'mass': 90}  # Mass of the power subsystem in kilograms

    # Define properties of the planet, including gravitational acceleration
    planet = {'g': 3.72}  # Gravitational acceleration on the planet in m/s^2

    # Define telemetry data, initially empty
    telemetry = {
        'Time': [],
        'completion_time': 0,
        'velocity': [],
        'position': [],
        'distane_traveled': 0,
        'max_velocity': 0,
        'average_velocity': 0,
        'power': [],
        'battery_energy': 0,
        'energy_per_distance': 0
    }

    # Define the complete rover by combining its components
    rover = {
        'wheel_assembly': wheel_assembly,
        'chassis': chassis,
        'science_payload': science_payload,
        'power_subsys': power_subsys
    }
    
    # Return the rover and planet properties
    return rover, planet

# Done

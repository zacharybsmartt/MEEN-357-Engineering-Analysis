# we are using functions here to define our rovers in case there ends up being more than one rover with
# different specs that we need to consider, eg rover_1, rover_2, rover_3 etc


def rover():
    """
    Defines the rover, units for everything follow
    m, kg, Nm, rad/s, and m/s^2.
    """

    wheel = {
        'radius':0.3,
        'mass':1.0
    }

    speed_reducer = {
        'type':'reverted',
        'diam_pinion':.04,
        'diam_gear':.07,
        'mass':1.5
    }

    motor = {
        'torque_stall':170,
        'torque_noload':0,
        'speed_noload':3.8,
        'mass':5
    }

    wheel_assembly = {
        'wheel':wheel,
        'speed_reducer':speed_reducer,
        'motor':motor
    }

    chassis = {'mass':659}

    science_payload = {'mass':75}

    power_subsys = {'mass':90}

    planet = {'g':3.72}

    telemetry = {
        'Time':[],
        'completion_time':0,
        'velocity':[],
        'position':[],
        'distane_traveled':0,
        'max_velocity':0,
        'average_velocity':0,
        'power':[],
        'battery_energy':0,
        'energy_per_distance':0
    }

    rover = {
        'wheel_assembly':wheel_assembly,
        'chassis':chassis,
        'science_payload':science_payload,
        'power_subsys':power_subsys
    }
    
    return rover, planet

# Done

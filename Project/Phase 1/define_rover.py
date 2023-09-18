# we are using functions here to define our rovers in case there ends up being more than one rover with
# different specs that we need to consider, eg rover_1, rover_2, rover_3 etc


def rover():
    """Defines the rover, tbh this should be a class, will ask Dr. Thomas, units for everything follow
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

    rover = {
        'wheel_assembly':wheel_assembly,
        'chassis':chassis,
        'science_payload':science_payload,
        'power_subsys':power_subsys
    }
    
    return rover, planet

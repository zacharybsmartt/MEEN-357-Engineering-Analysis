import numpy as np
from subfunctions import *
from define_experiment import*
import matplotlib.pyplot as plt

#call the experiment1 function provided and assign the returns to experiment and end_event variables
experiment, end_event = experiment1()
#update the information for the end_event regarding the maximum time distance and minimum velocity
end_event.update({'max_distance':1000,'max_time':10000,'min_velocity':0.01})

#call simulate_rover
simulate_rover(rover,planet,experiment,end_event)
#assign the velocity values from the velocity located inside rover telemetry
velocity = rover['telemetry']['velocity']
#assign the power values to a variable from the position subsection loacated within the telemetry of the rover dictionary
power = rover['telemetry']['power']
#assign the position values to a variable from the postiion subsection located within telemetry of the rover dictionary
position = rover['telemetry']['position']
#assign the time values from the time subsection of telemetry within the rover dictionary
time = rover['telemetry']['Time']


#create the graphing layout
fig, axs = plt.subplots(3, 1)
fig.tight_layout(h_pad=4)

# Plot the rover's position over time
axs[0].plot(time,position,'bo', time, position, 'b')
axs[0].set(xlabel='Time (s)', ylabel='Position (m)', title='Position vs. Time')
# Plot the rover's velocity over time
axs[1].plot(time, velocity, 'ro', time, velocity, 'r')  # Red dots and lines
axs[1].set(xlabel='Time (s)', ylabel='Velocity (m/s)', title='Velocity vs. Time')
# Plot the rover's power consumption over time
axs[2].plot(time, power, 'go', time, power, 'g')  # Green dots and lines
axs[2].set(xlabel='Time (s)', ylabel='Power (W)', title='Power vs. Time')


# Task 9 Calculations:
# Calculate energy consumption and compare with the battery's energy
energy_consumed = battenergy(times, velos, rover)
batt1 = 0.9072 * 10**6  # Initial battery energy in joules

print('Energy of Battery (J)', batt1)
print('Energy Consumed (J)', energy_consumed)
print('Difference: ', batt1 - energy_consumed)

# Task 9 Calculations:
    # energy = integral of power
# max_time = rover['telemetry']['completion_time']

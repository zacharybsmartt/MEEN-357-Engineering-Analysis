from scipy.interpolate import *
from subfunctions import *
from define_experiment import *
import matplotlib.pyplot as plt
from define_rover import *
import numpy as np



experiment, end_event = experiment1()


end_events = end_event
simulate_rover(rover, planet, experiment, end_events)

# Define variables for graphing
times = rover['telemetry']['Time']
positions = rover['telemetry']['position']
velos = rover['telemetry']['velocity']
pows = rover['telemetry']['power']



fig, ax = plt.subplots(3, 1)
fig.tight_layout(h_pad=4)

# # graph the first equation on the top plot
ax[0].plot(times, positions, 'bo', times, positions, 'b')
ax[0].set(xlabel='Time (s)', ylabel='Position (m)',title='Position vs. Time')

# # graph the second equation on the bottom plot
ax[1].plot(times, velos, 'ro', times, velos, 'r')
ax[1].set(xlabel='Time (s)', ylabel='Velocity (m/s)',title='Velocity vs. Time')

# # graph the third equation on the bottom plot
ax[2].plot(times, pows, 'go', times, pows, 'g')
ax[2].set(xlabel='Time (s)', ylabel='Power (W)',title='Power vs. Time')

# Task 9 Calculations:
    # energy = integral of power
# max_time = rover['telemetry']['completion_time']


energy_consumed = battenergy(times, velos, rover)
batt1 =  0.9072*10**6

print('Energy of Battery (J)', batt1)
print('Energy Consumed (J)', energy_consumed)
print('Difference: ', batt1 - energy_consumed)

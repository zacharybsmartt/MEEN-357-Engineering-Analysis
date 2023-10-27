import numpy as np
from subfunctions import *
import matplotlib.pyplot as plt

# Load experiment data and simulate the rover's behavior
experiment, end_event = experiment1()
end_events = end_event
simulate_rover(rover, planet, experiment, end_events)

# Define variables for graphing telemetry data
times = rover['telemetry']['Time']
positions = rover['telemetry']['position']
velos = rover['telemetry']['velocity']
pows = rover['telemetry']['power']

# Create a 3x1 grid of subplots for the telemetry data
fig, ax = plt.subplots(3, 1)
fig.tight_layout(h_pad=4)

# Plot the rover's position over time
ax[0].plot(times, positions, 'bo', times, positions, 'b')  # Blue dots and lines
ax[0].set(xlabel='Time (s)', ylabel='Position (m)', title='Position vs. Time')

# Plot the rover's velocity over time
ax[1].plot(times, velos, 'ro', times, velos, 'r')  # Red dots and lines
ax[1].set(xlabel='Time (s)', ylabel='Velocity (m/s)', title='Velocity vs. Time')

# Plot the rover's power consumption over time
ax[2].plot(times, pows, 'go', times, pows, 'g')  # Green dots and lines
ax[2].set(xlabel='Time (s)', ylabel='Power (W)', title='Power vs. Time')

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

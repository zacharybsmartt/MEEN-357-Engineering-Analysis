import numpy as np
import matplotlib.pyplot as plt

# defining everything needed for the problem
Temperature_in = 0
Temperature_out = 200
num_nodes = 10
alpha = 2
inner_radius = alpha / 2
outer_radius = alpha
radius_values = np.linspace(inner_radius, outer_radius, num_nodes) # mesh of radial positions
dr = radius_values[1] - radius_values[0] # step size 
A = np.zeros((num_nodes, num_nodes))
B = np.zeros(num_nodes)


# Boundary conditions
A[0, 0] = 1; B[0] = Temperature_in; A[-1, -1] = 1; B[-1] = Temperature_out

for i in range(1, num_nodes - 1):
    A[i, i-1] = 1 / dr ** 2 - 1 / (2 * dr * radius_values[i])
    A[i, i] = -2 / dr ** 2
    A[i, i+1] = 1 / dr ** 2 + 1 / (2 * dr * radius_values[i])

# Solve system of linear eqns 
T = np.linalg.solve(A, B)

# Set an analytical solution function and compute 
def analytical_solution(r):
    return 200 * (1 - np.log(r / alpha) / np.log(0.5))
Temperature_analytical = analytical_solution(radius_values)

# Plot
plt.plot(radius_values, T, 'h--', label='Numerical')
plt.plot(radius_values, Temperature_analytical, 'H', label='Analytical')
plt.grid(color='g', linestyle='-', linewidth=1)
plt.xlabel('Radius')
plt.ylabel('Temperature (C)')
plt.title('Temperature Distribution in the Cylinder')
plt.legend()
plt.show()

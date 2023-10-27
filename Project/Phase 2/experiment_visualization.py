# Import necessary libraries
from scipy.interpolate import interp1d
import numpy
import define_experiment as ex
import matplotlib.pyplot as plotter

# Load experiment data and extract the necessary parameters
alpha_dist = (experiment := ex.experiment1()[0])['alpha_dist']
alpha_deg = experiment['alpha_deg']

# Create a cubic spline interpolation function
alphaInterpolationFunction = interp1d(alpha_dist, alpha_deg, kind='cubic', fill_value='extrapolate')

# Generate a range of values for terrain space
terrainSpace = numpy.linspace(min(alpha_dist), max(alpha_dist), 100)

# Create a plot figure and subplot
figure, subplot = plotter.subplots(1, 1, figsize=(8, 5))

# Set plot labels and title
subplot.set_title('Terrain Angle at specific positions')
subplot.set_xlabel('Position (m)')
subplot.set_ylabel('Terrain Angle (deg)')

# Plot the interpolation with stars at data points
subplot.plot(terrainSpace, alphaInterpolationFunction(terrainSpace), label="interpolation")  # Use marker='*'
subplot.plot(alpha_dist, alpha_deg, 'b*', label="data points")  # Plot data points with stars

# Add a legend to the plot
plotter.legend()

# Display the plot
plotter.show()

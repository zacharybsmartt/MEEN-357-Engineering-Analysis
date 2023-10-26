from scipy.interpolate import interp1d
import numpy
import define_experiment as ex
import matplotlib.pyplot as plotter

alpha_dist = (experiment := ex.experiment1()[0])['alpha_dist']
alpha_deg = experiment['alpha_deg']
alphaInterpolationFunction = interp1d(alpha_dist, alpha_deg, kind = 'cubic', fill_value='extrapolate') #fit the cubic spline
terrainSpace = numpy.linspace(min(alpha_dist), max(alpha_dist), 100)

figure, subplot = plotter.subplots(1,1, figsize=(8,5))
subplot.set_title('Terrain Angle at specific positions')
subplot.set_xlabel('Position (m)')
subplot.set_ylabel('Terrain Angle (deg)')
subplot.plot(terrainSpace, alphaInterpolationFunction(terrainSpace))
plotter.show()

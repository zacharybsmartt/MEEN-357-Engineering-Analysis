####ANALYSIS_COMBINED_TERRAIN####
###IMPORTS###
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from subfunctions import *
import matplotlib.pyplot as plt
from define_rover import*
from scipy.optimize import root_scalar

#retrieve rover and planet values
rover, planet = rover()
gear_raio = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
wheel_radius = get_gear_ratio(rover['wheel_assembly']['wheel']['radius'])
#Create necessary Crr array
Crr_array = np.linspace(0.01,0.4,25)
#Create slope array of degrees
slope_array_deg = np.linspace(-10,35,25)
#initialize v_max by copying slope_array_deg
v_max = slope_array_deg.copy()
nload_speed = rover['wheel_assembly']['motor']['speed_nload']

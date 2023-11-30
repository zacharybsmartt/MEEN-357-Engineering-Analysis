import numpy as np
from subfunctions_Phase4 import *
from define_experiment import *
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
import pickle
import copy
import sys
import csv

# the following calls instantiate the needed structs and also make some of
# our design selections (battery type, etc.)
planet = define_planet()
edl_system = define_edl_system()
mission_events = define_mission_events()
edl_system = define_chassis(edl_system,'steel')
edl_system = define_motor(edl_system,'speed')
edl_system = define_batt_pack(edl_system,'NiMH', 10)
tmax = 5000

# Overrides what might be in the loaded data to establish our desired
# initial conditions
edl_system['altitude'] = 11000    # [m] initial altitude
####CHECK###############################################################################################
edl_system['velocity'] = -587     # [m/s] initial velocity
edl_system['parachute']['deployed'] = True   # our parachute is open
edl_system['parachute']['ejected'] = False   # and still attached
edl_system['rover']['on_ground'] = False # the rover has not yet landed

experiment, end_event = experiment1()

# constraints
max_rover_velocity = -1  # this is during the landing phase
min_strength=40000
max_cost = 7.2e6
max_batt_energy_per_meter = edl_system['rover']['power_subsys']['battery']['capacity']/1000


# ******************************
# DEFINING THE OPTIMIZATION PROBLEM
# ****
# Design vector elements (in order):
#   - parachute diameter [m]
#   - wheel radius [m]
#   - chassis mass [kg]
#   - speed reducer gear diameter (d2) [m]
#   - rocket fuel mass [kg]
#

# search bounds
#x_lb = np.array([14, 0.2, 250, 0.05, 100])
#x_ub = np.array([19, 0.7, 800, 0.12, 290])


############CONFIGURE BOUNDS###################################
bounds = Bounds([14, 0.2, 250, 0.05, 100], [19, 0.7, 800, 0.12, 290])

# initial guess
x0 = [17.01, 0.70, 445.0, 0.050, 266.0]
#x0 = [19, .7, 550.0, 0.09, 250.0]

# lambda for the objective function
obj_f = lambda x: obj_fun_time(x,edl_system,planet,mission_events,tmax,
                               experiment,end_event)

# lambda for the constraint functions
#   ineq_cons is for SLSQP
#   nonlinear_constraint is for trust-constr
cons_f = lambda x: constraints_edl_system(x,edl_system,planet,mission_events,
                                          tmax,experiment,end_event,min_strength,
                                          max_rover_velocity,max_cost,max_batt_energy_per_meter)

nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 0)  # for trust-constr
ineq_cons = {'type' : 'ineq',
             'fun' : lambda x: -1*constraints_edl_system(x,edl_system,planet,
                                                         mission_events,tmax,experiment,
                                                         end_event,min_strength,max_rover_velocity,
                                                         max_cost,max_batt_energy_per_meter)}

Nfeval = 1
def callbackF(Xi):  # this is for SLSQP reporting during optimization
    global Nfeval
    if Nfeval == 1:
        print('Iter        x0         x1        x2        x3         x4           fval')
        
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}  {5: 3.6f} \
          {6: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], obj_f(Xi)))
    Nfeval += 1



# The optimizer options below are
# 'trust-constr' 
# 'SLSQP'  
# 'COBYLA' 
# You should fully comment out all but the one you wish to use

###############################################################################
# call the trust-constr optimizer --------------------------------------------#
# options = {'maxiter': 15, 
#             # 'initial_constr_penalty': 5.0,
#             # 'initial_barrier_parameter': 1.0,
#             'verbose' : 3,
#             'disp' : True}
# res = minimize(obj_f, x0, method='trust-constr', constraints=nonlinear_constraint, 
#                 options=options, bounds=bounds)
#end call to the trust-constr optimizer -------------------------------------#
###############################################################################

###############################################################################
# call the SLSQP optimizer ---------------------------------------------------#
# bounds = Bounds([16, 0.4, 250, 0.05, 200], [19, 0.7, 800, 0.12, 290])  # these can be changed
# options = {'maxiter': 50,
#             'disp' : True}
# res = minimize(obj_f, x0, method='SLSQP', constraints=ineq_cons, bounds=bounds, 
#                 options=options, callback=callbackF)
# end call to the SLSQP optimizer --------------------------------------------#
###############################################################################

###############################################################################
# call the COBYLA optimizer --------------------------------------------------#
cobyla_bounds = [[14, 19], [0.3, 0.7], [100, 500], [0.05, 0.3], [100, 290]]
# # #construct the bound s in the form of constraints
cons_cobyla = []
for factor in range(len(cobyla_bounds)):
    lower, upper = cobyla_bounds[factor]
    l = {'type': 'ineq',
          'fun': lambda x, lb=lower, i=factor: x[i] - lb}
    u = {'type': 'ineq',
          'fun': lambda x, ub=upper, i=factor: ub - x[i]}
    cons_cobyla.append(l)
    cons_cobyla.append(u)
    cons_cobyla.append(ineq_cons)  # the rest of the constraints
#reduce the maxiteration size
options = {'maxiter':30,
            'disp' : True}
res = minimize(obj_f, x0, method='COBYLA', constraints=cons_cobyla, options=options)
# end call to the COBYLA optimizer -------------------------------------------#
###############################################################################

###############################################################################
# call the differential evolution optimizer ----------------------------------#
# popsize=5 # define the population size
# maxiter=10 # define the maximum number of iterations
# res = differential_evolution(obj_f, bounds=bounds, constraints=nonlinear_constraint, 
#                               popsize=popsize, maxiter=maxiter, disp=True, polish = False) 
# end call the differential evolution optimizer ------------------------------#
###############################################################################


# check if we have a feasible solution 
c = constraints_edl_system(res.x,edl_system,planet,mission_events,tmax,experiment,
                           end_event,min_strength,max_rover_velocity,max_cost,
                           max_batt_energy_per_meter)

#Iterate through the constraints to check conditions

feasible = True
if c[0] > 1e-15:
    feasible = False
    print('The Distance Restriction Did Not Meet The Constraint')
    
elif c[1] > 0:
    feasible = False
    print('The Strength Restriction Did Not Meet The Constraint')

elif c[2] > 0:
    feasible = False
    print('The Velocity Restriction Did Not Meet The Constraint')

elif c[3] > 0:
    feasible = False
    print('The Cost Restriction Did Not Meet The Constraint')

elif c[4] > 0:
    feasible = False
    print('The Battery Restriction Was Not Met')

elif res.x[0] > 19*1.000001 or res.x[0] < 14*0.999999:
    feasible = False
    print('The Parachute Restriction Exceeded the Bounds')

elif res.x[1] > 0.7*1.000001 or res.x[1] < 0.2*0.999999:
    feasible = False
    print('The Wheel Radius Restriction Exceeded the Bounds')

elif res.x[2] > 800*1.000001 or res.x[2] < 250*0.999999:
    feasible = False
    print('The Chassis Mass Restriction Exceeded the Bounds')

elif res.x[3] > 0.12*1.000001 or res.x[3] < 0.05*0.999999:
    feasible = False
    print('The Gear Diameter Restriction Exceeded the Bounds')

elif res.x[4] > 290*1.000001 or res.x[4] < 100*0.999999:
    feasible = False
    print('The Fuel Mass Restriction Exceeded the Bounds')


if feasible:
    xbest = res.x
    fbest = res.fun
else:  # nonsense to let us know this did not work
    xbest = [99999, 99999, 99999, 99999, 99999]
    fval = [99999]
    raise Exception('Solution not feasible, exiting code...')
    sys.exit()

# The following will rerun your best design and present useful information
# about the performance of the design
# This will be helpful if you choose to create a loop around your optimizers and their initializations
# to try different starting points for the optimization.
edl_system = redefine_edl_system(edl_system)

edl_system['parachute']['diameter'] = xbest[0]
edl_system['rover']['wheel_assembly']['wheel']['radius'] = xbest[1]
edl_system['rover']['chassis']['mass'] = xbest[2]
edl_system['rover']['wheel_assembly']['speed_reducer']['diam_gear'] = xbest[3]
edl_system['rocket']['initial_fuel_mass'] = xbest[4]
edl_system['rocket']['fuel_mass'] = xbest[4]

# *****************************************************************************
# These lines save your design for submission for the rover competition.
# You will want to change them to match your team information.

edl_system['team_name'] = 'Land or Die'  # change this to something fun for your team (or just your team number)
edl_system['team_number'] = 3    # change this to your assigned team number (also change it below when saving your pickle file)

# This will create a file that you can submit as your competition file.
with open('FA23_501team03.pickle', 'wb') as handle:
    pickle.dump(edl_system, handle, protocol=pickle.HIGHEST_PROTOCOL)
# *****************************************************************************

#del edl_system
#with open('challenge_design_team9999.pickle', 'rb') as handle:
#    edl_system = pickle.load(handle)

time_edl_run,_,edl_system = simulate_edl(edl_system,planet,mission_events,tmax,True)
time_edl = time_edl_run[-1]

edl_system['rover'] = simulate_rover(edl_system['rover'],planet,experiment,end_event)
time_rover = edl_system['rover']['telemetry']['completion_time']

total_time = time_edl + time_rover
 
edl_system_total_cost=get_cost_edl(edl_system)


print('----------------------------------------')
print('----------------------------------------')
print('Optimized parachute diameter   = {:.6f} [m]'.format(xbest[0]))
print('Optimized rocket fuel mass     = {:.6f} [kg]'.format(xbest[4]))
print('Time to complete EDL mission   = {:.6f} [s]'.format(time_edl))
print('Rover velocity at landing      = {:.6f} [m/s]'.format(edl_system['rover_touchdown_speed']))
print('Optimized wheel radius         = {:.6f} [m]'.format(xbest[1])) 
print('Optimized d2                   = {:.6f} [m]'.format(xbest[3])) 
print('Optimized chassis mass         = {:.6f} [kg]'.format(xbest[2]))
print('Time to complete rover mission = {:.6f} [s]'.format(time_rover))
print('Time to complete mission       = {:.6f} [s]'.format(total_time))
print('Average velocity               = {:.6f} [m/s]'.format(edl_system['rover']['telemetry']['average_velocity']))
print('Distance traveled              = {:.6f} [m]'.format(edl_system['rover']['telemetry']['distance_traveled']))
print('Battery energy per meter       = {:.6f} [J/m]'.format(edl_system['rover']['telemetry']['energy_per_distance']))
print('Total cost                     = {:.6f} [$]'.format(edl_system_total_cost))
print('----------------------------------------')
print('----------------------------------------')


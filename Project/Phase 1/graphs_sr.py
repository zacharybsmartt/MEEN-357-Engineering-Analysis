from define_rover import *
from matplotlib.pyplot import *
from numpy import *
from subfunctions import tau_dcmotor, get_gear_ratio

rover, planet = rover()
gear_ratio = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
omega = linspace(0,3.8,20) #increase last input to make more smooth


tau_out = zeros(len(omega))
omega_out = zeros(len(omega))
for w in range(len(omega)):  
    omega_out[w] = omega[w] / gear_ratio
    tau_out[w] = gear_ratio * tau_dcmotor(omega[w], rover['wheel_assembly']['motor'])

subplot(3,1,1)
plot(tau_out,omega_out)
xlabel('Speed Recuer Torque [Nm]')
ylabel('Speed Reducer Speed [rad/s]')

subplot(3,1,2)
power_out = multiply(omega_out, tau_out)
plot(tau_out,power_out)
xlabel('Speed Reducer Torque [Nm]')
ylabel('Speed Reducer Power [W]')

subplot(3,1,3)
plot(omega_out,power_out)
xlabel('Speed Reducer Speed [rad/s]')
ylabel('Speed Reducer Power [W]')

# If you need to see graphs for report
# show()

# Done

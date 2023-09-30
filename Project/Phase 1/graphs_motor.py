from matplotlib.pyplot import plot, xlabel, ylabel, subplot, show, figure
from define_rover import *
from subfunctions import tau_dcmotor
from numpy import linspace, zeros, multiply

rover, planet = rover()
omega = linspace(0,3.8,20) #increase last input to make more smooth


tau = zeros(len(omega))

for w in range(len(omega)):   
    tau[w] = tau_dcmotor(omega[w], rover['wheel_assembly']['motor'])
figure(figsize=(8,6.5))
subplot(3,1,1)
plot(tau,omega)
xlabel('Motor Shaft Torque [Nm]')
ylabel('Motor Shaft Speed [rad/s]')

subplot(3,1,2)
power = multiply(omega, tau)
plot(tau,power)
xlabel('Motor Shaft Torque [Nm]')
ylabel('Motor Power [W]')

subplot(3,1,3)
plot(omega,power)
xlabel('Motor Shaft Speed [rad/s]')
ylabel('Motor Power [W]')

# If you need to see graphs for report
show()

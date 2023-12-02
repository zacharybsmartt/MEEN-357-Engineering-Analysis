import numpy as np
import matplotlib.pyplot as plt

# Objective function
def objective_function(x, y):
    return 2.25 * x * y + 1.75 * y - 1.5 * x**2 - 2 * y**2

# Gradient of the objective function
def gradient(x, y):
    df_dx = 2.25 * y - 3 * x
    df_dy = 2.25 * x + 1.75 - 4 * y
    return np.array([df_dx, df_dy])

# Gradient Ascent method
def gradient_ascent(x, y, step_size, tolerance):
    iterations = 0
    while True:
        # Compute the gradient
        grad = gradient(x, y)

        # Update design variables
        x += step_size * grad[0]
        y += step_size * grad[1]

        # Update iteration count
        iterations += 1

        # Check convergence
        if np.abs(objective_function(x, y) - objective_function(x - step_size * grad[0], y - step_size * grad[1])) <= tolerance:
            break

    return x, y, objective_function(x, y), iterations

# Arrow function to add arrows to the plot
def add_arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(facecolor='black', arrowstyle='->'))

x0, y0 = 5, 5

# Part (a)
step_size_a = 0.1
tolerance_a = 1e-8

x_a, y_a, f_star_a, iterations_a = gradient_ascent(x0, y0, step_size_a, tolerance_a)

x_vals = np.linspace(-6, 6, 100)
y_vals = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = objective_function(X, Y)

plt.contour(X, Y, Z, levels=20)
plt.scatter(x_a, y_a, color='red', label=f'Optimal Point: ({x_a:.3f}, {y_a:.3f})')
plt.title('Gradient Ascent with Step Size 0.1')
plt.xlabel('x')
plt.ylabel('y')

# Add arrow for part (a)
add_arrow(plt.gca(), (x0, y0), (x_a, y_a))

plt.legend()
plt.show()

# Display results
print(f"Optimal function value (f*): {f_star_a:.5f}")
print(f"Optimal design variables (x*, y*): ({x_a:.5f}, {y_a:.5f})")
print(f"Number of iterations: {iterations_a}")

# Part (b)
step_size_b = 0.25
tolerance_b = 1e-8

x_b, y_b, f_star_b, iterations_b = gradient_ascent(x0, y0, step_size_b, tolerance_b)

plt.contour(X, Y, Z, levels=20)
plt.scatter(x_b, y_b, color='red', label=f'Optimal Point: ({x_b:.3f}, {y_b:.3f})')
plt.title('Gradient Ascent with Step Size 0.25')
plt.xlabel('x')
plt.ylabel('y')

# Add arrow for part (b)
add_arrow(plt.gca(), (x0, y0), (x_b, y_b))

plt.legend()
plt.show()

# Display results
print(f"Optimal function value (f*): {f_star_b:.5f}")
print(f"Optimal design variables (x*, y*): ({x_b:.5f}, {y_b:.5f})")
print(f"Number of iterations: {iterations_b}")

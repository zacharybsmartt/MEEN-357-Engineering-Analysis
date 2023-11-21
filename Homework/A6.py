import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.metrics import r2_score


x = np.array([3.0, 4.5, 7.0, 9.0])
fx = np.array([2.5, 1.0, 2.5, 0.5])
xnew = np.linspace(3.0, 9.0, 100)
xd = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
yd = [38.824, 31.382, 106.615, 84.958, 136.192, 221.649, 239.815,
326.167, 433.752, 527.402]


def QuadraticSpline(x, fx):
    
    #sets up points to base spline and matrixs 
    nsegs = len(x)-1
    A = np.zeros((3*nsegs, 3*nsegs))
    B = np.zeros((3*nsegs, 1))
    
    #sets up 0th order continuity and termination conditions
    for i in range(nsegs):
        
        A[2*i, 3*i] = x[i] ** 2
        A[2*i, 3*i+1] = x[i]
        A[2*i, 3*i+2] = 1
        
        A[2*i+1, 3*i] = x[i+1] **2
        A[2*i+1, 3*i+1] = x[i+1]
        A[2*i+1, 3*i+2] = 1
        
        B[2*i] = fx[i]
        B[2*i+1] = fx[i+1]

    #Sets up 1st ortder continuity
    for i in range(nsegs-1):
        
        A[nsegs*2 + i, 3*i] = 2 * x[i+1]
        A[nsegs*2 + i, 3*i+1] = 1
        A[nsegs*2 + i, 3*(1+i)] = -2 * x[i+1]
        A[nsegs*2 + i, 3*(1+i)+1] = -1

    #sets up derivative assumption
    A[-1, 0] = 1
    B[-1] = 0
    
    #solve the system
    coeffs = la.solve(A, B)
    
    
    #print(A)
    #print(B)
    return coeffs


def QuadraticSplineInterp(x, xnew, coeffs):
    xnew = np.atleast_1d(xnew) # makes sure 'len' will work if a scalar is input
    coeffs = coeffs.ravel()
    fapprox = np.zeros((len(xnew), 1))

    for i in range(len(xnew)):
        if xnew[i] < x[0] or xnew[i] > x[-1]:
            raise Exception('Extrapolation not allowed!')
        else:
            for j in range(0, len(x)-1):        # equation to interpolate using
                if x[j] <= xnew[i] <= x[j+1]:  # coefficients derived
                    fapprox[i, 0] = coeffs[3*j]*(xnew[i])**2 + coeffs[3*j+1]*xnew[i] + coeffs[3*j+2]
    return fapprox


def BackSubstitutionSolver(U, b):
    # Check if U is a square matrix
    if U.shape[0] != U.shape[1]:
        raise ValueError("Matrix U must be square.")

    # Check if b is a column vector
    if len(b.shape) != 2 or b.shape[1] != 1:
        raise ValueError("Vector b must be a column vector.")

    # Check if U and b have the same number of rows
    if U.shape[0] != b.shape[0]:
        raise ValueError("Matrix U and vector b must have the same number of rows.")

    # Check if U is an upper triangular matrix
    if not np.all(np.triu(U) == U):
        raise ValueError("Matrix U must be an upper triangular matrix.")

    n = U.shape[0]
    sol = np.zeros_like(b, dtype=float)

    # Perform back substitution
    for i in range(n-1, -1, -1):
        sol[i, 0] = (b[i, 0] - np.dot(U[i, i+1:], sol[i+1:, 0])) / U[i, i]

    return sol


def LinRegress(xd, yd, n):
    # Check if lengths of xd and yd are equal
    if len(xd) != len(yd):
        raise ValueError("Lengths of xd and yd must be equal.")

    # Check if a regression function of degree n can be created
    if len(xd) <= n:
        raise ValueError("Not enough data points to create a regression function of degree {}.".format(n))

    # Create the Vandermonde matrix
    A = np.vander(xd, n+1)

    # Solve the linear system of equations
    coeffs = np.linalg.solve(A.T @ A, A.T @ yd)

    return coeffs


coeffs = QuadraticSpline(x, fx)
fapprox = QuadraticSplineInterp(x, xnew, coeffs)

# plotting everything together

plt.plot(xnew, fapprox, 'b-', linewidth=3, label='Quadratic Spline')
plt.plot(x, fx, '*r', markersize=18, label='Input Points')
plt.axis([2, 10, 0, 3.5])
plt.legend(loc='lower left')
plt.title('Quadratic Spline')
plt.show()


U = np.array([
    [2, 4, 8, 2],
    [0, 2, 1, 9],
    [0, 0, 3, 5],
    [0, 0, 0, 8]
], dtype=float)

b = np.array([
    [0],
    [1],
    [6],
    [5]
], dtype=float)

sol = BackSubstitutionSolver(U, b)
# print("Solution using BackSubstitutionSolver:\n", sol)

np_solution = np.linalg.solve(U, b)
# print("Solution using np.linalg.solve:\n", np_solution)


# Test the function with the provided data
xd = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
yd = np.array([38.824, 31.382, 106.615, 84.958, 136.192, 221.649, 239.815, 326.167, 433.752, 527.402])

# Fit the data for a 6th order polynomial
coeffs_6th = LinRegress(xd, yd, 6)

# Fit the data for a cubic polynomial
coeffs_cubic = LinRegress(xd, yd, 3)

# Fit the data for a linear polynomial
coeffs_linear = LinRegress(xd, yd, 1)

# Calculate the predicted values for each fit
yd_pred_6th = np.polyval(coeffs_6th, xd)
yd_pred_cubic = np.polyval(coeffs_cubic, xd)
yd_pred_linear = np.polyval(coeffs_linear, xd)

# Calculate R-squared values for each fit
r2_6th = r2_score(yd, yd_pred_6th)
r2_cubic = r2_score(yd, yd_pred_cubic)
r2_linear = r2_score(yd, yd_pred_linear)

# Choose the best fit based on R-squared values
best_fit = max([(r2_6th, '6th Order Polynomial'), (r2_cubic, 'Cubic Polynomial'), (r2_linear, 'Linear Polynomial')],
               key=lambda x: x[0])

print("Best Fit: {}".format(best_fit[1]))
print("R-squared value: {}".format(best_fit[0]))

# Plot the results
plt.scatter(xd, yd, label='Data Points')
plt.plot(xd, yd_pred_6th, label='6th Order Polynomial Fit')
plt.plot(xd, yd_pred_cubic, label='Cubic Polynomial Fit')
plt.plot(xd, yd_pred_linear, label='Linear Polynomial Fit')
plt.xlabel('xd')
plt.ylabel('yd')
plt.legend()
plt.title('Linear Regression with Polynomials')
plt.show()

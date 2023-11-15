import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
A = np.zeros((6, 6))
b = [2.5, 1.0, 1.0, 2.5, 2.5, .5]
fx = [2.5, 1, 2.5, .5]
x = [3, 4.5, 7, 9]

A[0, 0], A[0, 1] = x[0], 1
A[1, 0], A[1, 1] = x[1], 1
print(A)
A[2, 2], A[2, 3] = x[1], 1
A[3, 2], A[3, 3] = x[2], 1
print(A)
A[4, 4], A[4, 5] = x[2], 1
A[5, 4], A[5, 5] = x[3], 1

coeffs = np.linalg.solve(A, b)
print(coeffs)

    
def LinearSpline(x, fx):
    nseg = len(x) - 1
    neq = 2 * nseg
    A = np.zeros((neq, neq))
    b = np.zeros(neq)

    for i in range(nseg):
        A[2 * i, 2 * i],     A[2 * i, 2 * i + 1] = x[i], 1
        A[2 * i + 1, 2 * i], A[2 * i + 1, 2 * i + 1] = x[i + 1], 1
        b[2 * i], b[2 * i + 1] = fx[i], fx[i + 1]

    print(A)
    print(b)


def LinearSplineInterp(x, xtest, coeffs):

    fapprox = np.zeros(len(xtest))
    for i in enumerate(xtest):
        for j in enumerate(x - 1):
            if (xtest[i] >= x[j] and xtest[i] <= x[j+1]):
                segn = j
                fapprox[i] = coeffs[2 * segn] * xtest[i] + coeffs[2 * segn]
    return fapprox



LinearSpline(x, fx) 

xtest = np.linspace(3.0, 9.0, 100)
fapprox = LinearSplineInterp(x, xtest, coeffs)
plt.figure()
plt.plot(x, fx, '*', label = 'data')

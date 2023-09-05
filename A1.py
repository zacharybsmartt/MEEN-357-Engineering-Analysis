import random as rand
import numpy as np
import matplotlib.pyplot as plt

#Task 1 Below this line

def sort_ascending(unsorted_list):
    """This function takes a list of numbers and sorts them in ascending order (smallest to largest). """
    sorted_list = []
    while unsorted_list:
        minimum = unsorted_list[0]
        for data in unsorted_list:
            if data < minimum:
                minimum = data
        sorted_list.append(minimum)
        unsorted_list.remove(minimum)

    return sorted_list


def A1_task1():
    """This function generates a list of random numbers and uses the sort_ascending function to sort them in ascending order (smallest to largest)."""
    random_list = [rand.randint(0,1000) for i in range(20)]
    sorted_list = sort_ascending(random_list)
    print(sorted_list)

    return sorted_list


A1_task1()

# Task 1 Completed
# Task 2 Below

def A1_task2():
    """This function creates Numpy arrays and performs some basic linear algebra."""
    A = np.array([[1, 2, 3, 4], 
                 [2, 4, 6, 8], 
                 [3, 6, 9, 12], 
                 [4, 8, 12, 16]]) # initialize array
    A_new = A.copy() # create a copy of array
    
    C = A[::-1, ::-1] # make columns rows and rows columns of A
    C_new = C.copy() # create a copy of C

    b = np.array([[-4],
                 [3],
                 [-2],
                 [1]]) # initialize column vector

    d = -1 * b[::-1] # flip the column vector and multiply its entirety by -1

    x = (A + np.transpose(A)) * b - C * d # calculate x

    A_new[:, [2, 3]] = A_new[:, [3, 2]] # swap columns via array indexing, what was now 1 is now 3 and vice versa
    C_new[[0, 2]] = C_new[[2, 0]] # more swapping

    x_new = (A_new + np.transpose(A_new)) * b - C_new * d # calculation of x_new
    M = A + np.transpose(A_new) # calculation of M

    M_max = np.max(M) # max
    M_min = np.min(M) # min

    return x, x_new, M_max, M_min


A1_task2()

# Task 2 Completed
# Task 3 Below

true_val = 6.737947 * 10 ** -3

def gen_approx(n):
    """This function generates our approximations given the nth term"""
    result = 1 # value for term 0
    x = -5

    for terms in range(n):
        if n == 1:
            result = 1 - x
            print("Term {}: {}" .format(n, result))
            return result
        else:
            
            print("Term {}: {}" .format(n, result))



import random as rand
import numpy as np

#Task 1 Below

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
                 [4, 8, 12, 16]])
    A_new = A.copy()
    
    C = A[::-1, ::-1]
    C_new = C.copy()

    b = np.array([[-4],
                 [3],
                 [-2],
                 [1]])

    d = -1 * b[::-1]

    x = (A + np.transpose(A)) * b - C * d

    A_new[:, [2, 3]] = A_new[:, [3, 2]]
    C_new[[0, 2]] = C_new[[2, 0]]

    x_new = (A_new + np.transpose(A_new)) * b - C_new * d
    M = A + np.transpose(A_new)

    M_max = np.max(M)
    M_min = np.min(M)

    return x, x_new, M_max, M_min


A1_task2()

# Task 2 Completed
# Task 3 Below



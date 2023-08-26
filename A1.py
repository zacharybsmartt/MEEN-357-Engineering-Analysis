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
    
    C = A[::-1, ::-1]

    b = 


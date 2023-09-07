from math import *
import matplotlib.pyplot as plt


def Approx_1(x, n):
    Approximation_1 = 0
    Approximation_1_list, True_error, Approximate_error = [], [], []
    true_value = 0.006737947
    for i in range(n):
        if i == 0:
            Approximation_1 += (x ** i)/factorial(i)
            Approximation_1_list.append(Approximation_1)
            True_err = ((abs(true_value-(Approximation_1)))/true_value)*100
            True_error.append(True_err)
            Approximate_err = ((abs(Approximation_1_list[i-1]-Approximation_1))/Approximation_1_list[i-1])*100
            Approximate_error.append(Approximate_err)
            print("After", i, "terms: \ne^x = ", Approximation_1, "\nTrue error:", True_err, 
                  "\nApproximate Error:",Approximate_err, "\n")
            
        elif i % 2 == 0:
            Approximation_1 += (x ** i)/factorial(i)
            Approximation_1_list.append(Approximation_1)
            True_err = ((abs(true_value-(Approximation_1)))/true_value)*100
            True_error.append(True_err)
            Approximate_err = ((abs(Approximation_1_list[i-1]-Approximation_1))/Approximation_1_list[i-1])*100
            Approximate_error.append(Approximate_err)
            print("After", i, "terms: \ne^x = ", Approximation_1, "\nTrue error:", True_err, 
                "\nApproximate Error:",Approximate_err, "\n")
            
        else:
            Approximation_1 -= (x ** i)/factorial(i)
            Approximation_1_list.append(Approximation_1)
            True_err = ((abs(true_value-(Approximation_1)))/true_value)*100
            True_error.append(True_err)
            Approximate_err = ((abs(Approximation_1_list[i-1]-Approximation_1))/Approximation_1_list[i-1])*100
            Approximate_error.append(Approximate_err)
            print("After", i, "terms: \ne^x = ", Approximation_1, "\nTrue error:", True_err, 
                "\nApproximate Error:",Approximate_err, "\n")
    
    print(Approximation_1_list)
    print(True_error)
    print(Approximate_error)

    plt.plot(list(range(0,n)),Approximation_1_list)
    plt.title("Method 1")
    plt.xlabel("n iterations")
    plt.ylabel("Approximation")
    plt.show()

    plt.plot(list(range(0,n)),True_error)
    plt.title("Method 1")
    plt.xlabel("n iterations")
    plt.ylabel("True Error")
    plt.show()

    plt.plot(list(range(0,n)),Approximate_error)
    plt.title("Method 1")
    plt.xlabel("n iterations")
    plt.ylabel("Approximate Error")
    plt.show()


def Approx_2(x, n):
    Approximation_2 = 0
    Approximation_2_list, True_error, Approximate_error = [], [], []
    true_value = 0.006737947
    for i in range(n):
        Approximation_2 += (x ** i)/factorial(i)
        Approximation_2_list.append(Approximation_2)
        True_err = ((abs(true_value-(1/Approximation_2)))/true_value)*100
        True_error.append(True_err)
        Approximate_err = ((abs(Approximation_2_list[i-1]-Approximation_2))/Approximation_2_list[i-1])*100
        Approximate_error.append(Approximate_err)
        print("After", i, "terms: \ne^x = ", 1/Approximation_2, "\nTrue error:", True_err, 
            "\nApproximate Error:",Approximate_err, "\n")
    
    print([1/i for i in Approximation_2_list])
    print(True_error)
    print(Approximate_error)

    plt.plot(list(range(0,n)),[1/i for i in Approximation_2_list])
    plt.title("Method 2")
    plt.xlabel("n iterations")
    plt.ylabel("Approximation")
    plt.show()

    plt.plot(list(range(0,n)),True_error)
    plt.title("Method 2")
    plt.xlabel("n iterations")
    plt.ylabel("True Error")
    plt.show()

    plt.plot(list(range(0,n)),Approximate_error)
    plt.title("Method 2")
    plt.xlabel("n iterations")
    plt.ylabel("Approximate Error")
    plt.show()


Approx_1(5, 20)

Approx_2(5, 20)

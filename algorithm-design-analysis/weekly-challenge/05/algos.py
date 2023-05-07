import time
import numpy as np
from matplotlib import pyplot as plt


def algo_A(arr: list):
    ''' Runs in \Theta(n^2) time
        input a size n array
    '''
    if len(arr) > 1:

        mid = len(arr) // 2

        L = arr[:mid]

        R = arr[mid:]

        algo_A(L)

        algo_A(R)

        for i in range(len(arr)):
            for j in range(len(arr)):
                arr[i] += arr[j] / 1000


def algo_B(arr: list):
    ''' Runs in \Theta(n^2) time
        input a size n array
    '''
    if len(arr) > 1:

        mid = len(arr) // 2
        quad = mid // 2

        first = arr[:mid]

        second = arr[mid:]

        third = arr[quad:mid + quad]

        fourth = arr[:quad] + arr[mid + quad:]

        algo_B(first)

        algo_B(second)

        algo_B(third)

        algo_B(fourth)

        m = min(arr)  # Min function does linear amount of work

        for i in range(len(arr)):
            arr[i] = m


def algo_C(arr: list):
    ''' Runs in \Theta(n \lg n) time
        Source: https://www.geeksforgeeks.org/merge-sort/
        input a size n array
    '''
    if len(arr) > 1:

        mid = len(arr) // 2

        L = arr[:mid]

        R = arr[mid:]

        algo_C(L)

        algo_C(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1


def algo_D(arr: list):
    ''' Runs in \Theta(n \lg n) time
        input a size n array
    '''
    if len(arr) > 1:

        mid = len(arr) // 2
        quad = mid // 2

        first = arr[:quad]

        second = arr[quad:mid]

        third = arr[mid:mid + quad]

        fourth = arr[mid + quad:]

        algo_D(first)

        algo_D(second)

        algo_D(third)

        algo_D(fourth)

        m = max(arr)  # Max function does linear amount of work

        for i in range(len(arr)):
            arr[i] = m


def Compare(start: int, end: int, step: int):
    algoA = []
    algoB = []
    algoC = []
    algoD = []

    for n in range(start, end, step):
        set_n = list(np.random.randint(1, n, n))

        start_time = time.perf_counter()
        arrA = set_n.copy()
        algo_A(arrA)
        end_time = time.perf_counter()
        algoA.append(end_time - start_time)

        start_time = time.perf_counter()
        arrB = set_n.copy()
        algo_B(arrB)
        end_time = time.perf_counter()
        algoB.append(end_time - start_time)

        start_time = time.perf_counter()
        arrC = set_n.copy()
        algo_C(arrC)
        end_time = time.perf_counter()
        algoC.append(end_time - start_time)

        start_time = time.perf_counter()
        arrD = set_n.copy()
        algo_D(arrD)
        end_time = time.perf_counter()
        algoD.append(end_time - start_time)

    X = np.arange(start, end, step)

    plt.plot(X, algoA, color='r', label='Algo A')
    plt.plot(X, algoB, color='b', label='Algo B')
    plt.plot(X, algoC, color='m', label='Algo C')
    plt.plot(X, algoD, color='g', label='Algo D')

    plt.title('plot of run time of algorithms with n')
    plt.xlabel('n')
    plt.ylabel('Run time')
    plt.legend()
    plt.show()


Compare(10, 1001, 10)

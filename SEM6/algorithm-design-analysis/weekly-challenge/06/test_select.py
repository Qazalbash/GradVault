import random
import time

import numpy as np
import pytest
from matplotlib import pyplot as plt


def select(A: [int], k: int):  # -> int:
    '''
    Returns the k-th largest element in A using divide and conquer.

    Parameters:
    - A: the sequence of numbers, with possible duplicates.
    - k: return the k-th largest element

    Constraints:
    - 1 <= k <= len(A)

    Return:
    the k-th largest element in A.
    '''
    S_L, S_V, S_R = [], [], []

    v = random.choice(A)

    for i in A:
        if i > v:
            S_L.append(i)
        elif i < v:
            S_R.append(i)
        else:
            S_V.append(i)

    if k <= len(S_L):
        return select(S_L, k)
    if k <= len(S_L) + len(S_V):
        return v
    return select(S_R, k - len(S_L) - len(S_V))


def select_sort(A: [int], k: int):  # -> int:
    '''
    Returns the k-th largest element in A using sorting.

    Parameters:
    - A: the sequence of numbers, with possible duplicates.
    - k: return the k-th largest element

    Constraints:
    - 1 <= k <= len(A)

    Return:
    the k-th largest element in A.
    '''
    return sorted(A, reverse=True)[k - 1]


@pytest.mark.parametrize("_", range(100))
def test_select(_):
    A = [random.randrange(10**3) for _ in range(random.randint(100, 10**5))]
    random.shuffle(A)
    k = random.randrange(1, len(A) + 1)
    assert sorted(A, reverse=True)[k - 1] == select(A, k)


def Compare(start: int, end: int, step: int):
    div = []
    norm = []

    for n in range(start, end, step):
        set_n = list(np.random.randint(1, n, n))
        # arr = list(range(1, n + 1))
        k = random.randint(1, n)
        A = set_n.copy()

        start_time = time.perf_counter()
        select(A, k)
        end_time = time.perf_counter()
        div.append(end_time - start_time)

        start_time = time.perf_counter()
        select_sort(A, k)
        end_time = time.perf_counter()
        norm.append(end_time - start_time)

    X = np.arange(start, end, step)

    plt.plot(X, div, color='r', label='select (divide and conquer algorithm)')
    plt.plot(X, norm, color='b', label='select_sort')

    plt.title('plot of run time of algorithms with n')
    plt.xlabel('n')
    plt.ylabel('Run time')
    plt.legend()
    plt.show()


Compare(100, 500000, 100)
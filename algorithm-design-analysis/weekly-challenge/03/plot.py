from matplotlib import pyplot as plt
from math import log10
import numpy as np


def plot_sequences(n: int) -> None:
    '''
    We have not used the function num_sequence(n) in this function.
    But the logic of the both functions is same.
    '''
    seq = [1, 1]
    for i in range(2, n):
        seq.append(seq[i - 1] + seq[i - 2])

    X = np.arange(0, n, 100)

    log_seq = [log10(i) for i in seq[:n:100]]
    log_seq_1 = [(10 * 2 * (5**(1 / 2))) * log10(i) for i in seq[:n:100]]
    log_seq_2 = [(1 / 20) * log10(i) for i in seq[:n:100]]
    log_seq_3 = [(5**(1 / 2)) * log10(i) for i in seq[:n:100]]

    plt.plot(X, log_seq, color='r', label='f(n)')
    plt.plot(X, log_seq_1, color='b', label='f1(n)')
    plt.plot(X, log_seq_2, color='g', label='f2(n)')
    plt.plot(X, log_seq_3, color='c', label='g(n)')

    plt.title(f'plot of log10(num_sequence(n)) for n = 0 to {n}')
    plt.xlabel('n')
    plt.ylabel('log10(num_sequence(n))')
    plt.legend()
    plt.show()

    # x = []
    # for i in range(n):
    #     x.append(i+1)

    # xpoints = np.arange(0, n, 1)
    # ypoints = np.array(seq)

    # plt.plot(xpoints, ypoints)
    # plt.show()


if __name__ == '__main__':
    plot_sequences(20002)
    # plot_sequences(200)

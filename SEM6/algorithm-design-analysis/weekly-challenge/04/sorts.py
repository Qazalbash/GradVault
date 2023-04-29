import math
import random

import numpy as np
from matplotlib import pyplot as plt


class Sort:

    def bubble_sort(self, array: list):  # -> int:
        """bubble sort algorithm

        Args:
            array (list): Input array

        Returns:
            int: number of comparisions
        """
        count = 0
        for i in range(len(array)):
            for j in range(len(array)):
                count += 1
                if array[i] < array[j]:
                    array[i], array[j] = array[j], array[i]
        return count

    def insertion_sort(self, array: list):  # -> int:
        """insertion sort algorithm

        Args:
            array (list): Input array

        Returns:
            int: number of comparisions
        """
        count = 0
        for i in range(1, len(array)):
            j = i
            count += 1
            while j > 0 and array[j] < array[j - 1]:
                array[j], array[j - 1] = array[j - 1], array[j]
                j -= 1
                count += 1
        return count

    def merge_sort(self, array: list):  # -> int:
        """merge sort algorithm

        Args:
            array (list): Input array

        Returns:
            int: number of comparisions
        """
        count = 0
        count = self.merge(array, count)
        return count

    def merge(self, array: list, count: int):  # -> int:
        """merge of merge sort algorithm

        Args:
            array (list): Input array
            count (int): number of comparisions

        Returns:
            int: number of comparisions
        """
        if len(array) > 1:
            r = len(array) >> 1
            L = array[:r]
            M = array[r:]
            self.merge(L, count)
            self.merge(M, count)

            i = j = k = 0
            while i < len(L) and j < len(M):
                count += 1
                if L[i] < M[j]:
                    array[k] = L[i]
                    i += 1
                else:
                    array[k] = M[j]
                    j += 1
                k += 1

            while i < len(L):
                count += 1
                array[k] = L[i]
                i += 1
                k += 1

            while j < len(M):
                count += 1
                array[k] = M[j]
                j += 1
                k += 1
        return count

    def selection_sort(self, array: list):  # -> int:
        """selection sort algorithm

        Args:
            array (list): Input array

        Returns:
            int: number of comparisions
        """
        count = 0
        for i in range(len(array)):
            min = i
            for j in range(i + 1, len(array)):
                count += 1
                if array[min] > array[j]:
                    min = j

            array[i], array[min] = array[min], array[i]
        return count

    def shell_sort(self, array: list):  # -> int:
        """shell sort algorithm

        Args:
            array (list): Input array

        Returns:
            int: number of comparisions
        """
        count = 0
        n = len(array)
        gap = n >> 1

        while gap > 0:
            j = gap

            while j < n:
                i = j - gap

                while i >= 0:
                    count += 1
                    if array[i + gap] > array[i]:
                        break
                    else:
                        array[i + gap], array[i] = array[i], array[i + gap]

                    i = i - gap
                j += 1
            gap >>= 1

        return count

    def Average_comparision(self, n: int):  # -> dict:
        """Average comparision of all algorithms

        Args:
            n (int): size of array

        Returns:
            dict: average comparision of all algorithms
        """
        arrays = []
        set_n = list(range(1, n + 1))

        for perm in range(10 * math.ceil(math.log2(n))):
            arr = set_n.copy()
            random.shuffle(arr)
            arrays.append(arr)

        bubble = []
        insertion = []
        selection = []
        merge = []
        shell = []

        for perm in range(len(arrays)):
            array = arrays[perm].copy()
            bubble.append(self.bubble_sort(array))

            array = arrays[perm].copy()
            insertion.append(self.insertion_sort(array))

            array = arrays[perm].copy()
            selection.append(self.selection_sort(array))

            array = arrays[perm].copy()
            merge.append(self.merge_sort(array))

            array = arrays[perm].copy()
            shell.append(self.shell_sort(array))

        averages = {
            "bubble": sum(bubble) / len(bubble),
            "insertion": sum(insertion) / len(insertion),
            "selection": sum(selection) / len(selection),
            "merge": sum(merge) / len(merge),
            "shell": sum(shell) / len(shell)
        }
        return averages

    def plot(self, start: int, end: int, step: int):  # -> None:
        bubble = []
        insertion = []
        selection = []
        merge = []
        shell = []

        for n in range(start, end, step):
            avgs = self.Average_comparision(n)
            bubble.append(avgs["bubble"])
            insertion.append(avgs["insertion"])
            selection.append(avgs["selection"])
            merge.append(avgs["merge"])
            shell.append(avgs["shell"])

        X = np.arange(start, end, step)

        plt.plot(X, bubble, color='r', label='bubble')
        plt.plot(X, insertion, color='b', label='insertion')
        plt.plot(X, selection, color='g', label='selection')
        plt.plot(X, merge, color='c', label='merge')
        plt.plot(X, shell, color='k', label='shell')

        plt.title('plot of avagerage comparisions with n')
        plt.xlabel('n')
        plt.ylabel('average comparisions')
        plt.legend()
        plt.show()


if __name__ == "__main__":

    s = Sort()
    s.plot(10, 501, 10)

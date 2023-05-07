from random import randint
from time import perf_counter

import plotly.offline as pyo
from plotly import graph_objects as go


def readFile(filename):
    return open(filename, "r").read().split(".")


def getWordsFromLineList(listOfLines):
    return [
        word.lower()
        for word in (" ".join(listOfLines)).split(" ")
        if word.isalnum()
    ]


def countFrequency(lst):
    frequency = {}
    for word in getWordsFromLineList(lst):
        frequency[word] = frequency.get(word, 0) + 1
    return [(word, frequency[word]) for word in frequency.keys()]


def mergeSort(lst, column, ascending=True):
    if len(lst) > 1:
        mid = len(lst) // 2
        leftLst, rightLst = lst[:mid], lst[mid:]
        mergeSort(leftLst, column, ascending)
        mergeSort(rightLst, column, ascending)
        i = j = k = 0
        if ascending:
            while i < len(leftLst) and j < len(rightLst):
                if leftLst[i][column] < rightLst[j][column]:
                    lst[k] = leftLst[i]
                    i += 1
                else:
                    lst[k] = rightLst[j]
                    j += 1
                k += 1
        else:
            while i < len(leftLst) and j < len(rightLst):
                if leftLst[i][column] > rightLst[j][column]:
                    lst[k] = leftLst[i]
                    i += 1
                else:
                    lst[k] = rightLst[j]
                    j += 1
                k += 1
        while i < len(leftLst):
            lst[k] = leftLst[i]
            i += 1
            k += 1
        while j < len(rightLst):
            lst[k] = rightLst[j]
            j += 1
            k += 1


def partition(arr, column, low, high, ascending):
    pivot = randint(low, high + 1)
    arr[pivot], arr[high] = arr[high], arr[pivot]
    i = low
    if ascending:
        for j in range(low, high + 1):
            if arr[j][column] <= arr[high][column]:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
    else:
        for j in range(low, high + 1):
            if arr[j][column] >= arr[high][column]:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
    return i - 1


def quickSort(arr, column, low, high, ascending=True):
    if low < high:
        pivot = partition(arr, column, low, high, ascending)
        quickSort(arr, column, low, pivot - 1, ascending)
        quickSort(arr, column, pivot + 1, high, ascending)


def partition1(arr, column, low, high, ascending):
    pivot = low
    arr[pivot], arr[high] = arr[high], arr[pivot]
    i = low
    if ascending:
        for j in range(low, high + 1):
            if arr[j][column] <= arr[high][column]:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
    else:
        for j in range(low, high + 1):
            if arr[j][column] >= arr[high][column]:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
    return i - 1


def quickSort1(arr, column, low, high, ascending=True):
    if low < high:
        pivot = partition1(arr, column, low, high, ascending)
        quickSort1(arr, column, low, pivot - 1, ascending)
        quickSort1(arr, column, pivot + 1, high, ascending)


def partition2(arr, column, low, high, ascending):
    pivot = (low + high) // 2
    arr[pivot], arr[high] = arr[high], arr[pivot]
    i = low
    if ascending:
        for j in range(low, high + 1):
            if arr[j][column] <= arr[high][column]:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
    else:
        for j in range(low, high + 1):
            if arr[j][column] >= arr[high][column]:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
    return i - 1


def quickSort2(arr, column, low, high, ascending=True):
    if low < high:
        pivot = partition2(arr, column, low, high, ascending)
        quickSort2(arr, column, low, pivot - 1, ascending)
        quickSort2(arr, column, pivot + 1, high, ascending)


def partition3(arr, column, low, high, ascending):
    pivot = high
    arr[pivot], arr[high] = arr[high], arr[pivot]
    i = low
    if ascending:
        for j in range(low, high + 1):
            if arr[j][column] <= arr[high][column]:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
    else:
        for j in range(low, high + 1):
            if arr[j][column] >= arr[high][column]:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
    return i - 1


def quickSort3(arr, column, low, high, ascending=True):
    if low < high:
        pivot = partition3(arr, column, low, high, ascending)
        quickSort3(arr, column, low, pivot - 1, ascending)
        quickSort3(arr, column, pivot + 1, high, ascending)


timeLst1, timeLst2, timeLst3, timeLst4 = [], [], [], []

filename = input("Enter filename: ")
arr = countFrequency(getWordsFromLineList(readFile(f"{filename}")))
ascending = False
column = 1
for pointer in range(len(arr)):

    piratedArr1 = arr[:pointer]
    piratedArr2 = piratedArr1.copy()
    piratedArr3 = piratedArr1.copy()
    piratedArr4 = piratedArr1.copy()

    start = perf_counter()
    quickSort1(piratedArr1, column, 0, len(piratedArr1) - 1, ascending)
    end = perf_counter()
    timeLst1.append(end - start)

    start = perf_counter()
    quickSort2(piratedArr2, column, 0, len(piratedArr1) - 1, ascending)
    end = perf_counter()
    timeLst2.append(end - start)

    start = perf_counter()
    quickSort3(piratedArr3, column, 0, len(piratedArr1) - 1, ascending)
    end = perf_counter()
    timeLst3.append(end - start)

    start = perf_counter()
    mergeSort(piratedArr4, column, ascending)
    end = perf_counter()
    timeLst4.append(end - start)

size = [i for i in range(len(arr))]

layout = go.Layout(
    title=
    f"Quick Sort VS Merge Sort, file2.txt, ascending = {ascending}, column = {column}",
    plot_bgcolor="rgb(230,230,230)",
    showlegend=True)

trace1 = go.Scatter(x=size,
                    y=timeLst1,
                    mode="lines+markers",
                    name="quick sort (pivot = low)")
trace2 = go.Scatter(x=size,
                    y=timeLst2,
                    mode="lines+markers",
                    name="quick sort (pivot = mid)")
trace3 = go.Scatter(x=size,
                    y=timeLst3,
                    mode="lines+markers",
                    name="quick sort (pivot = high)")
trace4 = go.Scatter(x=size, y=timeLst4, mode="lines+markers", name="merge sort")

fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
pyo.plot(
    fig,
    filename=
    f"Quick Sort VS Merge Sort, file1.txt, ascending = {ascending}, column = {column}",
)

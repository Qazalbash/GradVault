def partition(A, start, end):
    pivot, pindex = A[end], start
    for i in range(start, end):
        if A[i] <= pivot:
            A[i], A[pindex] = A[pindex], A[i]
            pindex += 1
    A[end], A[pindex] = A[pindex], A[end]
    return pindex


def quickSort(A, start, end):
    if start >= end:
        return
    pIndex = partition(A, start, end)
    quickSort(A, start, pIndex - 1)
    quickSort(A, pIndex + 1, end)

def merge(A, L, R):
    nL, nR, i, j, k = len(L), len(R), 0, 0, 0
    while i < nL and j < nR:
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
        k += 1
    while i < nL:
        A[k] = L[i]
        i += 1
        k += 1
    while j < nR:
        A[k] = R[j]
        j += 1
        k += 1


def mergeSort(A):
    n = len(A)
    if n > 1:
        mid = n // 2
        L, R = A[:mid], A[mid:]
        mergeSort(L)
        mergeSort(R)
        merge(A, L, R)

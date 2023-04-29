def mergeSort(lst, n):
    tempLst = [0] * n
    return mSort(lst, tempLst, 0, n - 1)


def mSort(lst, tempLst, left, right):
    count = 0
    if left < right:
        mid = (left + right) // 2
        count += (mSort(lst, tempLst, left, mid) +
                  mSort(lst, tempLst, mid + 1, right) +
                  merger(lst, tempLst, left, mid, right))
    return count


def merger(lst, tempLst, left, mid, right):
    i, j, k, count = left, mid + 1, left, 0
    while i <= mid and j <= right:
        if lst[i] <= lst[j]:
            tempLst[k] = lst[i]
            i += 1
        else:
            tempLst[k] = lst[j]
            count += mid - i + 1
            j += 1
        k += 1
    while i <= mid:
        tempLst[k] = lst[i]
        k += 1
        i += 1
    while j <= right:
        tempLst[k] = lst[j]
        k += 1
        j += 1
    for dumy in range(left, right + 1):
        lst[dumy] = tempLst[dumy]
    return count


lst = [2, 4, 1, 3, 5]
inrversion = mergeSort(lst, len(lst))
print(f"Number of inversions are {inrversion}")

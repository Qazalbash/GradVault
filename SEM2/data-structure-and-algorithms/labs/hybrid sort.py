def binary_search(lst, val, low, high):
    if low > high:
        return low
    elif low == high:
        if lst[low] > val:
            return low
        return low + 1
    guess = (low + high) // 2
    if lst[guess] < val:
        return binary_search(lst, val, guess+1, high)
    elif lst[guess] > val:
        return binary_search(lst, val, low, guess-1)
    return guess
    
def BinaryInsertionSort(lst):
    for i in range(1, len(lst)):
        val = lst[i]
        j = binary_search(lst, val, 0, i-1)
        lst = lst[:j] + [val] + lst[j:i] + lst[i+1:]
    return lst

print(BinaryInsertionSort(lst))

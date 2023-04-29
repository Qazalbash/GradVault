def binary_search_recursive_modified(lst, item, low, high):
    guess = (low + high) // 2
    if high < low:
        return guess + 1
    elif item == lst[guess]:
        return guess
    elif item < lst[guess]:
        return binary_search_recursive_modified(lst, item, low, guess - 1)
    elif item > lst[guess]:
        return binary_search_recursive_modified(lst, item, guess + 1, high)

print(binary_search_recursive_modified(lst, item, low, high))

def binary_search_recursive(lst, item, low, high):
    guess = (low + high) // 2
    if high < low:
        return -1
    elif item == lst[guess]:
        return guess
    elif item < lst[guess]:
        return binary_search_recursive(lst, item, low, guess - 1)
    elif item > lst[guess]:
        return binary_search_recursive(lst, item, guess + 1, high)    

print(binary_search_recursive(lst, item, low, high))

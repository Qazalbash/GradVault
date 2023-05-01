def insertion_sort(lst):
    for i in range(1, len(lst)):
        key, j = lst[i], i - 1
        while j >= 0 and key < lst[j]:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = key
        print(lst)


insertion_sort(lst)

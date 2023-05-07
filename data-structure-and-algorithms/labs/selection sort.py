def selection_sort(lst):
    for j in range(len(lst)):
        local_minimum = j
        for i in range(local_minimum + 1, len(lst)):
            if lst[local_minimum] > lst[i]:
                local_minimum = i
        if local_minimum != j:
            lst[local_minimum], lst[j] = lst[j], lst[local_minimum]
        print(lst)


selection_sort(lst)

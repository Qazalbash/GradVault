def selection_sort(lst):
    for j in range(len(lst)):
        local_minimum = j
        for i in range(local_minimum + 1, len(lst)):
            if lst[local_minimum] > lst[i]:
                local_minimum = i
        if local_minimum != j:
            lst[local_minimum], lst[j] = lst[j], lst[local_minimum]
    return lst


def sort_matrix_by_row(lst):
    for row in range(len(lst)):
        lst[row] = selection_sort(lst[row])
    return lst


print(sort_matrix_by_row(lst))

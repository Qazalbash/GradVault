def insertion_sort(lst):
    for i in range(1, len(lst)):
        key, j = lst[i], i - 1
        while j >= 0 and key < lst[j]:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j+1] = key

def smallest_absdiff_pairs(lst):
    insertion_sort(lst)
    pairs, minimum = {}, lst[1] - lst[0]
    for i in range(len(lst) - 1):
        if abs(lst[i+1]-lst[i]) < minimum:
            minimum = abs(lst[i+1]-lst[i])
        pairs[abs(lst[i+1]-lst[i])] = pairs.get(abs(lst[i+1]-lst[i]), []) + [(lst[i], lst[i+1])]
    return pairs[minimum]

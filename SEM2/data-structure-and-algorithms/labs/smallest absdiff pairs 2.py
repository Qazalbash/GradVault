def sorter(lst):
    start = 0
    while start != (len(lst)-1):
        a = start
        count = start
        for k in lst[start:]:
            if k < lst[a]:
                a = count
            count+=1
        lst[start],lst[a] = lst[a],lst[start]
        start+=1
    return lst

def smallest_absdiff_pairs(lst):
    sorted_lst = sorter(lst)
    stor = []
    diff_stor = 0
    lowest_sum = abs(lst[1] - lst[0])
    for k in range(len(lst)-1):
        if lowest_sum > abs(lst[k+1] - lst[k]):
            lowest_sum = abs(lst[k+1] - lst[k])
    for k in range(len(lst)-1):
        if lowest_sum == abs(lst[k+1] - lst[k]):
            stor.append((lst[k], lst[k+1]))
    return stor

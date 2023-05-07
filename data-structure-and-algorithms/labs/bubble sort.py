def bubble_sort(lst):
    if len(lst) < 2:
        print(lst)
    for j in range(len(lst) - 1):
        flag = 0
        for i in range(len(lst) - j - 1):
            if lst[i] > lst[i + 1]:
                lst[i], lst[i + 1] = lst[i + 1], lst[i]
                flag = 1
        if flag == 0:
            break
        print(lst)


bubble_sort(lst)

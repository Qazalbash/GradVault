def b_search(lst, item):
    low, high = 0, len(lst) - 1
    while True:
        guess = (low + high) // 2
        if high < low:
            return -1
        elif item == lst[guess]:
            return guess
        elif item < lst[guess]:
            high = guess - 1
        elif item > lst[guess]:
            low = guess + 1


def length_of_line(points_list, length):
    return b_search([
        round(((i[0][0] - i[1][0])**2 + (i[0][1] - i[1][1])**2)**0.5, 2)
        for i in points_list
    ], length)


print(length_of_line(points_list, length))

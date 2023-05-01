from math import log

index, count = 0, 0


def sum_digits(n):
    global index, count
    if n == 0:
        return 0
    elif index == log(n, 10) // 1 + 1:
        return int(count)
    count += (n % 10 * (index + 1) - n % 10 * index) / 10**index
    index += 1
    return sum_digits(n)

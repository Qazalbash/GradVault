from math import log

n = int(input())
base = int(input())


def base_converter(n, base, num="", index=int(log(n, base))):
    if index == -1:
        return num
    digit = int((n % base**(index + 1) - n % base**index) / base**index)
    if base == 16 and digit > 9:
        digit = {10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F"}[digit]
    num += str(digit)
    return base_converter(n, base, num, index - 1)

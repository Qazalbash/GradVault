import math

result, I = "", int(math.log(n, base))


def base_converter(n, base):
    global I, result
    number = n // (base * I) - base * (n // (base * (I + 1)))
    hexa = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}
    if number >= 10 and base == 16:
        number = hexa[number % 10]
    result += str(number)
    if I == 0:
        return result
    I -= 1
    return base_converter(n, base)

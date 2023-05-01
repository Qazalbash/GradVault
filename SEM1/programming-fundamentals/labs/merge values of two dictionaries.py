d1, d2 = [eval(input()) for _ in range(2)]


def merge_value(d1, d2):
    d3 = {}
    for (key1, value1), (key2, value2) in zip(d1.items(), d2.items()):
        d3[value1] = [key1, key2]
        d3[value2] = [key1, key2]
    return d3


print(sorted(merge_value(d1, d2).items()))

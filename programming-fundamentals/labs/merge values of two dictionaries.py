def merge_value(d1, d2):
    d3 = {}
    for (key1, value1), (key2, value2) in zip(d1.items(), d2.items()):
        d3[value1] = [key1, key2]
        d3[value2] = [key1, key2]
    return d3

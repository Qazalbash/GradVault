t = [
    16.4, 17.1, 17.0, 15.6, 16.2, 14.8, 16.0, 15.6, 17.3, 17.4, 15.6, 15.7,
    17.2, 16.6, 16.0, 15.3, 15.4, 16.0, 15.8, 17.2, 14.6, 15.5, 14.9, 16.7, 16.3
]


def mean(t):
    return sum(t) / len(t)


def variance(t, mu):
    return sum((x - mu)**2 for x in t) / len(t)


print(mean(t))
print(variance(t, mean(t)))
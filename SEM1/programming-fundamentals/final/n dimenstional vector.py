import math

v1 = eval(input())
v2 = eval(input())


def get_length(v):
    length = 0
    for i in range(len(v)):
        length += v[i]**2
    return length**0.5


def dot_product(v1, v2):
    prod = 0
    for i in range(len(v1)):
        prod += v1[i] * v2[i]
    return prod


print(f"V1 Length: {(get_length(v1))}")
print(f"V2 Length: {(get_length(v2))}")
if len(v1) == len(v2):
    print(f"Dot Product(V1,V2): {(dot_product(v1, v2))}")
else:
    print("Dot Product(V1,V2): Lengths of the two vectors must be same")
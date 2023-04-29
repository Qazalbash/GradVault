import math

v1 = eval(input())
v2 = eval(input())


def get_length(v):
    length = 0
    for i in range(len(v)):
        length += v[i] ** 2
    return length ** 0.5


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

# laiba's code

# import math

# v1 = eval(input())
# v2 = eval(input())


# def get_length(v, i):
#     lst = list(v)
#     v_sum = sum(map(lambda x: x * 2, lst))
#     v_length = math.sqrt(v_sum)
#     print(f"V{i} Length:", v_length)


# def dot_product(v1, v2):
#     if len(v1) != len(v2):
#         print("Dot Product(V1,V2): Lengths of the two vectors must be same")
#     else:
#         d = 0
#         for component in range(len(v1)):
#             d += v1[component] * v2[component]
#         print("Dot Product(V1,V2):", d)


# get_length(v1, 1)
# get_length(v2, 2)
# dot_product(v1, v2)

# Dua's code


# def dot_product(v1, v2):
#     v1_final = 0
#     v2_final = 0
#     v3 = 0
#     for i, j in zip(v1, v2):
#         v1_final += i ** 2
#         v2_final += j ** 2
#         v3 += i * j
#     print("V1 Length:", v1_final ** 0.5)
#     print("V2 Length:", v2_final ** 0.5)
#     if len(v1) == len(v2):
#         print("Dot Product(V1,V2):", v3)
#     else:
#         print("Dot Product(V1,V2): Lengths of the two vectors must be same")


# dot_product(v1, v2)

# hammad's code

# v1 = eval(input())
# v2 = eval(input())


# def get_length(v):
#     length = int()
#     for component in v:
#         length += component ** 2
#     return length ** (1 / 2)


# def dot_product(v1, v2):
#     scalar_product = int()
#     if len(v1) == len(v2):
#         for component_number in range(len(v1)):
#             scalar_product += v1[component_number] * v2[component_number]
#         return scalar_product
#     elif len(v1) != len(v2):
#         return "Lengths of the two vectors must be same"


# print(
#     f"V1 Length: {get_length(v1)}\nV2 Length: {get_length(v2)}\nDot Product(V1,V2): {dot_product(v1,v2)}"
# )

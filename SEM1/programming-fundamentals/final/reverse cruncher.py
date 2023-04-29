msg = input()


def reverse(word):
    nword = " ".join(
        [j[0].upper() + j[1:] for j in [i[::-1] for i in word.lower().split(" ")]]
    )
    print(nword)


reverse(msg)

# Hammad's Code

# msg = input()


# def reverse_string(string):
#     if string == "":
#         return ""
#     else:
#         return reverse_string(string[1:]) + string[0]


# def reverse(word):
#     return " ".join(
#         [
#             (j[0].upper() + j[1:])
#             for j in ([reverse_string(i) for i in word.lower().split()])
#         ]
#     )


# print(reverse(msg))

import math
import os
import random
import re
import sys


def minimalOperations(words):
    change = []
    for i in range(len(words)):
        w, index, cd = list(words[i]), 0, True
        for j in range(len(w) - 1):
            if cd == False:
                cd = True
            elif w[j] == w[j + 1]:
                cd = False
                index += 1
        change.append(index)
    return change


if __name__ == "__main__":
    fptr = open(os.environ["OUTPUT_PATH"], "w")

    words_count = int(input().strip())

    words = []

    for _ in range(words_count):
        words_item = input()
        words.append(words_item)

    result = minimalOperations(words)

    fptr.write("\n".join(map(str, result)))
    fptr.write("\n")

    fptr.close()

# safi's code

# import math
# import os
# import random
# import re
# import sys

# def minimalOperations(words):
#     arr = []
#     for letter in words:
#         errors = 0
#         parity = 0
#         for index in range(len(letter) - 1):
#             if parity == 1:
#                 parity = 0
#                 continue
#             if letter[index] == letter[index+1]:
#                 parity = 1
#                 errors += 1
#         arr.append(errors)
#     return arr

# if __name__ == '__main__':
#     fptr = open(os.environ['OUTPUT_PATH'], 'w')

#     words_count = int(input().strip())

#     words = []

#     for _ in range(words_count):
#         words_item = input()
#         words.append(words_item)

#     result = minimalOperations(words)

#     fptr.write('\n'.join(map(str, result)))
#     fptr.write('\n')

#     fptr.close()

# hammad's code

# def minimalOperations(words):
#     required_array = list()
#     for word in words:
#         element = int()
#         index = 1
#         while index < len(word):
#             if word[index-1] == word[index]:
#                 element += 1
#                 index += 1
#             index += 1
#         required_array.append(element)
#     return required_array


# if __name__ == '__main__':
#     fptr = open(os.environ['OUTPUT_PATH'], 'w')

#     words_count = int(input().strip())

#     words = []

#     for _ in range(words_count):
#         words_item = input()
#         words.append(words_item)

#     result = minimalOperations(words)

#     fptr.write('\n'.join(map(str, result)))
#     fptr.write('\n')

#     fptr.close()

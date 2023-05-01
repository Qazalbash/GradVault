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

from typing import Any


def minimalOperations(words: Any) -> Any:
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
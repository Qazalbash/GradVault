def count_letters(word):
    w = {}
    for i in word.lower():
        w[i] = w.get(i, 0) + 1
    return w


print(sorted(count_letters(input()).items()))

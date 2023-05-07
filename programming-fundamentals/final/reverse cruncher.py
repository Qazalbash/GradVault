def reverse(word: str) -> None:
    nword = " ".join([
        j[0].upper() + j[1:]
        for j in [i[::-1] for i in word.lower().split(" ")]
    ])
    print(nword)
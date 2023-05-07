def biggest_bucket(corpus):
    corpus = corpus.lower().replace('"', " ")
    NC = ""
    for Q in corpus:
        if Q.isalpha() or Q == " ":
            NC += Q
    first_letters = [j[0] for j in [i for i in NC.split(" ") if i.isalpha()]]
    letter_count = {}
    for k in first_letters:
        letter_count[k] = letter_count.get(k, 0) + 1
    mode = max(letter_count.values())
    modes = []
    for key in letter_count.keys():
        if letter_count[key] == mode:
            modes.append([key, mode])
    return sorted(modes)

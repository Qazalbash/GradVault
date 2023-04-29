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


print(biggest_bucket(input()))

# hammad's code


# def biggest_bucket(corpus):
#     freq_dict = dict()
#     req_list = list()
#     max_freq = int()
#     for word in corpus.lower().split():
#         index = int()
#         while True:
#             if 97 <= ord(word[index]) <= 122:
#                 freq_dict[word[index]] = freq_dict.get(word[index], 0) + 1
#                 break
#             index += 1
#     for key, value in freq_dict.items():
#         if (key == list(freq_dict.keys())[0]) or (value > max_freq):
#             req_list = [[key, value]]
#             max_freq = value
#         elif value == max_freq and key != list(freq_dict.keys())[0]:
#             req_list.append([key, value])
#             max_freq = value
#     for i in range(len(req_list)):
#         for j in range(len(req_list) - i - 1):
#             if req_list[j][0] > req_list[j + 1][0]:
#                 temp = req_list[j]
#                 req_list[j] = req_list[j + 1]
#                 req_list[j + 1] = temp
#     return req_list


# print(biggest_bucket(input()))

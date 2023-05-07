def transmogrify():

    num = int(input())
    encryption = {}

    for i in range(num):
        alpha = input().split(" ")
        encryption[alpha[0]] = alpha[1]

    word = input()
    return "".join([encryption[j] for j in word])

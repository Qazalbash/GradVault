def getTopIndex_UnimodelSequence(unimodel_sequence):
    low, high = 0 , len(unimodel_sequence) - 1
    while True:
        guess = (low + high) // 2
        if high < low:
            return -1
        if unimodel_sequence[guess-1] <= unimodel_sequence[guess] < unimodel_sequence[guess+1]:
            low = guess + 1
        elif unimodel_sequence[guess-1] > unimodel_sequence[guess]:
            high = guess - 1
        elif unimodel_sequence[guess-1] <= unimodel_sequence[guess] > unimodel_sequence[guess+1]:
            return guess

print(getTopIndex_UnimodelSequence(unimodel_sequence))

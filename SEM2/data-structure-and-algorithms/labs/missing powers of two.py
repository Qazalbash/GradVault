def findMissingNumber(powerTwoList, size):
    low, high = 0, len(powerTwoList) - 1
    while True:
        guess = (low + high) // 2
        if high < low:
            return 2 ** low
        elif powerTwoList[guess] == 2**guess:
            low = guess + 1
        elif powerTwoList[guess] == 2**(guess+1):
            high = guess - 1
        

print(findMissingNumber(powerTwoList, size))

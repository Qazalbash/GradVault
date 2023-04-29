def upper(lst, item): 
    low, high, ans = 0, len(lst) - 1, -1
    while low <= high: 
        guess = (low + high) // 2
        if lst[guess] <= item: 
            low = guess + 1 
        else: 
            ans = guess
            high = guess - 1
    return ans

def lower(lst, item):
    low, high, ans = 0, len(lst) - 1, -1
    while low <= high:  
        guess = (low + high) // 2
        if lst[guess] >= item:
            high = guess - 1
        else:  
            ans = guess
            low = guess + 1
    return ans

def finding_multiple(lst, item):
    low, high = lower(lst, item), upper(lst,item)
    if high < 0:
        high = len(lst) - high - 1
    return list(range(low+1,high))

print(finding_multiple(lst, item))

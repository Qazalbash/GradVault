def pop(lst):
    try:
        lst.pop()
        return lst
    except:
        return False


def top(lst):
    try:
        return lst[-1]
    except:
        return False


def is_empty(lst):
    return len(lst) == 0


def string_reversal(s):
    alphStack = list(s)
    aplh = ""
    while not (is_empty(alphStack)):
        aplh += top(alphStack)
        s = pop(alphStack)
    return aplh


print(string_reversal(s))

def enQueue(lst, data):
    lst.append(data)
    return lst


def push(lst, item):
    lst.append(item)
    return lst


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


def mirror(q):
    rStack = []
    for i in q:
        rStack = push(rStack, i)
    for j in range(len(q)):
        q = enQueue(q, top(rStack))
        rStack = pop(rStack)
    return q


print(mirror(queue))

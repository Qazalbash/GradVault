import ast

lst = input()
lst = ast.literal_eval(lst)


def push(lst, item):
    lst.append(item)


def pop(lst):
    if is_empty(lst):
        return False
    return lst.pop()


def is_empty(lst):
    return len(lst) == 0


def enQueue(lst, item):
    lst.append(item)


def deQueue(lst):
    return lst.pop(0)


def size(lst):
    return len(lst)


def InterLeaveQueue(lst):
    s, siz = [], size(lst)
    for i in range(int(siz / 2)):
        push(s, deQueue(lst))
    while not (is_empty(s)):
        enQueue(lst, pop(s))
    for i in range(int(siz / 2)):
        enQueue(lst, deQueue(lst))
    for i in range(int(siz / 2)):
        push(s, deQueue(lst))
    while not (is_empty(s)):
        enQueue(lst, pop(s))
        enQueue(lst, deQueue(lst))
    return lst


print(InterLeaveQueue(lst))
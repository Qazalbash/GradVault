s=input()
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

def is_empty(lst):
    return len(lst) == 0

def balanced_braces(s):
    stack = []
    ana = {"(":1, "{":2, "[":3, ")":1, "}":2, "]":3}
    for i in s:
        if i in "({[":
            stack = push(stack, ana[i])
        else:
            if top(stack) == ana[i]:
                stack = pop(stack)
            else:
                return False
    return is_empty(stack)          

print(balanced_braces(s))
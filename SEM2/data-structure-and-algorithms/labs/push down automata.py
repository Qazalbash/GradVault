def push(lst, item):
    lst.append(item)


def pop(lst):
    return lst.pop()


def top(lst):
    return lst[-1]


def is_empty(lst):
    return len(lst) == 0


def L1(s, patt):
    if len(s) % 2 != 0:
        return False
    stack = []
    for i in range(len(s) - 1):
        if i % 2 != 0:
            continue
        push(stack, (s[i] + s[i + 1]))
    if len(s) % 4 == 2:
        patt2 = [
            f"{patt[0]}{patt[0]}", f"{patt[1]}{patt[1]}", f"{patt[0]}{patt[1]}"
        ]
        while not (is_empty(stack)):
            if top(stack) in patt2:
                if top(stack) == top(patt2):
                    pop(patt2)
                pop(stack)
            else:
                return False
        if top(patt2) == f"{patt[0]}{patt[1]}":
            return False
    elif len(s) % 4 == 0:
        patt0 = [f"{patt[0]}{patt[0]}", f"{patt[1]}{patt[1]}"]
        while not (is_empty(stack)):
            if top(stack) in patt0:
                pop(stack)
            else:
                return False
    return True


def L2(s):
    if len(s) % 3 != 0:
        return False
    stack = []
    for i in range(len(s) - 2):
        if i % 3 != 0:
            continue
        push(stack, s[i] + s[i + 1] + s[i + 2])
    if len(s) % 9 == 3:
        patt3 = ["111", "000", "110"]
        while not (is_empty(stack)):
            if top(stack) in patt3:
                if top(stack) == top(patt3):
                    pop(patt3)
                pop(stack)
            else:
                return False
        if top(patt3) == "110":
            return False
    elif len(s) % 9 == 6:
        patt6 = ["111", "000", "100"]
        while not (is_empty(stack)):
            if top(stack) in patt6:
                if top(stack) == top(patt6):
                    pop(patt6)
                pop(stack)
            else:
                return False
        if top(patt6) == "100":
            return False
    elif len(s) % 9 == 0:
        patt0 = ["111", "000"]
        while not (is_empty(stack)):
            if top(stack) in patt0:
                if top(stack) == top(patt0):
                    pop(patt0)
                pop(stack)
            else:
                return False
    return True


def L3(s):
    s1, s2 = "", ""
    for i in s:
        if i in "01":
            s1 += i
        elif i in "23":
            s2 += i
    return L1(s1, "01") and L1(s2, "23")


def L4(s):
    try:
        stack = []
        for i in s:
            if i == "0":
                push(stack, i)
            else:
                pop(stack)
        return not (is_empty(stack))
    except:
        return False


def Verify(InputString, G):
    if G == 1:
        return L1(InputString, "10")
    elif G == 2:
        return L2(InputString)
    elif G == 3:
        if ("0" not in InputString) or ("1" not in InputString) or (
                "2" not in InputString) or ("3" not in InputString):
            return False
        return L3(InputString)
        # print(InputString)
    elif G == 4:
        if ("0" not in InputString) or ("1" not in InputString):
            return False
        return L4(InputString)


# below commented code works 100%

# def push(lst, item):
#     lst.append(item)
#     return lst

# def pop(lst):
#     try:
#         lst.pop()
#         return lst
#     except:
#         return False

# def is_empty(lst):
#     return len(lst) == 0

# def Verify(InputString, G):
#     stack = []
#     if str(G) in "12":
#         count = 0
#         for i in InputString:
#             if i == "1":
#                 stack, count = push(stack, i), count + 1
#             else:
#                 break
#         for i in InputString:
#             if i == "0":
#                 if G == 1:
#                     stack, count = pop(stack), count + 1
#                 elif G == 2:
#                     stack, count = pop(pop(stack)), count + 1
#             if stack == False:
#                 return False
#         return count == len(InputString) and is_empty(stack)
#     elif str(G) in "34":
#         count = 0
#         for i in InputString:
#             if i == "0":
#                 stack, count = push(stack, i), count + 1
#             else:
#                 break
#         if count == len(InputString):
#             return False
#         for i in InputString:
#             if i == "1":
#                 stack, count = pop(stack), count + 1
#             if stack == False:
#                 return False
#         if G == 4:
#             return count == len(InputString) and not(is_empty(stack))
#         if count == 0:
#             return False
#         count2 = count
#         for i in InputString:
#             if i == "2":
#                 stack, count = push(stack, i), count + 1
#         for i in InputString:
#             if i == "3":
#                 stack, count = pop(stack), count + 1
#             if stack == False:
#                 return False
#         return count == len(InputString) and is_empty(stack) and count2 != count

print(Verify(InputString, G))

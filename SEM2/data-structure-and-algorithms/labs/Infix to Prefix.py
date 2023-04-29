expression = input()

def push(lst, item):
    lst.append(item)

def pop(lst):
    return lst.pop()
    

def top(lst):
    return lst[-1]

def is_empty(lst):
    return len(lst) == 0
    
def checkPrecedence(operator):
    operatorPrecedence = {"(":0,"+":1, "-":1, "*":2, "/":2}
    return operatorPrecedence[operator]

def Infix_to_Prefix(exp):
    expression = []
    stack = []
    result = ""
    for i in exp.split(" ")[::-1]:
        if i == "(":
            expression += [")"]
        elif i == ")":
            expression += ["("]
        else:
            expression += [i]
    for i in expression:
        if i.isalpha():
            result += i + " "
        elif i == "(":
            push(stack, i)
        elif i == ")":
            while not(is_empty(stack)) and top(stack) != "(":
                result += pop(stack) + " "
            pop(stack)
        else:
            while not(is_empty(stack)) and checkPrecedence(i) <= checkPrecedence(top(stack)):
                result += pop(stack) + " "
            push(stack, i)
    while not is_empty(stack):
        result += pop(stack) + " "
    return result[::-1][1:]
print(Infix_to_Prefix(expression))
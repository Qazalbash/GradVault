expression = input()

def push(lst, item):
    lst.append(item)

def pop(lst):
    return lst.pop()

def top(lst):
    return lst[-1]

def is_empty(lst):
    return len(lst) == 0

def EvalutePrefix(expression):
    expression, operandStack = expression.split(" ")[::-1], []
    for i in expression:
        try:
            if type(eval(i)) == int:
                push(operandStack, i)
        except:
            op1 = top(operandStack)
            pop(operandStack)
            op2 = top(operandStack)
            pop(operandStack)
            push(operandStack, str(eval(op1+i+op2)))
    return int(eval(top(operandStack)))

print(EvalutePrefix(expression))
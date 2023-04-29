rawHTML = input()


def push(lst, item):
    lst.append(item)


def pop(lst):
    return lst.pop()


def top(lst):
    try:
        return lst[-1]
    except:
        return False


def is_empty(lst):
    return len(lst) == 0


def isHtmlMatched(rawHTML):
    tag = {
        "<h1>": 0,
        "<h2>": 1,
        "<body>": 2,
        "<center>": 3,
        "<p>": 4,
        "<ol>": 5,
        "<li>": 6,
        "</h1>": 0,
        "</h2>": 1,
        "</body>": 2,
        "</center>": 3,
        "</p>": 4,
        "</ol>": 5,
        "</li>": 6
    }
    stack = []
    for i in rawHTML.split(" "):
        if "<" not in i and ">" not in i:
            continue
        if i in "<h1><h2><body><center><p><ol><li>":
            try:
                push(stack, tag[i])
            except:
                return False
        else:
            if i in "</h1></h2></body></center></p></ol></li>":
                if top(stack) == tag[i]:
                    pop(stack)
                else:
                    return False
    return is_empty(stack)


print(isHtmlMatched(rawHTML))

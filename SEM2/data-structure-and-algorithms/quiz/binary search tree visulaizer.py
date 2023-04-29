def insert(bst, key):
    if not (bst):
        return {"value": key, "left": {}, "right": {}}
    elif bst["value"] == key:
        return bst
    elif bst["value"] < key:
        bst["right"] = insert(bst["right"], key)
    else:
        bst["left"] = insert(bst["left"], key)
    return bst


def exist(bst, key, path=""):
    if not (bst):
        return None
    elif bst["value"] == key:
        return path
    elif bst["value"] > key:
        path += "L"
        return exist(bst["left"], key, path)
    path += "R"
    return exist(bst["right"], key, path)


def createBST(lst):
    bst = {}
    for i in lst:
        bst = insert(bst, i)
    return bst


def TracingKey_to_LRpath(bst, query):
    return exist(bst, query, path="")


lst = eval(input())
bst = createBST(lst)
print(bst)

N = int(input())

for i in range(N):
    query = int(input())
    print(TracingKey_to_LRpath(bst, query))

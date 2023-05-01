def insert(bst, key=0):

    if bst == {}:
        return {"value": key, "left": {}, "right": {}}
    elif bst["value"] == key:
        return bst
    elif bst["value"] < key:
        bst["right"] = insert(bst["right"], key)
    else:
        bst["left"] = insert(bst["left"], key)
    return bst


def exist(bst, key):

    if bst == {}:
        return False
    elif bst["value"] == key:
        return True
    elif bst["value"] > key:
        return exist(bst["left"], key)
    return exist(bst["right"], key)


def search(bst, key):

    if bst["value"] == key or bst is {}:
        return bst
    if bst["value"] > key:
        return search(bst["left"], key)
    return search(bst["right"], key)


def minimum(bst, starting_node):

    temp = search(bst, starting_node)
    while temp["left"]:
        temp = temp["left"]
    return temp["value"]


def maximum(bst, starting_node):

    temp = search(bst, starting_node)
    while temp["right"]:
        temp = temp["right"]
    return temp["value"]


def inorder_traversal(bst):

    if bst:
        inorder_traversal(bst["left"])
        print(bst["value"], end=" ")
        inorder_traversal(bst["right"])


def preorder_traversal(bst):

    if bst:
        print(bst["value"], end=" ")
        preorder_traversal(bst["left"])
        preorder_traversal(bst["right"])


def postorder_traversal(bst):

    if bst:
        postorder_traversal(bst["left"])
        postorder_traversal(bst["right"])
        print(bst["value"], end=" ")


print(
    "******************************************** Question 1 ********************************************"
)

bst = {}

for key in [68, 88, 61, 89, 94, 50, 4, 76, 66, 82]:
    bst = insert(bst, key)

print(bst, end="\n\n")

print(f"50 exist in binary search tree: {exist(bst, 50)}", end="\n\n")

print(f"49 exist in binary search tree: {exist(bst, 49)}", end="\n\n")

print(minimum(bst, 68), end="\n\n")

print(minimum(bst, 88), end="\n\n")

print(maximum(bst, 68), end="\n\n")

print(maximum(bst, 61), end="\n\n")

print("Inorder Tarnsversal", end=" ")
inorder_traversal(bst)

print("\n\nPreorder Tarnsversal", end=" ")
preorder_traversal(bst)

print("\n\nPostorder Tarnsversal", end=" ")
postorder_traversal(bst)

print("\n")

print(
    "******************************************** Question 2 ********************************************"
)

BST = {}

for key in ["begin", "do", "else", "end", "if", "then", "while"]:
    BST = insert(BST, key)

print(BST)

d = {}
for i in [13, 3, 4, 12, 14, 10, 5, 1, 8, 2, 7, 9, 11, 6]:
    d = insert(d, i)

print('\n\n\n\n\n\n\n\n')

print("Inorder Tarnsversal", end=" ")
inorder_traversal(d)

print("\n\nPreorder Tarnsversal", end=" ")
preorder_traversal(d)

print("\n\nPostorder Tarnsversal", end=" ")
postorder_traversal(d)

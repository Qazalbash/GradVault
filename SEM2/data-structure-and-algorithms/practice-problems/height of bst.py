def insert(bst, key=0):

    if bst == {}:
        return {"value": key, "left": {}, "right": {}}
    elif bst["value"] == key:
        return bst
    elif bst["value"] < key:
        bst["right"] = insert(bst["right"], key)
    elif bst["value"] > key:
        bst["left"] = insert(bst["left"], key)
    return bst


def main(bst):
    if not (bst):
        return 0
    return 1 + max(main(bst["left"]), main(bst["right"]))


N = int(input())
A = [int(i) for i in input().split(" ")]

bst = {}
for key in A:
    bst = insert(bst, key)
print(main(bst))

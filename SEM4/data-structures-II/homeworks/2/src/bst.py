class Node:

    def __init__(self, data) -> None:
        self.data = data
        self.left = self.right = None


class BST:

    def __init__(self):
        self.root = None

    def _insert(self, root: Node, n: int) -> None:
        if n <= root.data:
            if root.left:
                self._insert(root.left, n)
            else:
                root.left = Node(n)
        else:
            if root.right:
                self._insert(root.right, n)
            else:
                root.right = Node(n)

    def insert(self, n: int):
        if not self.root:
            self.root = Node(n)
        else:
            self._insert(self.root, n)

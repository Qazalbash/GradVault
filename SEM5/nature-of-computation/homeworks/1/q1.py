from itertools import permutations

from typing import Optional


class Node(object):

    def __init__(self) -> None:
        self.children: dict = {}

    def __repr__(self) -> str:
        return f"{self.children}"


class Trie:

    ENDWORD: str = "^"

    def __init__(self) -> None:
        self._root: Node = Node()

    def add(self, word) -> None:
        node = self._root
        for letter in word:
            if letter not in node.children:
                node.children[letter] = Node()
            node = node.children[letter]
        node.children[Trie.ENDWORD] = Node()

    def __repr__(self) -> str:
        return f"{self._root}"

    def traverse(self, node: Node, prefix: str) -> None:
        if node is None or Trie.ENDWORD in node.children:
            print(prefix)

        for char in node.children:
            self.traverse(node.children[char], prefix + char)


t = Trie()
lst = [0, 0]
for i in range(6):
    lst_ = lst + [1] * i
    for k in set(permutations(lst_)):
        t.add("".join(map(str, k)))
from pprint import pprint

pprint(t)
# t.traverse(t._root, "")

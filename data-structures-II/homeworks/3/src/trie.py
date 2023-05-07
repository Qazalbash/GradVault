from doc import *

ENDWORD: str = "^"


class Node(object):

    def __init__(self) -> None:
        self.children: dict = {}


class Trie:

    def __init__(self) -> None:

        self._root: Node = Node()

    @property
    def tree(self):
        return self._root

    @tree.setter
    def tree(self, doc: Document) -> None:

        for word, locs in doc.word_loc_map.items():
            node = self._root
            # adding words
            for letter in word:

                if letter not in node.children:
                    node.children[letter] = Node()

                node = node.children[letter]

            if ENDWORD not in node.children:
                node.children[ENDWORD] = locs

            else:
                node.children[ENDWORD].extend(locs)

    def complete(self, word: str) -> [(str, [(str, int, int)])]:

        tokenized_word = word.split()  # prefix tokenization

        matches = {}

        for w in tokenized_word:
            s = self.match(self._root, w, w)

            for i, locs in s:
                matches[i] = matches.get(i, []) + locs

        return matches

    @staticmethod
    def match(node: Node, prefix: str, trace: str) -> [(str, [str])]:

        prefix_matching = []

        if prefix == trace:

            for prefix_char in trace:

                if prefix_char not in node.children:
                    return prefix_matching

                node = node.children[prefix_char]

        for postfix_char in node.children:

            if postfix_char == ENDWORD:
                prefix_matching.extend([(trace, node.children[postfix_char])])

            else:
                prefix_matching.extend(
                    Trie.match(node.children[postfix_char], prefix,
                               trace + postfix_char))

        return prefix_matching

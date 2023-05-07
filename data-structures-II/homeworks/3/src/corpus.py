from doc import *
from invertedindex import *
from trie import *


class Corpus:

    def __init__(self,
                 path: str,
                 index: bool = True,
                 trie: bool = True) -> None:

        self._index: InvertedIndex = InvertedIndex() if index else None

        self._trie: Trie = Trie() if trie else None

        _path = Path(path)

        for doc in _path.rglob("*"):

            try:
                doc = Document(doc)
            except:
                continue

            if index:
                self._index.indexs = doc

            if trie:
                self._trie.tree = doc

    def query(self, query: str, k: int) -> [(float, str)]:

        if self._index:
            return self._index.query(query, k)

    def prefix_complete(self, query: str) -> [(str, (str, int, int))]:

        if self._trie:
            query = query.lower()
            return self._trie.complete(query)

from math import log10

from doc import *


class InvertedIndex:

    def __init__(self) -> None:

        self._index_dict: dict = {}
        self._corpus_size: int = 0

    @property
    def indexs(self):
        return self._index_dict

    @indexs.setter
    def indexs(self, doc: Document) -> None:

        word_loc_map = doc.word_loc_map.items()

        for word, loc in word_loc_map:

            self._index_dict[word] = self._index_dict.get(word, {})
            self._index_dict[word][doc.doc_id] = len(loc) / doc.size

        self._corpus_size += 1

    def query(self, query_string: str, k: int) -> [(float, str)]:

        query_words = query_string.split()  # tokenizing query
        rank = {}

        for word in query_words:
            try:
                doc_freq = len(self._index_dict[word])
            except KeyError:
                continue

            for doc_id, tf in self._index_dict[word].items():
                rank[doc_id] = rank.get(
                    doc_id, 0) + tf * log10(self._corpus_size / doc_freq)

        k = (k - len(rank)) * (k < len(rank)) + len(rank)
        rank = [(tf_idfs, id) for id, tf_idfs in rank.items()]
        rank = sorted(rank, key=lambda k: k[0], reverse=True)
        return rank[:k]

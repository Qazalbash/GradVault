from random import randrange

from const import *
from myset import MySet


class ChainedSet(MySet):
    """Overrides and implementes the methods defined in MySet. Uses a chained
    hash set to implement the set.
    """

    def __init__(self, elements: [object] = [], p: int = MODULO) -> None:

        self._set = [[None]] * CAP
        self._n = 0
        self._prime = p
        self._scale = 1 + randrange(p - 1)
        self._shift = randrange(p)

        for ele in elements:
            self.add(ele)

    def __hash__(self, element: object) -> int:

        return ((hash(element) * self._scale + self._shift) % self._prime %
                len(self._set))

    def __getitem__(self, element: object) -> object:

        bucket = self._set[self.__hash__(element)]

        if bucket == [None] or element not in bucket:
            raise KeyError("Key Error: " + repr(element))

        return element

    def add(self, element: int) -> None:

        hash_element = self.__hash__(element)

        if self._set[hash_element] == [None]:
            self._set[hash_element] = []

        if element not in self._set[hash_element]:
            self._set[hash_element].append(element)
            self._n += 1

        if self._n > len(self._set) // 2:
            self._resize(2 * len(self._set) - 1)

    def discard(self, element: int) -> None:

        hash_element = self.__hash__(element)

        if element in self._set[hash_element]:
            self._set[hash_element].remove(element)

        self._n -= 1

    def _resize(self, c: int) -> None:

        old = self._set
        self._set = c * [[None]]
        self._n = 0

        for bucket in old:

            if bucket != [None]:

                for ele in bucket:
                    self.add(ele)

    def __iter__(self):

        for bucket in self._set:

            if bucket != [None]:

                for ele in bucket:

                    if ele != None:
                        yield ele

    def __len__(self) -> int:

        return self._n

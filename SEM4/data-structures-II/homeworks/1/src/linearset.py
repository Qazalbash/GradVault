from random import randrange

from const import *
from myset import MySet


class LinearSet(MySet):
    """Overrides and implementes the methods defined in MySet. Uses a linear
    probing hash set to implement the set.
    """

    _AVAIL = object()

    def __init__(self, elements: [object] = [], p: int = MODULO) -> None:

        self._set = [None] * CAP
        self._n = 0
        self._prime = p
        self._scale = 1 + randrange(p - 1)
        self._shift = randrange(p)

        for ele in elements:
            self.add(ele)

    def __hash__(self, element: object) -> int:

        return (
            (hash(element) * self._scale + self._shift) % self._prime % len(self._set)
        )

    def __getitem__(self, key: object, default: object = None) -> object:

        found, s = self._find_slot(self.__hash__(key), key)

        if not found:
            return default

        return self._set[s]

    def add(self, element: int) -> None:

        found, s = self._find_slot(self.__hash__(element), element)
        if not found:
            self._n += 1
            self._set[s] = element

        if self._n > len(self._set) // 2:
            self._resize(2 * len(self._set) - 1)

    def discard(self, element: int) -> None:

        hash_element = self.__hash__(element)
        found, s = self._find_slot(hash_element, element)

        if found:
            self._set[s] = LinearSet._AVAIL
            self._n -= 1

    def _resize(self, c: int) -> None:

        old = self._set
        self._set = c * [None]
        self._n = 0

        for ele in old:

            if ele != None and ele != LinearSet._AVAIL:
                self.add(ele)

    def __len__(self) -> int:

        return self._n

    def _is_available(self, hash_element: int) -> bool:

        return (
            self._set[hash_element] == None
            or self._set[hash_element] == LinearSet._AVAIL
        )

    def _find_slot(self, hash_element: int, element: int) -> (bool, object):

        firstAvail = None

        while True:

            if self._is_available(hash_element):

                if firstAvail == None:
                    firstAvail = hash_element

                if self._set[hash_element] == None:
                    return False, firstAvail

            elif element == self._set[hash_element]:
                return True, hash_element

            hash_element = (hash_element + 1) % len(self._set)

    def __iter__(self):

        for ele in self._set:

            if ele != None and ele != LinearSet._AVAIL:
                yield ele

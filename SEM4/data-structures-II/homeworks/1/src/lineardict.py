from random import randrange

from const import *
from mydict import MyDict


class LinearDict(MyDict):
    """Overrides and implementes the methods defined in MyDict. Uses a linear
    probing hash table to implement the dictionary.
    """

    _AVAIL = object()

    def __init__(self, p: int = MODULO) -> None:

        self._table = [None] * CAP
        self._n = 0
        self._prime = MODULO
        self._scale = 1 + randrange(p - 1)
        self._shift = randrange(p)

    def __hash__(self, key: object) -> int:

        return (hash(key) * self._scale + self._shift) % self._prime % len(
            self._table)

    def __getitem__(self, key: object, default: object = None) -> object:

        found, s = self._find_slot(self.__hash__(key), key)

        if not found:
            return default
        return self._table[s][1]

    def __setitem__(self, key: int, newvalue: object) -> None:

        found, s = self._find_slot(self.__hash__(key), key)
        self._table[s] = (key, newvalue)
        self._n += not found

        if self._n > len(self._table) // 2:
            self._resize(2 * len(self._table) - 1)

    def __delitem__(self, key: int) -> None:

        found, s = self._find_slot(self.__hash__(key), key)

        if not found:
            raise KeyError("Key Error: " + repr(key))

        self._table[s] = LinearDict._AVAIL
        self._n -= 1

    def _resize(self, c: int) -> None:

        old = self.items()
        self._table = c * [None]
        self._n = 0
        for key, value in old:
            self[key] = value

    def __len__(self) -> int:
        return self._n

    def _is_available(self, hash_key: int) -> bool:

        return (self._table[hash_key] == None or
                self._table[hash_key] == LinearDict._AVAIL)

    def _find_slot(self, hash_key: int, key: int) -> (bool, object):

        firstAvail = None

        while True:

            if self._is_available(hash_key):

                if firstAvail == None:
                    firstAvail = hash_key

                if self._table[hash_key] == None:
                    return False, firstAvail

            elif key == self._table[hash_key][0]:
                return True, hash_key

            hash_key = (hash_key + 1) % len(self._table)

    def __iter__(self):

        for hash_key in range(len(self._table)):

            if not self._is_available(hash_key):
                yield self._table[hash_key][0]

    def items(self) -> [(object, object)]:

        return [(kv[0], kv[1]) for kv in self._table if kv != None]

    def clear(self) -> None:

        self._table = [None] * CAP

    def get(self, key: int, default: object = None) -> object:

        return self.__getitem__(key, default)

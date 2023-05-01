from random import randrange

from const import *
from mydict import MyDict


class ChainedDict(MyDict):
    """Overrides and implementes the methods defined in MyDict. Uses a chained
    hash table to implement the dictionary.
    """

    def __init__(self, p: int = MODULO) -> None:

        self._table = [[None]] * CAP
        self._n = 0
        self._prime = p
        self._scale = 1 + randrange(p - 1)
        self._shift = randrange(p)

    def __hash__(self, key: object) -> int:

        return (hash(key) * self._scale + self._shift) % self._prime % len(
            self._table)

    def __getitem__(self, key: object, default: object = None) -> object:

        hash_key = self.__hash__(key)

        if self._table[hash_key] != [None]:

            for item in self._table[hash_key]:

                if key == item[0]:

                    if item[1] == "*" and default == None:
                        raise KeyError("Key Error: " + repr(key))

                    elif item[1] == "*" and default != None:
                        return default

                    return item[1]

        return default

    def __setitem__(self, key: int, newvalue: object) -> None:

        hash_key = self.__hash__(key)

        if self._table[hash_key] == [None]:
            self._table[hash_key] = [(key, newvalue)]

        else:
            for ele in range(len(self._table[hash_key])):

                if self._table[hash_key][ele][0] == key:
                    self._table[hash_key][ele] = (key, newvalue)
                    return

            self._table[hash_key].append((key, newvalue))
            self._n += 1

        if self._n > len(self._table) // 2:
            self._resize(2 * len(self._table) - 1)

    def __delitem__(self, key: int) -> None:

        hash_key = self.__hash__(key)
        bucket = self._table[hash_key]

        if bucket == [None]:
            raise KeyError("Key Error: " + repr(key))

        for ele_index in range(len(bucket)):

            if bucket[ele_index][0] == key:
                self._table[hash_key][ele_index] = (key, "*")

        self._n -= 1

    def _resize(self, c: int) -> None:

        old = self.items()
        self._table = c * [[None]]
        self._n = 0

        for key, value in old:
            self[key] = value

    def __iter__(self):

        for bucket in self._table:

            if bucket != [None]:

                for ele in bucket:
                    yield ele[0]

    def __len__(self) -> int:
        return self._n

    def items(self) -> [(object, object)]:

        item_list = []

        for bucket in self._table:

            if bucket != [None]:

                for ele in bucket:
                    item_list.append(ele)

        return item_list

    def clear(self) -> None:

        self._table = [[None]] * CAP
        self._n = 0

    def get(self, key: int, default: object) -> object:

        return self.__getitem__(key, default)

    def set(self, key, value) -> None:

        self[key] = value

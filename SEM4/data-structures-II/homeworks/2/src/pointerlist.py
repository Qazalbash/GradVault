from bst import Node
from mylist import *


class PointerList(MyList):

    def __init__(self, size: int, value=None) -> None:
        self.head = prev = None
        self.size = size

        for _ in range(size):
            node = Node(value)
            if prev == None:
                prev = node
                self.head = node
            else:
                prev.left = node
            prev = node

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int):
        assert (
            0 <= i < len(self)
        ), f"Getting invalid list index {i} from list of size {len(self)}"
        current = self.head
        for _ in range(i):
            current = current.left
        return current.data

    def __setitem__(self, i: int, value) -> None:
        assert (
            0 <= i < len(self)
        ), f"Getting invalid list index {i} from list of size {len(self)}"
        current = self.head
        for _ in range(i):
            current = current.left
        current.data = value

    def __iter__(self) -> "MyList":
        return super().__iter__()

    def __next__(self):
        return super().__next__()

    def get(self, index: int):
        return self.__getitem__(index)

    def set(self, index: int, value: (int, int, int)) -> None:
        self.__setitem__(index, value)

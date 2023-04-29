import array as arr
from mylist import *


class ArrayList(MyList):

    def __init__(self, size: int, value=None) -> None:
        self.rgb = arr.array("i", MyList(3 * size, value[0]))
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> (int, int, int):

        assert (
            0 <= i < len(self)
        ), f"Getting invalid list index {i} from list of size {len(self)}"
        r, g, b = self.rgb[3 * i:3 * i + 3]
        return (r, g, b)

    def __setitem__(self, i: int, value) -> None:
        assert (
            0 <= i < len(self)
        ), f"Setting invalid list index {i} in list of size {len(self)}"
        self.rgb[3 * i] = value[0]
        self.rgb[3 * i + 1] = value[1]
        self.rgb[3 * i + 2] = value[2]

    def get(self, index: int):
        return self.__getitem__(index)

    def set(self, index: int, value) -> None:
        self.__setitem__(index, value)

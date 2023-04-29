class MyList:
    """A list interface. Also implements Iterator functions in order to support
    iteration over this list.
    """

    def __init__(self, size: int, value=None) -> None:
        """Creates a list of the given size, optionally intializing elements to value.
        The list is static. It only has space for size elements.
        Parameters:
        - self: mandatory reference to this object
        - size: size of the list; space is reserved for these many elements.
        - value: the optional initial value of the created elements.
        Returns:
        none
        """

        self.lst = [value for _ in range(size)]

    def __len__(self) -> int:
        """Returns the size of the list. Allows len() to be called on it.
        Ref: https://stackoverflow.com/q/7642434/1382487
        Parameters:
        - self: mandatory reference to this object
        Returns:
        the size of the list.
        """
        return len(self.lst)

    def __getitem__(self, i: int):
        """Returns the value at index, i. Allows indexing syntax.
        Ref: https://stackoverflow.com/a/33882066/1382487
        Parameters:
        - self: mandatory reference to this object
        - i: the index from which to retrieve the value.
        Returns:
        the value at index i.
        """
        # Ensure bounds.
        assert (
            0 <= i < len(self)
        ), f"Getting invalid list index {i} from list of size {len(self)}"
        return self.lst[i]

    def __setitem__(self, i: int, value) -> None:
        """Sets the element at index, i, to value. Allows indexing syntax.
        Ref: https://stackoverflow.com/a/33882066/1382487
        Parameters:
        - self: mandatory reference to this object
        - i: the index of the elemnent to be set
        - value: the value to be set
        Returns:
        none
        """
        # Ensure bounds.
        assert (
            0 <= i < len(self)
        ), f"Setting invalid list index {i} in list of size {len(self)}"
        self.lst[i] = value

    def __iter__(self) -> "MyList":
        """Iterator function to return an iterator (self) that allows iteration over
        this list.
        Parameters:
        - self: mandatory reference to this object
        Returns:
        an iterator (self) that allows iteration over this list.
        """
        # Initialize iteration index.
        self._iter_index: int = 0
        return self

    def __next__(self):
        """'Iterator function to return the next value from this list.
        Parameters:
        - self: mandatory reference to this object
        Returns:
        the next value in this list since the last iteration.
        """
        if self._iter_index < len(self):
            value = self.get(self._iter_index)
            self._iter_index += 1
            return value
        else:
            self._index = 0
            raise StopIteration

    def get(self, i: int):
        """Returns the value at index, i.
        Alternate to use of indexing syntax.
        Parameters:
        - self: mandatory reference to this object
        - i: the index from which to retrieve the value.
        Returns:
        the value at index i.
        """
        return self.__getitem__(i)

    def set(self, i: int, value) -> None:
        """Sets the element at index, i, to value.
        Alternate to use of indexing syntax.
        Parameters:
        - self: mandatory reference to this object
        - i: the index of the elemnent to be set
        - value: the value to be set
        Returns:
        none
        """
        self.__setitem__(i, value)

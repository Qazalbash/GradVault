class MyDict(object):
    """An abstract class that provides a dictionary interface which is just
    sufficient for the implementation of this assignment.
    """

    def __init__(self) -> None:
        """Initializes this dictionary.

        Args:
        - self: manadatory reference to this object.

        Returns:
        none
        """
        pass

    def __setitem__(self, key: object, newvalue: object) -> None:
        """Adds (key, newvalue) to the dictionary, overwriting object prior value.

        dunder method allows assignment using indexing syntax, e.g.
        d[key] = newvalue

        key must be hashable by pytohn.

        Args:
        - self: manadatory reference to this object.
        - key: the key to add to the dictionary
        - newvalue: the value to store for the key, overwriting object prior value

        Returns:
        None
        """
        pass

    def __getitem__(self, key: object, default: object = None) -> object:
        """Returns the value stored for key, default if no value exists.

        key must be hashable by pytohn.

        Args:
        - self: manadatory reference to this object.
        - key: the key whose value is sought.
        - default: the value to return if key does not exist in this dictionary

        Returns:
        the stored value for key, default if no such value exists.
        """
        pass

    def items(self) -> [(object, object)]:
        """Returns the key-value pairs of the dictionary as tuples in a list.

        Args:
        - self: manadatory reference to this object.

        Returns:
        the key-value pairs of the dictionary as tuples in a list.
        """
        pass

    def clear(self) -> None:
        """Clears the dictionary.

        Args:
        - self: manadatory reference to this object.

        Returns:
        None.
        """
        pass

    def __iter__(self):
        """Generate iteration of the MyDict"""
        pass

    def __delitem__(self, key) -> None:
        pass

    def __len__(self) -> int:
        pass

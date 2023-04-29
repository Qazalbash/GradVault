from PIL import Image
from arraylist import *
from pointerlist import *


class MyImage:
    """Holds a flattened RGB image and its dimensions. Also implements Iterator
    methods to allow iteration over this image.
    """

    def __init__(self, size: (int, int), pointer=False) -> None:
        """Initializes a black image of the given size.

        Parameters:
        - self: mandatory reference to this object
        - size: (width, height) specifies the dimensions to create.
        - pointer: if True then the backing list is pointer-based else array-based.

        Returns:
        none
        """
        # Save size, create a list of the desired size with black pixels.
        width, height = self.size = size
        if pointer:
            self.pixels: PointerList = PointerList(width * height,
                                                   value=(0, 0, 0))
        else:
            self.pixels: ArrayList = ArrayList(width * height, value=(0, 0, 0))

    def __iter__(self) -> "MyList":
        """Iterator function to return an iterator (self) that allows iteration over
        this image.

        Parameters:
        - self: mandatory reference to this object

        Returns:
        an iterator (self) that allows iteration over this image.
        """
        # Initialize iteration indexes.
        self._iter_r: int = 0
        self._iter_c: int = 0
        return self

    def __next__(self):
        """'Iterator function to return the next value from this list.

        Image pixels are iterated over in a left-to-right, top-to-bottom order.

        Parameters:
        - self: mandatory reference to this object

        Returns:
        the next value in this image since the last iteration.
        """
        if self._iter_r < self.size[1]:  # Iteration within image bounds.
            # Save current value as per iteration variables. Update the
            # variables for the next iteration as per the iteration
            # order. Return saved value.
            value = self.get(self._iter_r, self._iter_c)
            self._iter_c += 1
            if self._iter_c == self.size[0]:
                self._iter_c = 0
                self._iter_r += 1
            return value
        else:  # Image bounds exceeded, end of iteration.
            # Reset iteration variables, end iteration.
            self._iter_r = self._iter_c = 0
            raise StopIteration

    def _get_index(self, r: int, c: int) -> int:
        """Returns the list index for the given row, column coordinates.

        This is an internal function for use in class methods only. It should
        not be used or called from outside the class.

        Parameters:
        - self: mandatory reference to this object
        - r: the row coordinate
        - c: the column coordinate

        Returns:
        the list index corresponding to the given row and column coordinates
        """
        # Confirm bounds, compute and return list index.
        width, height = self.size
        assert 0 <= r < height and 0 <= c < width, (
            "Bad image coordinates: "
            f"(r, c): ({r}, {c}) for image of size: {self.size}")
        return r * width + c

    def open(path: str, pointer=False) -> "MyImage":
        """Creates and returns an image containing from the information at file path.

        The image format is inferred from the file name. The read image is
        converted to RGB as our type only stores RGB.

        Parameters:
        - path: path to the file containing image information
        - pointer: if True then the backing list is pointer-based, else array-based.

        Returns:
        the image created using the information from file path.
        """
        # Use PIL to read the image information and store it in our instance.
        img: Image = Image.open(path)
        myimg: MyImage = MyImage(img.size, pointer)
        # width, height = img.size # redundant b/c they are not used
        # Covert image to RGB. https://stackoverflow.com/a/11064935/1382487
        img: Image = img.convert("RGB")
        # Get list of pixel values (https://stackoverflow.com/a/1109747/1382487),
        # copy to our instance and return it.
        for i, rgb in enumerate(list(img.getdata())):
            myimg.pixels.set(i, rgb)
        return myimg

    def save(self, path: str) -> None:
        """Saves the image to the given file path.

        The image format is inferred from the file name.

        Parameters:
        - self: mandatory reference to this object
        - path: the image has to be saved here.

        Returns:
        none
        """
        # Use PIL to write the image.
        img: Image = Image.new("RGB", self.size)
        img.putdata([rgb for rgb in self.pixels])
        img.save(path)

    def get(self, r: int, c: int) -> (int, int, int):
        """Returns the value of the pixel at the given row and column coordinates.

        Parameters:
        - self: mandatory reference to this object
        - r: the row coordinate
        - c: the column coordinate

        Returns:
        the stored RGB value of the pixel at the given row and column coordinates.
        """
        return self.pixels[self._get_index(r, c)]

    def set(self, r: int, c: int, rgb: (int, int, int)) -> None:
        """Write the rgb value at the pixel at the given row and column coordinates.

        Parameters:
        - self: mandatory reference to this object
        - r: the row coordinate
        - c: the column coordinate
        - rgb: the rgb value to write

        Returns:
        none
        """
        self.pixels[self._get_index(r, c)] = rgb

    def show(self) -> None:
        """Display the image in a GUI window.

        Parameters:

        Returns:
        none
        """
        # Use PIL to display the image.
        img: Image = Image.new("RGB", self.size)
        img.putdata([rgb for rgb in self.pixels])
        img.show()

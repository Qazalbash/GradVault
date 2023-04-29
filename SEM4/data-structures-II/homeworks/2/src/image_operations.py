from myimage import MyImage


def remove_channel(src: MyImage,
                   red: bool = False,
                   green: bool = False,
                   blue: bool = False) -> MyImage:
    """Returns a copy of src in which the indicated channels are suppressed.
    Suppresses the red channel if no channel is indicated. src is not modified.
    
    Args:
    - src: the image whose copy the indicated channels have to be suppressed.
    - red: suppress the red channel if this is True.
    - green: suppress the green channel if this is True.
    - blue: suppress the blue channel if this is True.
    
    Returns: a copy of src with the indicated channels suppressed.
    """
    col, row = src.size
    returnSrc = MyImage((row, col), False)
    for i in range(row):
        for j in range(col):
            rgb = src.get(i, j)
            returnSrc.set(i, j, (
                rgb[0] * (not red),
                rgb[1] * (not green),
                rgb[2] * (not blue),
            ))

    return returnSrc


def rotations(src: MyImage) -> MyImage:
    """Returns an image containing the 4 rotations of src.
    The new image has twice the dimensions of src. src is not modified.
    
    Args:
    - src: the image whose rotations have to be stored and returned.
    
    Returns:
    an image twice the size of src and containing the 4 rotations of src.
    """
    col, row = src.size

    returnSrc = MyImage((row + col, row + col))

    for srcRow in range(row):
        for srcCol in range(col):
            pixel = src.get(srcRow, srcCol)
            # first rotation
            returnSrc.set(col - srcCol - 1, srcRow, pixel)
            # second rotation
            returnSrc.set(row + col - srcRow - 1, col - srcCol - 1, pixel)
            # third rotatioin
            returnSrc.set(row + srcCol, row + col - srcRow - 1, pixel)
            # # fourth rotation
            returnSrc.set(srcRow, row + srcCol, pixel)

    return returnSrc


def apply_mask(src: MyImage, maskfile: str, average: bool = True) -> MyImage:
    """Returns an copy of src with the mask from maskfile applied to it.

    maskfile specifies a text file which contains an n by n mask. It has the
    following format:
    - the first line contains n
    - the next n^2 lines contain 1 element each of the flattened mask

    Args:
    - src: the image on which the mask is to be applied
    - maskfile: path to a file specifying the mask to be applied
    - average: if True, averaging should to done when applying the mask

    Returns:
    an image which the result of applying the specified mask to src.
    """
    col, row = src.size

    f = open(maskfile)
    kernel = f.read().splitlines()
    f.close()

    kernelSize = int(kernel[0])
    kernel = [int(i) for i in kernel[1:]]

    paddingKernel = kernelSize // 2

    returnSrc = MyImage((col, row))

    mean_pixel = lambda rgb: (rgb[0] + rgb[1] + rgb[2]) // 3

    for srcRow in range(row):
        for srcCol in range(col):

            sum = 0
            sumOfKernel = 0
            for kernelRow in range(-paddingKernel, paddingKernel + 1):
                for kernelCol in range(-paddingKernel, paddingKernel + 1):

                    if 0 <= srcRow + kernelRow < row and 0 <= srcCol + kernelCol < col:
                        pixel = mean_pixel(
                            src.get(srcRow + kernelRow, srcCol + kernelCol))
                        kPixel = kernel[(paddingKernel + kernelRow) *
                                        kernelSize + paddingKernel + kernelCol]

                        sum += pixel * kPixel
                        sumOfKernel += kPixel
            if average and sumOfKernel != 0:
                sum //= sumOfKernel
            if sum > 255:
                sum = 255
            elif sum < 0:
                sum = 0
            returnSrc.set(srcRow, srcCol, (sum, sum, sum))

    return returnSrc

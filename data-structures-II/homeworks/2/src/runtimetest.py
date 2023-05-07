from time import time
import json
from image_operations import *


def test():
    """This function was meant to calculate time each file took in ArrayList and PointerList"""
    masks = {
        "mblur": r"masks\\mask-blur-more.txt",
        "sblur": r"masks\\mask-blur-slightly.txt",
        "blur": r"masks\\mask-blur.txt",
        "sobelx": r"masks\\mask-sobel-x.txt",
        "sobely": r"masks\\mask-sobel-y.txt"
    }

    files = {
        "csLogo": r"images\\cs-logo.png",
        "campus": r"images\\campus.jpeg",
        "hu-logo": r"images\\hu-logo.png",
        "plants": r"images\\plants.jpg"
    }

    # T = [0:ArrayList, 1:PointerList]

    for p in range(2):
        T = {
            "remove_channel": {fileName: 0 for fileName in files.keys()},
            "rotations": {fileName: 0 for fileName in files.keys()},
            "apply_mask": {maskName: {} for maskName in masks.keys()}
        }
        for file, file_path in files.items():
            img = MyImage.open(file_path, p)
            print(file)

            # remove channel
            print("remove channel")
            start = time()
            outputImg = remove_channel(img, True, False, False)
            T["remove_channel"][file] = T["remove_channel"].get(
                file, 0.0) + time() - start

            start = time()
            outputImg = remove_channel(img, False, True, False)
            T["remove_channel"][file] = T["remove_channel"].get(
                file, 0.0) + time() - start

            start = time()
            outputImg = remove_channel(img, False, False, True)
            T["remove_channel"][file] = T["remove_channel"].get(
                file, 0.0) + time() - start

            T["remove_channel"][file] = T["remove_channel"][file] / 3

            # rotations
            print("rotation")
            start = time()
            outputImg = rotations(img)
            T["rotations"][file] = T["rotations"].get(file,
                                                      0.0) + time() - start

            # masking
            print("maksing")
            for mask, mask_path in masks.items():
                print(mask)
                start = time()
                outputImg = apply_mask(img, mask_path,
                                       mask == "sobelx" + mask == "sobely")
                T["apply_mask"][mask][file] = T["apply_mask"].get(
                    file, 0.0) + time() - start
            print()

        object = json.dumps(T, indent=4)
        if p == 0:
            with open('ArrayListtime.json', "w") as outfile:
                outfile.write(object)
        else:
            with open('PointerListtime.json', "w") as outfile:
                outfile.write(object)


if __name__ == "__main__":
    test()
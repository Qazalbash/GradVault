import sys

sys.path.append("./src")

from urllib.request import urlopen
from PIL import Image
from image_operations import *

casefile = "https://waqarsaleem.github.io/cs201/hw1/tests.csv"
OUTPUT_IMAGE = "tmp-output.png"
OUTPUT_MASK = "tmp-mask.txt"


class Case:

    def __init__(self):
        self.source = ""
        self.suppressed = ""
        self.rotated = ""
        self.masked = ""
        self.suppress = []
        self.maskfiles = []
        self.maskaverages = []

    def __repr__(self):
        return (
            f"src: {self.source}, rot: {self.rotated}\n"
            f"sup: {self.suppress}, sup: {self.suppressed}\n"
            f"masks: {self.maskfiles}, avg: {self.maskaverages}, masked: {self.masked}"
        )


def fetch_testcases(path):
    testcases = []
    csv_lines = [
        line.decode("utf-8").strip() for line in urlopen(path).readlines()[1:]
    ]
    for row in csv_lines:
        if not row:
            continue
        row = row.split(",")
        case = Case()
        case.source = row[0]
        if len(row) > 1:
            case.rotated = row[1]
        if len(row) > 5:
            case.suppressed = row[5]
            case.suppress = [f == "1" for f in row[2:5]]
        if len(row) > 8:
            case.masked = row[8]
            case.maskfiles = row[6].split("::")
            case.maskaverages = [a == "1" for a in row[7].split(":")]
        testcases.append(case)
    return testcases


cases = fetch_testcases(casefile)


def test_array_rotation():
    for case in cases:
        if not case.rotated:
            continue
        source = urlopen(case.source)
        rotations(MyImage.open(source)).save(OUTPUT_IMAGE)
        rotated = urlopen(case.rotated)
        assert Image.open(OUTPUT_IMAGE) == Image.open(rotated), (
            f"rotation of {case.source} does not match reference"
            f"{case.rotated}")


def test_array_suppression():
    for case in cases:
        if not case.suppressed:
            continue
        sp = case.suppress
        source = urlopen(case.source)
        remove_channel(MyImage.open(source), red=sp[0], green=sp[1],
                       blue=sp[2]).save(OUTPUT_IMAGE)
        suppressed = urlopen(case.suppressed)
        assert Image.open(OUTPUT_IMAGE) == Image.open(suppressed), (
            f"suppression of {case.source} does not match reference"
            f"{case.suppressed} under channels {case.suppress}")


def test_array_mask():
    for case in cases:
        if not case.masked:
            continue
        avgs = case.maskaverages
        masks = [urlopen(path) for path in case.maskfiles]
        open(OUTPUT_MASK,
             "w").write(urlopen(case.maskfiles[0]).read().decode("utf-8"))
        source = urlopen(case.source)
        dst = apply_mask(MyImage.open(source),
                         OUTPUT_MASK,
                         average=case.maskaverages[0])
        for mask, avg in zip(case.maskfiles[1:], case.maskaverages[1:]):
            open(OUTPUT_MASK, "w").write(urlopen(mask).read().decode("utf-8"))
            dst = apply_mask(dst, OUTPUT_MASK, average=avg)
        dst.save(OUTPUT_IMAGE)
        masked = urlopen(case.masked)
        assert Image.open(OUTPUT_IMAGE) == Image.open(masked), (
            f"masking of {case.source} does not match reference "
            f"{case.masked}\nunder masks {case.maskfiles} "
            f"and averages {case.maskaverages}")


def test_pointer_rotation():
    for case in cases:
        if not case.rotated:
            continue
        source = urlopen(case.source)
        rotations(MyImage.open(source, pointer=True)).save(OUTPUT_IMAGE)
        rotated = urlopen(case.rotated)
        assert Image.open(OUTPUT_IMAGE) == Image.open(rotated), (
            f"rotation of {case.source} does not match reference"
            f"{case.rotated}")


def test_pointer_suppression():
    for case in cases:
        if not case.suppressed:
            continue
        sp = case.suppress
        source = urlopen(case.source)
        remove_channel(MyImage.open(source, pointer=True),
                       red=sp[0],
                       green=sp[1],
                       blue=sp[2]).save(OUTPUT_IMAGE)
        suppressed = urlopen(case.suppressed)
        assert Image.open(OUTPUT_IMAGE) == Image.open(suppressed), (
            f"suppression of {case.source} does not match reference"
            f"{case.suppressed} under channels {case.suppress}")


def test_pointer_mask():
    for case in cases:
        if not case.masked:
            continue
        avgs = case.maskaverages
        masks = [urlopen(path) for path in case.maskfiles]
        open(OUTPUT_MASK,
             "w").write(urlopen(case.maskfiles[0]).read().decode("utf-8"))
        source = urlopen(case.source)
        dst = apply_mask(
            MyImage.open(source, pointer=True),
            OUTPUT_MASK,
            average=case.maskaverages[0],
        )
        for mask, avg in zip(case.maskfiles[1:], case.maskaverages[1:]):
            open(OUTPUT_MASK, "w").write(urlopen(mask).read().decode("utf-8"))
            dst = apply_mask(dst, OUTPUT_MASK, average=avg)
        dst.save(OUTPUT_IMAGE)
        masked = urlopen(case.masked)
        assert Image.open(OUTPUT_IMAGE) == Image.open(masked), (
            f"masking of {case.source} does not match reference "
            f"{case.masked}\nunder masks {case.maskfiles} "
            f"and averages {case.maskaverages}")

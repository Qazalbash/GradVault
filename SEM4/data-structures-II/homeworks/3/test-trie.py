import pytest
import sys
import zipfile

sys.path.append("./src")

from corpus import *
from urllib.request import urlopen

path = "https://waqarsaleem.github.io/cs201/sp2022/hw3/"
path += urlopen(path + "folder.txt").read().decode("utf-8").strip()
corpus_zipfilename = "corpus.zip"
casefilename = "trie-cases.txt"


class Case:
    def __init__(self, prefix: str, terms: int, instances: int):
        self.prefix, self.terms, self.instances = prefix, terms, instances


def get_corpus(zipfilename):
    open(zipfilename, "wb").write(urlopen(path + zipfilename).read())
    zipfile.ZipFile(zipfilename, "r").extractall("corpus")
    return Corpus("corpus/")


def fetch_testcases(path: str) -> [Case]:
    testcases = []
    if path.startswith("http"):
        input_lines = [
            line.decode("utf-8").strip() for line in urlopen(path).readlines()
        ]
    else:
        input_lines = open(path).readlines()
    line = iter(input_lines)
    test_count = int(next(line).strip())
    for _ in range(test_count):
        prefix = next(line).strip()
        terms, instances = map(int, next(line).strip().split())
        testcases.append(Case(prefix, terms, instances))
    return testcases


corpus = get_corpus(corpus_zipfilename)


@pytest.mark.parametrize("case", fetch_testcases(path + casefilename))
def test_trie(case):
    completions = corpus.prefix_complete(case.prefix)
    assert (
        len(completions) == case.terms
    ), f"bad number of completions for {case.prefix=}"
    instances = sum(map(len, completions.values()))
    assert (
        instances == case.instances
    ), f"bad number of completion {instances=} for {case.prefix=}"
    corpuspath = Path("./corpus/")
    for completion, locations in completions.items():
        for location in locations:
            doc_id, start, end = location
            content = open(
                corpuspath / doc_id, encoding="ascii", errors="ignore"
            ).read()
            assert (
                content[start:end] == completion
            ), f"bad {location=} for {completion=} of {case.prefix=}"

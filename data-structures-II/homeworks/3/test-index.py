import pytest
import sys
import zipfile

sys.path.append("./src")

from corpus import *
from urllib.request import urlopen

path = "https://waqarsaleem.github.io/cs201/sp2022/hw3/"
path += urlopen(path + "folder.txt").read().decode("utf-8").strip()
corpus_zipfilename = "corpus.zip"
casefilename = "query-cases.txt"


class Case:

    def __init__(self, query: str, k: int, result: [str]):
        self.query, self.k, self.result = query, k, result


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
        *query, k = next(line).strip().split()
        query = " ".join(query)
        k = int(k)
        result = next(line).strip().split()
        testcases.append(Case(query, k, result))
    return testcases


corpus = get_corpus(corpus_zipfilename)


@pytest.mark.parametrize("case", fetch_testcases(path + casefilename))
def test_index(case):
    result = corpus.query(case.query, case.k)
    result = [doc for _, doc in result]
    assert result == case.result, f"bad {result=} for {case.query=}, {case.k=}"

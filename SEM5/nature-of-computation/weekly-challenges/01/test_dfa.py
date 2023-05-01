import pytest
from urllib.request import urlopen

from dataclasses import dataclass
from itertools import islice

from dfa import *

casefile = 'https://waqarsaleem.github.io/cs212/2022_Fall/wc01/testcases.txt'

@dataclass
class Case:
    dfa: [str]
    strings: [str]
    accepts: [bool]

def fetch_testcases(path: str) -> [Case]:
    testcases = []
    if path.startswith('http'):
        input_lines = [line.decode('utf-8').strip()
                       for line in urlopen(path).readlines()]
    else:
        input_lines = open(path).readlines()
    line = iter(input_lines)
    test_count = int(next(line).strip())
    for _ in range(test_count):
        num_lines = int(next(line).strip())
        dfa = [l.strip() for l in list(islice(line, num_lines))]
        num_cases = int(next(line).strip())
        strings = [l.strip() for l in list(islice(line, num_cases))]
        accepts = [bool(int(l.strip())) for l in list(islice(line, num_cases))]
        testcases.append(Case(dfa, strings, accepts))
    return testcases

@pytest.mark.parametrize('case', fetch_testcases(casefile))
def test_index(case):
    dfa = DFA(case.dfa)
    for w, status in zip(case.strings, case.accepts):
        accept = dfa.accepts(w)
        assert status == accept, \
        f'bad status {accept=} for {w=} by {case.dfa=}'

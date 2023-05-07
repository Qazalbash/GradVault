import pytest
from urllib.request import urlopen
import hashlib
from second_shortest_path import second_shortest

path = 'https://munawwar-anwar.github.io/tests.txt'

HASHES = [
    '36d144ea081f24500bf72163ccde3d47487366cc47e9ea2fa199f60bbbcee648',
    '108c995b953c8a35561103e2014cf828eb654a99e310f87fab94c2f4b7d2a04f',
    'a512db2741cd20693e4b16f19891e72b9ff12cead72761fc5e92d2aaf34740c1',
    'd6d824abba4afde81129c71dea75b8100e96338da5f416d2f69088f1960cb091',
    'd6a4031733610bb080d0bfa794fcc9dbdcff74834aeaab7c6b927e21e9754037',
    '28dae7c8bde2f3ca608f86d0e16a214dee74c74bee011cdfdd46bc04b655bc14',
    '580811fa95269f3ecd4f22d176e079d36093573680b6ef66fa341e687a15b5da',
    'e0f05da93a0f5a86a3be5fc0e301606513c9f7e59dac2357348aa0f2f47db984',
    '6e4001871c0cf27c7634ef1dc478408f642410fd3a444e2a88e301f5c4a35a4d',
    'd86580a57f7bf542e85202283cb845953c9d28f80a8e651db08b2fc0b2d6a731',
    'd4ee9f58e5860574ca98e3b4839391e7a356328d4bd6afecefc2381df5f5b41b',
    'ff2ccb6ba423d356bd549ed4bfb76e96976a0dcde05a09996a1cdb9f83422ec4',
    '0f8ef3377b30fc47f96b48247f463a726a802f62f3faa03d56403751d2f66c67',
    '210e3b160c355818509425b9d9e9fd3ea2e287f2c43a13e5be8817140db0b9e6',
    'd6f0c71ef0c88e45e4b3a2118fcb83b0def392d759c901e9d755d0e879028727',
    '3d3286f7cd19074f04e514b0c6c237e757513fb32820698b790e1dec801d947a',
    'bfa7634640c53da7cb5e9c39031128c4e583399f936896f27f999f1d58d7b37e',
    '3068430da9e4b7a674184035643d9e19af3dc7483e31cc03b35f75268401df77',
    'dbae772db29058a88f9bd830e957c695347c41b6162a7eb9a9ea13def34be56b',
    '3d3286f7cd19074f04e514b0c6c237e757513fb32820698b790e1dec801d947a'
]


def hashcode(n: int) -> str:
    return hashlib.sha256(str(n).encode('utf-8')).hexdigest()


def fetch_testcases(path):
    testcases = []
    input_lines = [
        line.decode('utf-8').strip() for line in urlopen(path).readlines()
    ]
    n = int(input_lines[0])
    input_lines = input_lines[1:]
    for i in range(n):
        size = int(input_lines[0])
        matrix = []
        for j in range(1, size + 1):
            matrix.append(list(map(int, input_lines[j].strip().split())))
        input_lines = input_lines[size + 1:]
        testcases.append(matrix)
    return testcases


testcases = fetch_testcases(path)


@pytest.mark.parametrize('i', range(20))
def test_second(i):
    assert hashcode(second_shortest(testcases[i])) == HASHES[i]

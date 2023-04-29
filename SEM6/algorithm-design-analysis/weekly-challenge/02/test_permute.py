import pytest
import hashlib
from permutation import permute
from urllib.request import urlopen
import requests


def fetch_file():
    urls = ["input1.txt", "input2.txt", "input3.txt"]
    for url in urls:
        URL = "https://munawwar-anwar.github.io/"
        response = requests.get(URL + url)
        open(url, "wb").write(response.content)


fetch_file()


@pytest.mark.parametrize(
    "input,out",
    [("input1.txt",
      "15145b26951c9f20116bfbabcccf5754bd4770e3a47dcc46b5c644f37631b6bf"),
     ("input2.txt",
      "9e0503c4df9870ff65c857d0c45fbf456ee5b85510200aca9b651f4171832d34"),
     ("input3.txt",
      "4a2f2f08a18c90c872b82849d714621e48a7dc58aa59107fe63d7494e8779e37")])
def test_permute(input, out):
    output = []
    with open(input, "r") as f:
        T = f.readlines()
        t = int(T[0])
        for i in range(1, t * 2, 2):
            m, s = map(int, T[i].split())
            a = list(map(int, T[i + 1].strip().split()))
            output.append(permute(m, s, a))
        output_hash = ' '.join(map(str, [i.strip() for i in output]))
        hash = hashlib.sha256(output_hash.encode('utf-8')).hexdigest()
        assert hash == out
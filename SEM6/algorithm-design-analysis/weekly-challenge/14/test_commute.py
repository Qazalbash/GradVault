import hashlib
from itertools import product

import pytest


def commutation_cost(fname: str) -> int:
    """
    Computes the optimal cost to visit all sites in sequence.
    Args:
        fname: the input filename. See the accomapanied latex file for details.
    Returns:
       The optimal cost to visit all sites in sequence.
    """
    with open(fname) as f:
        # Read number of sites and pairs
        numberOfSites, numberOfPairs = map(int, f.readline().split())

        # Initialize graph with infinite weights between all pairs of sites
        graph = [[float('inf')
                  for _ in range(numberOfSites)]
                 for _ in range(numberOfSites)]

        # Nullify diagonal elements
        for i in range(numberOfSites):
            graph[i][i] = 0

        # Read pairs and update graph accordingly
        for _ in range(numberOfPairs):
            u, v, w = map(int, f.readline().split())
            graph[u - 1][v - 1] = w  # -1 because sites are 1-indexed

        # Run Floyd-Warshall to compute shortest paths between all pairs of sites
        for k, i, j in product(range(numberOfSites), repeat=3):
            graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])

        # Read sequence and compute cost of visiting all sites in sequence
        Q = int(f.readline().strip()) - 1
        sequence = list(map(int, f.readline().split()))
        cost = 0
        for i in range(Q):
            # Find the shortest path between sites u and v in sequence
            r, t = sequence[i] - 1, sequence[i + 1] - 1
            cost += min(graph[r][s] + graph[s][t] for s in range(numberOfSites))
        return cost


HASHES = [
    'eb1e33e8a81b697b75855af6bfcdbcbf7cbbde9f94962ceaec1ed8af21f5a50f',
    'c49b0dece16c3a6b89be2938c6fef6d0c91783c0d1b4176a23a0fe6d7f8ad0ba',
    'fe0bae021682b307e3f1d1d28fcfeaaf0fd5eca5d853dbd2688254253fa564bc',
    '7d4c1271a755dfd5e14270393e1d7380055a4f27db2868e1b49a02b4d1083a68',
    '4431ed99fceb3ca10b17a659501a45a69898306629d65288c8d8c4b0083dc0e8',
    '0c943b6adf66adf86bb7f84faa1d9150a1e6e5559bf2dd7ada2f6897cc3b0e0f'
]


def hashcode(n: int) -> str:
    return hashlib.sha256(str(n).encode('utf-8')).hexdigest()


@pytest.mark.parametrize("i", range(len(HASHES)))
def test_survey(i: int):
    fname = f'tests/input_{i}.txt'
    ans = commutation_cost(fname)
    assert hashcode(ans) == HASHES[i], \
        f'Test failed for {fname}'

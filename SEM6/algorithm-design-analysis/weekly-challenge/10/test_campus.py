import pytest
import hashlib
import collections


def bfs(graph, root):  # bfs code from https://www.programiz.com/dsa/graph-bfs

    visited, queue = set(), collections.deque([root])
    visited.add(root)

    while queue:

        # Dequeue a vertex from queue
        vertex = queue.popleft()

        # If not visited, mark it as visited, and
        # enqueue it
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def connect_campus(fname: str):  # -> list:
    '''
    Returns list of boolean values whether the addition of the ith passage makes the campus 
    DEI compliant
    Parameters:
    - fname: the path to the information of the passages on campus
    Returns:
    list of boolean values whether the addition of the ith passage makes the campus 
    DEI compliant
    '''
    A = dict()
    B = dict()

    ispathA = {}
    ispathB = {}
    f = open(fname, "r")
    # print(f)
    nq = f.readline().split()
    n = int(nq[0])
    q = int(nq[1])

    # create graph
    for vertex in range(n):
        A[vertex + 1] = list()
        B[vertex + 1] = list()
        ispathA[vertex + 1] = [0] * n
        ispathB[vertex + 1] = [0] * n
        ispathA[vertex + 1][vertex] = 1
        ispathA[vertex + 1][vertex] = 1

    DEI = []
    for e in range(q):
        edge = f.readline().split()
        pass_type, u, v = edge[0], int(edge[1]), int(edge[2])

        if pass_type == "A":
            A[u].append(v)
            A[v].append(u)
            for s in range(n):
                if 0 in ispathA[s + 1]:
                    pathsS = bfs(A, s + 1)
                    for v in pathsS:
                        ispathA[s + 1][v - 1] = 1

        else:
            B[u].append(v)
            B[v].append(u)
            for s in range(n):
                if 0 in ispathB[s + 1]:
                    pathsS = bfs(B, s + 1)
                    for v in pathsS:
                        ispathB[s + 1][v - 1] = 1

        DEI.append(not any([ispathA[v + 1] != ispathB[v + 1]
                            for v in range(n)]))

    return DEI


HASHES = [
    '122b7f848d6db820beccf6f73a9cb67ff58093326d9945e2f19936b7badcd425',
    'ae9d53302cbcaadce88ff056d64cf9cf6e94cfc42299cc06e3c4d6160994492d',
    '9fb61dad8c1d72ab617996a78420d37a43b4f44b3c961e8d7a316a90a0a4a1a7',
    'a8e6c2c8095e5534d1f4ffeba8a37785b972e28918bdd9c075f24686f718e3d6',
    '99940195088398e727185ac78ec5664a965647a011a069cc609cc7f2ef62d48f',
    '66056a3551184a0821d87820a000be01169e6600a16360dec544fbd83c195e85',
    'f76159dcbaceec2ae2c72964a85a5916fb9ac77322ab96dca860de24d0e6902a',
    '06ce68cfc44ec20c2421f33f34da2b9850fdb9e5c5551a723e954a179b77614b',
    'cfec07a4d3443e12d9a80af6df9088d9f62e1c40d437fdd31e0e2429bc0595fb',
    'b19e363ebc94c57a01ec1c12c4b4961619d1d31a1b01146c9bc8e4b048bc2024',
    'ab104310943b2fee1876ee2e84f576ac2083afd344a2cedc97c2017c7a79d446',
    '5a63b3465352e4e493d5900804c2e2a36e757366cef5e3706249c35483161ab1',
    '2620035085139885af9c41bf26e85fa22a058774abdf96f7fd069222fe55139d',
    'd0ad4d2ab9faeebfcbb059649a36f41b513e32151e1bf40bc00a596a73a8aaaf',
    'bd98435f18584e528724fc1a5c48fe6922009a1d003130ec7ff2c133e58ff963',
    '58606262df0efc0625daed0922e69717c3d3cb9cc5ec1374685bd93f37e69b2c',
    '57d44ce61adf1f7498aaa691a7e8cd13191928bcd1c77651abdaf13b53a606c2',
    '784a5bd45fdc5b3275e4be25dd206a0efbb15aac04e083a2e6874cd00f008003',
    '3fe0979f14452469b787f953e1e9dda1d92454c991b363429b9eb8658fdd922d',
    '500ce1ccad670b81af23cbcd344837f099d9c5631f29e2297f15f57e46ba49f0',
    '8f9426af3db6a92b4195a014ba7dc5f0e4fd4ad82128bbd8467ed80ebf350380',
    '069cda0a1bcfe5a391914656180384d567d831094cfb8a0f6156a901def74758',
    '365b7a3e532ce8a32a99493455e873394febd2d9eadaab5a95f8f08b24459947',
    'd00780b3a2730aef92ecead1275726bc6ee1ef177ff52cc248294d7bb0d855e3',
    'e32e89b183a26b1c9ab7591f084643cb6c07bbe3fd043bf918c0cd440b5cb74d',
    '47fe82153e1ffb7e899f87e3ce96722a8215304060d745c997de670e1fcfae39',
    '41110de8befb1ad44af91702fd11a7ffd40432d96490c17ac6ed27c9f28e973c',
    '2ec4ba9a81693976bc99604602ae6e5c41d95529094abe8651681b30de045d78',
    '440a71b2db0283b09db1fab25d0e4ceee5b86c0a7cfbfb2542a39ec1a270a2c1',
    'f47f75cb63517da14214e79988a2874ef791ad7db498bfa0c6a186b81ac17836',
    'f206ce10b3d93e351a97a8be7a446d91e0d5949795670a028f6c7427cb476f0e',
    'b58dd477412084de8cae3d1e57f69154084790cfb6f9f705954b9ff813c28cbc',
    '12577ac90c562cd46fba41ad6d192b66b064e3aa974a363c267fd704e1a0ed1a',
    'e8c78f82bee69304e36ae3db849464cac5fb9cbc0aa7cb4e5c26e9e56622ab1e',
    '7b8420c53f0452f236e70f2ffb6d00256b7797b7093a33bf3331147b4d4219f7',
    'e6cbe63b4a3f9299a15fa990c920f7b00cd09a1dd9bb7d8149bdbb3daf031756',
    '11f7bff1dbbb5ae8cd44c5e39df8088d335394e5b1f01e6be175e3442ff959bc',
    '2df1a7b8670d00b88da6345c6fb86d4fa045a6f560334f493db35d9fd2e87601',
    'b3f995d81dcede0676ee7f0a8d8fde18b05f2e0dd72e4b4145df937a6fac01e4',
    'd6db0d05e281cbab8d8567cd37fab31ed1d21a2e450a4850c2dbd8fec60bd8a6',
    '6cf258d784899beb651bb2d5fad9d2f7e3032f7f8b0f082623b2275c482bfe96',
    '5b3d0c6ebd1eda1e70ac34b0843a81cfde39f3cb05e881d6f76915a40c16522f',
    '0dca3394eab5b9969e94986b16a67399f900c50d5f05ca59484221c11745869d',
    '17b0ad534c5a58cb1f940a7b2cef87af54ad868324a94d9540861f696f63e78f',
    'b3e942b3487341ca0ee90efbf3ecf820570d6664f369f1bfe2c3767b2e2769a7',
    'fe035217d9a49fd3eb524691e8608f42580f47863d1ee2c70475a915a19952bd',
    '5a9986cde0c672e4056bdcfc1ec731c13640bba17b0e5e7e3db556b29034684a',
    'c34b432aabff7f678dd401f57079fef9529678595af2ca07ac6324d6683925bb',
    'c6278c68457d892cc290d16d845ff46f5f3aa5797b37e97d0a8e338f53bebd1c',
    '0a13589251d204cc77d6375233a2e813a6a8e00b30ae94add795e43e31ac0966',
    '193b06c52d62d68cd61e8d6b1573acf1cabb3293af5f012a74bbbab2be0c3c93',
    '95707a4ce560c93583c5611d5f6f026f2a2de5c8800e663892dbd13ea6a03439',
    '656d7371a3240bd5e9b8b45b4f4fa19eee033947b91984caa79b0803405b2fbe',
    '4716918f7b2a9a15594aa7aba668fd2edaa39f4bf353b74b8ad30425868d745e',
    '258fcb285c9290188c875f2a9dedd06516e34cde491be31321ee6f011c72788d',
    '5acafcf3aa316712086833b36df2b151cb7f4f24f12cabc052274b7413a10764',
    'f4e0f9b0e9215e44728bc64bf25cfd5b30f8eff5533092ba64d4eae4180162af',
    '030450a25924e0c1fed711c8a70d28ca073bc8c6b7585c0728670923995f81ab',
    'b45bd32ccbc1be3ed8782910ca7cb602360aa92eed1fd30fbbae79dd92653581',
    '76bed5411c2fc23317234cdab56877779cc60bb12f48b629e0ada18ccf54a844',
    'e7775423ffd55f434f05962d2cd6922bd430225829b1354ccb1cb579b3b6cf44'
]


def hashcode(n: int) -> str:
    return hashlib.sha256(str(n).encode('utf-8')).hexdigest()


@pytest.mark.parametrize("i", range(61))
def test_connect_campus(i: int):
    fname = f'tests/survey_{i+1}.txt'
    assert hashcode(connect_campus(fname)) == HASHES[i], \
       f'Test failed for {fname}'


# print(connect_campus('tests/survey_1.txt'))

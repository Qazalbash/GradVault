def addNodes(G, nodes):
    G.update({node: [] for node in nodes})


def addEdges(G, edges):
    for edge in edges:
        G[edge[0]] = G.get(edge[0], []) + [edge[1]]
        G[edge[1]] = G.get(edge[1], []) + [edge[0]]


def listOfNodes(G):
    return list(G.keys())


def getNeighbors(G, node):
    return [neighbour for neighbour in G[node]]


def removeNode(G, node):
    for i in getNeighbors(G, node):
        G[i] = [j for j in G[i] if j != node]
    del G[node]


def displayGraph(G):
    print("G = {")
    for node in G.keys():
        print(f"\t{node}: {G[node]},")
    print("}")


def degree(G, node):
    return len(G[node])


G = {}

addNodes(G, [i for i in range(9)])
addEdges(
    G,
    [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (3, 4),
        (3, 6),
        (3, 7),
        (4, 6),
        (4, 7),
        (5, 6),
        (5, 8),
        (6, 7),
        (6, 8),
    ],
)


def kCores(graph, k):
    flag = 0
    for node in listOfNodes(graph):
        if degree(graph, node) < k:
            removeNode(graph, node)
            flag = 1
    if flag == 1:
        kCores(graph, k)
    else:
        displayGraph(graph)


kCores(G, 3)

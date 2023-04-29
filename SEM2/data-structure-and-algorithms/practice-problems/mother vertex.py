def addNodes(G, nodes):
    G.update({node: [] for node in nodes})


def addEdges(G, edges):
    for edge in edges:
        G[edge[0]] = G.get(edge[0], []) + [edge[1]]


def listOfNodes(G):
    return list(G.keys())


def getNeighbors(G, node):
    return [neighbour for neighbour in G[node]]


def deQueue(lst):
    return lst.pop(0)


def is_empty(lst):
    return len(lst) == 0


def enQueue(lst, item):
    lst.append(item)


def bfs(G, source):
    visited = []
    check, q = [False] * len(G), [source]
    while not (is_empty(q)):
        v = deQueue(q)
        if not (check[v]):
            visited.append(v)
            check[v] = True
            for w in getNeighbors(G, v):
                if not (check[w]):
                    enQueue(q, w)
    return visited


def motherVertex(graph):
    result = []
    nodes = listOfNodes(graph)
    for node in nodes:
        count = 0
        for j in nodes:
            if j in bfs(graph, node):
                count += 1
        if count == len(nodes):
            result.append(node)
    return result


graph1 = {}
addNodes(graph1, [i for i in range(7)])
addEdges(graph1, [(0, 1), (0, 2), (1, 3), (4, 1), (5, 2), (5, 6), (6, 0),
                  (6, 4)])

graph2 = {}
addNodes(graph2, [i for i in range(5)])
addEdges(graph2, [(0, 3), (0, 2), (1, 0), (2, 1), (3, 4)])

print(motherVertex(graph1))
print(motherVertex(graph2))

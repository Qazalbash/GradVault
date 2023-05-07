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


def inParent(G, node):
    return [i for i in G.keys() if node in G[i]]


def inDegree(G, node):
    return len(inParent(G, node))


courses = {}
nodes = [
    "LA15", "LA16", "LA22", "LA31", "LA32", "LA126", "LA127", "LA141", "LA169"
]
edges = [
    ("LA15", "LA16"),
    ("LA15", "LA31"),
    ("LA16", "LA32"),
    ("LA16", "LA127"),
    ("LA16", "LA141"),
    ("LA22", "LA16"),
    ("LA22", "LA126"),
    ("LA22", "LA141"),
    ("LA31", "LA32"),
    ("LA32", "LA126"),
    ("LA32", "LA169"),
]

addNodes(courses, nodes)
addEdges(courses, edges)


def courseSchedule(G, source):
    sources = [i for i in listOfNodes(G) if inDegree(G, i) == 0]
    addNodes(G, "S")
    addEdges(G, [("S", j) for j in sources])
    visited = []
    nodes_ = listOfNodes(G)
    enc = {node: i for node, i in zip(nodes_, range(len(nodes_)))}
    check, q = [False] * len(G), ["S"]

    while not (is_empty(q)):

        v = deQueue(q)
        if not (check[enc[v]]):

            visited.append(v)
            check[enc[v]] = True

            for w in getNeighbors(G, v):
                if not (check[enc[w]]) and all(
                    [check[enc[i]] for i in inParent(G, w)]):
                    enQueue(q, w)

    return visited[1:]


print(courseSchedule(courses, "S"))

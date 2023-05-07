def addNodes(G, nodes):
    G.update({node: [] for node in nodes})


def addEdges(G, edges, directed=False):
    for edge in edges:
        G[edge[0]] = G.get(edge[0], []) + [(edge[1], edge[2])]
        if not (directed):
            G[edge[1]] = G.get(edge[1], []) + [(edge[0], edge[2])]


def listOfNodes(G):
    return list(G.keys())


def getNeighbors(G, node):
    return [neighbour[0] for neighbour in G[node]]


def is_empty(lst):
    return len(lst) == 0


def enQueue(lst, item):
    lst.append(item)


def deQueue(lst):
    return lst.pop(0)


print(
    "************************************exercise 1************************************\n\n"
)

G = {}
addNodes(G, [i for i in range(6)])
addEdges(G, [(0, 1, 1), (0, 2, 1), (1, 2, 1), (1, 3, 1), (2, 4, 1), (3, 4, 1),
             (3, 5, 1), (4, 5, 1)], True)


def dfs(G, node, visited=[], check=[False] * len(G)):
    visited.append(node)
    check[node] = True
    for w in getNeighbors(G, node):
        if not (check[w]):
            dfs(G, w, visited, check)
    if all(check):
        return visited


print(dfs(G, 0))

print(
    "\n\n************************************exercise 2************************************\n\n"
)

nodes = [
    "Dallas", "Austin", "Denver", "Washington", "Chicago", "Houston", "Atlanta"
]

edges = [("Dallas", "Austin", 200), ("Dallas", "Denver", 780),
         ("Dallas", "Chicago", 900), ("Austin", "Dallas", 200),
         ("Austin", "Houston", 160), ("Denver", "Chicago", 1000),
         ("Denver", "Atlanta", 1400), ("Washington", "Dallas", 1300),
         ("Washington", "Atlanta", 600), ("Chicago", "Denver", 1000),
         ("Houston", "Atlanta", 800), ("Atlanta", "Houston", 800),
         ("Atlanta", "Washington", 600)]

rosen = {}

addNodes(rosen, nodes)
addEdges(rosen, edges, True)


def check_cycles(G, airports, index=0, visited=[], check=[False] * len(G)):
    encrypt = {airports[i]: i for i in range(len(airports))}
    visited.append(encrypt[airports[index]])
    check[encrypt[airports[index]]] = True
    for w in getNeighbors(G, airports[index]):
        try:
            if check[encrypt[w]]:
                return "Yes"
        except KeyError:
            return "No"
        if not (check[encrypt[w]]):
            check_cycles(G, airports, index + 1, visited, check)
    return "No"

    # for i in range(len(airports)-1):
    #     if not(airports[i+1] in getNeighbors(G, airports[i])):
    #         return "No"
    # if not(airports[0] in getNeighbors(G, airports[-1])):
    #     return "No"
    # return "Yes"


print(
    check_cycles(rosen,
                 ['Austin', 'Houston', 'Atlanta', 'Washington', 'Dallas']))
print(check_cycles(rosen, ['Austin', 'Houston', 'Atlanta', 'Washington']))

print(
    "\n\n************************************exercise 3************************************\n\n"
)

bfsG = {}
addNodes(bfsG, [i for i in range(8)])
addEdges(bfsG, [(0, 1, 1), (1, 3, 1), (1, 4, 1), (1, 5, 1), (0, 2, 1),
                (2, 6, 1), (6, 7, 1)])


def bfs(G, source, visited=[]):
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


print(bfs(bfsG, 0))

print(
    "\n\n************************************exercise 4a************************************\n\n"
)


def nodes_of_level(G, lev, source=0, visited=[]):
    check, q, level = [False] * len(G), [source], [0] * len(G)
    while not (is_empty(q)):
        v = deQueue(q)
        if not (check[v]):
            visited.append(v)
            check[v] = True
            for w in getNeighbors(G, v):
                if not (check[w]):
                    level[w] = level[v] + 1
                    enQueue(q, w)
    return [i for i in range(len(level)) if level[i] == lev]


print(nodes_of_level(bfsG, 1))
print(nodes_of_level(bfsG, 2))

print(
    "\n\n************************************exercise 4b************************************\n\n"
)


def get_node_level(G, node, source=0, visited=[]):
    check, q, level = [False] * len(G), [source], [0] * len(G)
    while not (is_empty(q)):
        v = deQueue(q)
        if not (check[v]):
            visited.append(v)
            check[v] = True
            for w in getNeighbors(G, v):
                if not (check[w]):
                    level[w] = level[v] + 1
                    enQueue(q, w)
    return level[node]


print(get_node_level(bfsG, 2))
print(get_node_level(bfsG, 6))

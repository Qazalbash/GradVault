def edgesToGraph(G, EL):
    for edge in EL:
        G[edge[0]] = G.get(edge[0], []) + [(edge[1], edge[2])]
        G[edge[1]] = G.get(edge[1], []) + [(edge[0], edge[2])]


def listOfEdges(G):
    edgesList = []
    for key, value in G.items():
        edgesList += [
            (key, i[0], i[1])
            for i in value
            if ((key, i[0], i[1]) not in edgesList and (i[0], key,
                                                        i[1]) not in edgesList)
        ]
    return edgesList


def matToList(arr):
    G = {}
    n = len(arr)
    for node in range(1, n + 1):
        G[node] = G.get(node, []) + [
            (i, j) for i, j in zip(range(1, n + 1), arr[node - 1]) if j > 0
        ]
    return G


def getEdge(edges, minimum=True):
    prior_edge, priority = tuple(edges[0][:2]), edges[0][2]
    if minimum:
        for edge in edges:
            if priority > edge[2]:
                prior_edge, priority = tuple(edge[:-1]), edge[-1]
    else:
        for edge in edges:
            if priority < edge[2]:
                prior_edge, priority = tuple(edge[:-1]), edge[-1]

    return prior_edge[0], prior_edge[1], priority


print("*" * 40 + " exercise 1a " + "*" * 40 + "\n\n")

G = {}

edgesToGraph(G, [("A", "B", 5), ("A", "D", 3), ("A", "E", 6), ("B", "C", 6),
                 ("C", "D", 10), ("C", "G", 2), ("D", "F", 8), ("E", "F", 9),
                 ("F", "G", 10)])

# def minEdge(edges):
#     min_Edge, minimum = tuple(edges[0][0:2]), edges[0][2]
#     for i in edges:
#         if minimum > i[2]:
#             min_Edge, minimum = tuple(i[:-1]), i[-1]
#     return min_Edge[0], min_Edge[1], minimum


def MinSTPrim(G, node):
    queue, visited, cost, tree = [(node, i[0], i[1]) for i in G[node]
                                 ], [node], 0, []
    while len(queue):
        # vertex = minEdge(queue)
        vertex = getEdge(queue)
        queue.remove(vertex)
        if vertex[1] not in visited:
            visited.append(vertex[1])
            cost += vertex[2]
            tree.append(vertex)
        queue += [
            (vertex[1], i[0], i[1]) for i in G[vertex[1]] if i[0] not in visited
        ]
        # for i in G[vertex[1]]:
        #     if i[0] not in visited:
        #         queue.append((vertex[1], i[0], i[1]))
    return tree, cost


print(f"Minimum Spanning Tree = {MinSTPrim(G, 'A')}")

print("\n\n" + "*" * 40 + " exercise 1b " + "*" * 40 + "\n\n")

# def maxEdge(edges):
#     max_Edge, maximum = tuple(edges[0][0:2]), edges[0][2]
#     for i in edges:
#         if maximum < i[2]:
#             max_Edge, maximum = tuple(i[:-1]), i[-1]
#     return max_Edge[0], max_Edge[1], maximum


def MaxSTPrim(G, node):
    queue, visited, cost, tree = [(node, i[0], i[1]) for i in G[node]
                                 ], [node], 0, []
    while len(queue):
        # vertex = maxEdge(queue)
        vertex = getEdge(queue, False)
        queue.remove(vertex)
        if vertex[1] not in visited:
            visited.append(vertex[1])
            cost += vertex[2]
            tree.append(vertex)
        queue += [
            (vertex[1], i[0], i[1]) for i in G[vertex[1]] if i[0] not in visited
        ]
        # for i in G[vertex[1]]:
        #     if i[0] not in visited:
        #         queue.append((vertex[1], i[0], i[1]))
    return tree, cost


print(f"Maximum Spanning Tree = {MaxSTPrim(G, 'A')}")

print("\n\n" + "*" * 40 + " exercise 2a " + "*" * 40 + "\n\n")

GK = {}

edgesToGraph(GK, [("A", "B", 5), ("A", "D", 3), ("A", "E", 6), ("B", "C", 6),
                  ("C", "D", 10), ("C", "G", 2), ("D", "F", 8), ("E", "F", 9),
                  ("F", "G", 10)])


def MSTKruskal(G, visit=[], tree={}):
    edges = listOfEdges(G)
    for edge in edges:
        if edge[1] not in tree.keys():
            edgesToGraph(tree, [edge])
            MSTKruskal(G, visit, tree)
    result = listOfEdges(tree)
    cost = 0
    for i in result:
        cost += i[-1]
    return result, cost


print(MSTKruskal(GK))

print("\n\n" + "*" * 40 + " exercise 3 " + "*" * 40 + "\n\n")

arr = [[0, 240, 210, 340, 280, 200, 345, 120],
       [0, 0, 265, 175, 215, 180, 185, 155], [0, 0, 0, 260, 115, 350, 435, 195],
       [0, 0, 0, 0, 160, 330, 295, 230], [0, 0, 0, 0, 0, 360, 400, 170],
       [0, 0, 0, 0, 0, 0, 175, 205], [0, 0, 0, 0, 0, 0, 0, 305],
       [0, 0, 0, 0, 0, 0, 0, 0]]

lst = matToList(arr)

print(MinSTPrim(lst, 1))

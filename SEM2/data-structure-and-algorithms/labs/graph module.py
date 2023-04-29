def addNodes(G, nodes):
    G.update({node: [] for node in nodes})

def addEdges(G, edges, directed = False):
    for edge in edges:
        G[edge[0]] = G.get(edge[0], []) + [(edge[1], edge[2])]
        if not(directed):
            G[edge[1]] = G.get(edge[1], []) + [(edge[0], edge[2])]

def listOfNodes(G):
    return list(G.keys())

def listOfEdges(G, directed = False):
    # here directed is redundant. but i added it because of the question.
    edgesList = []
    for key, value in G.items():
        edgesList += [(key, i[0], i[1]) for i in value if ((key, i[0], i[1]) not in edgesList and (i[0], key, i[1]) not in edgesList)]
    return edgesList

def print_OutDegree(G):
    inDegree, outDegree = {i: 0 for i in G.keys()}, {i: 0 for i in G.keys()}
    for node, edge in G.items():
        outDegree[node] = len(edge)
        for itter in edge:
            inDegree[itter[0]] = inDegree.get(itter[0], 0) + 1
    for node in G.keys():
        print(f"{node} => In-Degree: {inDegree[node]}, Out-Degree: {outDegree[node]}")

def printDegree(G):
    for node in G.keys():
        print(f"{node} => {len(G[node])}")

def getNeighbors(G, node):
    return [neighbour[0] for neighbour in G[node]]

def getInNeighbours(G, node):
    neighbors = []
    for Node, edge in G.items():
        for itter in edge:
            if itter[0] == node:
                neighbors.append(Node)
    return neighbors

def getOutNeighbours(G, node):
    return getNeighbors(G, node)

def getNearestNeighbor(G, node):
    nearest, minimun = G[node][0][0], G[node][0][1]
    for edge in G[node]:
        if edge[1] < minimun:
            minimun, nearest = edge[1], edge[0]
    return nearest

def isNeighbor(G, Node1, Node2):
    for vertex in G[Node1]:
        if vertex[0] == Node2:
            return True
    return False

def removeNode(G, node):
    for i in getNeighbors(G, node):
        G[i] = [j for j in G[i] if j[0] != node]
    del G[node]

def removeNodes(G, nodes):
    for node in nodes:
        removeNode(G, node)

def displayGraph(G):
    print("G = {")
    for node in G.keys():
        print(f"{node}: {G[node]},")
    print("}")

def display_adj_matrix(G):
    matrix = [[0 for col in range(len(G.keys()))] for row in range(len(G.keys()))]
    encryption = {node: i for i, node in zip(range(len(G.keys())), G.keys())}
    for node, edge in G.items():
        for itter in edge:
            matrix[encryption[node]][encryption[itter[0]]] = itter[1]
    print(matrix)

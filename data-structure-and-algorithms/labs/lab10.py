def addNodes(G, nodes):
    G.update({node: [] for node in nodes})


def addEdges(G, edges, directed=False):
    for edge in edges:
        G[edge[0]] = G.get(edge[0], []) + [(edge[1], edge[2])]
        if not (directed):
            G[edge[1]] = G.get(edge[1], []) + [(edge[0], edge[2])]


def listOfNodes(G):
    return list(G.keys())


def listOfEdges(G, directed=False):
    edgesList = []
    for key, value in G.items():
        edgesList += [
            (key, i[0], i[1])
            for i in value
            if ((key, i[0], i[1]) not in edgesList and (i[0], key,
                                                        i[1]) not in edgesList)
        ]
    return edgesList


def print_OutDegree(G):
    inDegree, outDegree = {i: 0 for i in G.keys()}, {i: 0 for i in G.keys()}
    for node, edge in G.items():
        outDegree[node] = len(edge)
        for itter in edge:
            inDegree[itter[0]] = inDegree.get(itter[0], 0) + 1
    for node in G.keys():
        print(
            f"{node} => Out-Degree: {outDegree[node]}, In-Degree: {inDegree[node]}"
        )


def getNeighbors(G, node):
    return [neighbour[0] for neighbour in G[node]]


def getInNeighbours(G, node):
    neighbors = []
    for Node, edge in G.items():
        for itter in edge:
            if itter[0] == node:
                neighbors.append(Node)
    return neighbors


def displayGraph(G):
    print("G = {")
    for node in G.keys():
        print(f"{node}: {G[node]},", end="\n")
    print("}")


def display_adj_matrix(G):
    matrix = [
        [0 for col in range(len(G.keys()))] for row in range(len(G.keys()))
    ]
    encryption = {node: i for i, node in zip(range(len(G.keys())), G.keys())}
    for node, edge in G.items():
        for itter in edge:
            matrix[encryption[node]][encryption[itter[0]]] = itter[1]
    for line in matrix:
        print(line)


# ------------------------------
# exercise 1
# ------------------------------

print("\n\nBehold folks exercise 1 starts from here\n\n")

UG = {}
# a
addNodes(UG, [1, 2, 3, 4, 5])
# b
edges = [(1, 2, 1), (1, 5, 1), (2, 5, 1), (2, 4, 1), (2, 3, 1), (3, 4, 1),
         (4, 5, 1)]
addEdges(UG, edges, False)
# c
print(f"part c\n\n{listOfEdges(UG)}")
# d
print("\npart d\n")
display_adj_matrix(UG)
# e
print("\npart e\n")
displayGraph(UG)
# f
print("\npart f\n")
for node in listOfNodes(UG):
    print(
        f"{node}: Neighbour => {getNeighbors(UG, node)}, Degree: {len(getNeighbors(UG, node))}"
    )

# -------------------------------
# exercise 2
# -------------------------------

print("\n\nladies and gentelmen here we mark the start of exercise 2\n\n")
# a
DG = {}
vertices = [1, 2, 3, 4]
edges = [(1, 2, 1), (2, 4, 1), (3, 1, 1), (3, 2, 1), (4, 3, 1), (4, 4, 1)]
addNodes(DG, vertices)
addEdges(DG, edges, True)
# b
print("part b\n")
displayGraph(DG)
# c
print("\npart c\n")
print_OutDegree(DG)
# d
print("\npart d\n")
for node in listOfNodes(DG):
    print(f"{node}: {getNeighbors(DG, node)},")
# e
print("\npart e\n")
inDegree, outDegree = {i: 0 for i in DG.keys()}, {i: 0 for i in DG.keys()}
for node, edge in DG.items():
    outDegree[node] = len(edge)
    for itter in edge:
        inDegree[itter[0]] = inDegree.get(itter[0], 0) + 1
print(
    sum(list(inDegree.values())) == sum(list(outDegree.values())) == len(
        listOfEdges(DG, True)))

# -------------------------------
# exercise 3
# -------------------------------

print("\n\nwe have come so far to the start of exercise 3\n\n")

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
# a
print("part a\n")
displayGraph(rosen)
# b
print("\npart b\n")


def inBound(G):
    inD = {i: 0 for i in G.keys()}
    for node, edge in G.items():
        for itter in edge:
            inD[itter[0]] = inD.get(itter[0], 0) + 1
    maximum = 0
    Node = None
    for node in inD:
        if inD[node] > maximum:
            maximum = inD[node]
            Node = node
    print(f"Maximum number of Inbound is: {Node}")


def outBound(G):
    outD = {i: 0 for i in G.keys()}
    for node, edge in G.items():
        outD[node] = len(edge)
    maximum = 0
    Node = None
    for node in outD:
        if outD[node] > maximum:
            maximum = outD[node]
            Node = node
    print(f"Maximum number of Outbound is: {Node}")


outBound(rosen)
inBound(rosen)

# c
print("\npart c\n")


def introvertAirports(G):
    frequency = {}
    for a1, mate in G.items():
        for a2 in mate:
            pair = (a2[0], a1)
            if a1 > a2[0]:
                pair = (a1, a2[0])
            frequency[pair] = frequency.get(pair, 0) + 1
    result = []
    for pairs, freq in frequency.items():
        if freq == 1:
            result.append(pairs)
    return result


print(introvertAirports(rosen))

# d
print("\npart d\n")


def nearestAirport(G, airport):
    minimum = G[airport][0][1]
    a = G[airport][0][0]
    for i in G[airport]:
        if i[1] < minimum:
            a = i[0]
            minimum = i[1]
    return f"""Nearest_Airport("{airport}") will give: {a}"""


print(nearestAirport(rosen, "Dallas"))
print(nearestAirport(rosen, "Austin"))
print(nearestAirport(rosen, "Denver"))
print(nearestAirport(rosen, "Washington"))
print(nearestAirport(rosen, "Chicago"))
print(nearestAirport(rosen, "Houston"))
print(nearestAirport(rosen, "Atlanta"))

# e
print("\npart e\n")


def connectedFlights(G, airport):
    f1 = getInNeighbours(G, airport)
    result = [j for j in f1]
    for i in f1:
        result += getInNeighbours(G, i)
    return [k for k in result if k != airport]


print(connectedFlights(rosen, "Dallas"))

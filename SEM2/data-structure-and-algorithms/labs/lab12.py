import pandas as pd

def addNodes(G, nodes):
    G.update({node: [] for node in nodes})

def addEdges(G, edges):
    for edge in edges:
        G[edge[0]] = G.get(edge[0], []) + [(edge[1], edge[2])]
        G[edge[1]] = G.get(edge[1], []) + [(edge[0], edge[2])]

def totalWeight(G):
    count = 0
    for i in G.values():
        count += sum([int(j[1]) for j in i])
    return count

print("************************************************exercise 1a************************************************\n\n")
G = {}
addNodes(G, ["A", "B", "C", "D", "E", "F", "G"])
addEdges(G, [("A", "B", 5), ("A", "E", 6), ("A", "D", 3), ("B", "C", 6), ("C", "D", 10), ("C", "G", 2), ("D", "F", 8), ("E", "F", 9), ("F", "G", 10)])

def getShortestPath(graph, source, sink):
    track, unvisitedNodes, inf, route = {}, graph.copy(), totalWeight(graph) + 1, []
    shortestDist = {node: inf for node in graph}
    shortestDist[source] = 0

    while unvisitedNodes:
        minDistNode = None

        for node in unvisitedNodes:
            if minDistNode == None:
                minDistNode = node
            elif shortestDist[node] < shortestDist[minDistNode]:
                minDistNode = node
        
        neighbours = graph[minDistNode]
        for i in neighbours:
            if i[1] + shortestDist[minDistNode] < shortestDist[i[0]]:
                shortestDist[i[0]] = i[1] + shortestDist[minDistNode]
                track[i[0]] = minDistNode
        unvisitedNodes.pop(minDistNode)
    
    currentNode = sink

    while currentNode != source:
        route.insert(0, currentNode)
        currentNode = track[currentNode]
    route.insert(0, source)
    return [(route[i], route[i+1])for i in range(len(route)-1)]

print(getShortestPath(G, "A", "G"))


print("\n\n************************************************exercise 1b************************************************\n\n")
city = {}

# import csv
# with open("connections.csv", "r") as csvfile:
#     csv_reader = csv.reader(csvfile, delimiter=",")
#     count = 0
#     cities = None
#     for line in csv_reader:
#         if count == 0:
#             cities = line[1:]
#             addNodes(city, cities)
#         else:
#             city[line[0]] = [(cities[i-1], int(line[i])) for i in range(1, len(cities)) if int(line[i]) > 0]
#         count = 1

data = pd.read_csv("connections.csv")
cities = [i for i in data][1:]
addNodes(city, cities)

for node in cities:
    city[node] = [(i, j) for i, j in zip(cities, data[node]) if j > 0]

print(getShortestPath(city, "Islamabad", "Kaghan"))

print("\n\n************************************************exercise 2************************************************\n\n")

G2 = {}

addNodes(G2, ["A", "B", "C", "D", "E", "F", "G"])
addEdges(G2, [
    ("A", "B", 7), ("A", "E", 6), ("A", "D", 2), ("B", "C", 3), ("C", "D", 2), ("C", "G", 2), ("D", "F", 8), ("E", "F", 9), ("F", "G", 4)
])

print(getShortestPath(G2, "A", "F"))
print(getShortestPath(G2, "A", "B"))

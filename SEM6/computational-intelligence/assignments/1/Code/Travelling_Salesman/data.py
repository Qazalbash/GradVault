import tsplib95

file = "Datasets/qa194.tsp"

with open(file) as f:
    text = f.read()

problem = tsplib95.parse(text)

graph = problem.get_graph()

num_nodes = int(problem.dimension)

matrix = []

for node in range(1, num_nodes + 1):
    connections = list(graph[node].keys())
    weights = [graph[node][city]["weight"] for city in connections]
    matrix.append(weights)

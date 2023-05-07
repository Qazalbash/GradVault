file = "Datasets/gcol1.txt"

openedFile = open(file, "r")
lines = openedFile.read().split("\n")

num_nodes = int(lines[0].split(" ")[2])
num_edges = int(lines[0].split(" ")[3])

del lines[0]

matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

for i in range(num_edges):
    _, u, v = lines[i].split(" ")
    u = int(u) - 1
    v = int(v) - 1
    matrix[u][v] = 1
    matrix[v][u] = 1


def is_valid(individual):
    return all(matrix[i][j] == 0 or individual[i] != individual[j]
               for i in range(num_nodes)
               for j in range(i, num_nodes))

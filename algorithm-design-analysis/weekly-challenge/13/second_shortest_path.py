from typing import List
import collections
from collections import deque


def Dijikstra(graph: List[List[int]], source: int, sink: int) -> List[int]:
    """
    Computes the shortest distance of the given graph using Dijikstra's algorithm.
    Args:
        graph: a 2D array representing the graph in the form of an adjacency matrix.
        source: the starting node.
        sink: the ending node.
    Returns:
       The shortest path from source to sink in the given graph using Dijikstra's algorithm.
    """
    n = len(graph)
    dist = [float("inf")] * n
    dist[source] = 0
    prev = [None] * n
    q = collections.deque()
    q.append(source)
    while q:
        u = q.popleft()
        for v in range(n):
            if graph[u][v] != float("inf") and dist[v] > dist[u] + graph[u][v]:
                dist[v] = dist[u] + graph[u][v]
                prev[v] = u
                q.append(v)

    path = []
    currentNode = sink
    while currentNode != source:
        path.insert(0, currentNode)
        currentNode = prev[currentNode]
    path.insert(0, source)
    return [path[i] for i in range(len(path))]


def YenKSP(graph: List[List[int]], source: int, sink: int,
           K: int) -> List[List[int]]:
    """
    Computes the K shortest paths of the given graph using Yen's algorithm.
    Args:
        graph: a 2D array representing the graph in the form of an adjacency matrix.
        source: the starting node.
        sink: the ending node.
        K: the number of shortest paths to compute.
    Returns:
       The K shortest paths from source to sink in the given graph using Yen's algorithm.
    """
    A = [Dijikstra(graph, source, sink)]
    B = []
    for k in range(1, K):
        for i in range(len(A[k - 1]) - 1):
            spurNode = A[k - 1][i]
            rootPath = A[k - 1][:i + 1]
            for path in A:
                if rootPath == path[:i + 1]:
                    graph[path[i]][path[i + 1]] = float("inf")
            for path in B:
                if rootPath == path[:i + 1]:
                    graph[path[i]][path[i + 1]] = float("inf")
            spurPath = Dijikstra(graph, spurNode, sink)
            if spurPath:
                totalPath = rootPath[:-1] + spurPath
                B.append(totalPath)
            for path in A:
                if rootPath == path[:i + 1]:
                    graph[path[i]][path[i + 1]] = graph[path[i + 1]][path[i]]
            for path in B:
                if rootPath == path[:i + 1]:
                    graph[path[i]][path[i + 1]] = graph[path[i + 1]][path[i]]
        if B:
            B.sort(key=lambda x: sum(
                [graph[x[i]][x[i + 1]] for i in range(len(x) - 1)]))
            A.append(B[0])
            B.pop(0)
        else:
            break
    return A


def second_shortest(adj_matrix: List[List[int]]) -> int:
    """
    Computes the second shortest distance of the given graph using an adjacency matrix.
    Args:
        adj_matrix: a 2D array representing the graph in the form of an adjacency matrix.
    Returns:
       The second shortest distance of the given graph using an adjacency matrix.
    """
    n = len(adj_matrix)
    for i in range(n):
        for j in range(n):
            if not (1 <= adj_matrix[i][j] <= 1000):
                adj_matrix[i][j] = float("inf")
    paths = YenKSP(adj_matrix, 0, n - 1, 2)
    return sum([
        adj_matrix[paths[1][i]][paths[1][i + 1]]
        for i in range(len(paths[1]) - 1)
    ])

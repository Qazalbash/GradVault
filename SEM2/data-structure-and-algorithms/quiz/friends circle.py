import ast

G = ast.literal_eval(input())


def DFS(G, node, visit=[]):
    visit.append(node)
    for edge in G[node]:
        if edge not in visit:
            DFS(G, edge, visit)
    return visit


def BFS(G, node):
    from queue import Queue

    queue = Queue()
    queue.put(node)
    visited = []
    while not queue.empty():
        vertex = queue.get()
        visited.append(vertex)
        for i in G[vertex]:
            if i not in visited:
                queue.put(i)
    return visited


def insertionSort(lst):
    for i in range(1, len(lst)):
        key, j = lst[i], i - 1
        while key < lst[j] and j >= 0:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = key
    return lst


def extractFriendCircles(G):
    results = []
    for i in list(G.keys()):
        circle = insertionSort(DFS(G, i, visit=[]))
        if circle not in results:
            results.append(circle)
    return results


friendCircles = extractFriendCircles(G)

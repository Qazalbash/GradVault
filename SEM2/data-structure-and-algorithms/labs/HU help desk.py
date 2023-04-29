def Enqueue(queue, item, priority):
    qdict = {}
    for i in queue:
        qdict[int(i[1])] = qdict.get(int(i[1]), []) + [i[0]]
    qdict[priority] = qdict.get(priority, []) + [item]
    queue.clear()
    for key in sorted(qdict.keys(), reverse = True):
        for value in (qdict[key]):
            queue.append((value, key))

def Dequeue(queue):
    return queue.pop(0)[0]

def size(queue):
    return len(queue)

queue = [
    ("AC Not working in Tariq Rafi",5),
    ("Password Change Issue", 4),
    ("Need Installation on laptop", 3),
    ("Need license", 1),
    ("Lab PCs Setup", 3),
    ("Login Issue", 4)
]

def issues(queue):
    pseudoq = []
    for i in queue:
        Enqueue(pseudoq,i[0], i[1])
    for i in range(size(queue)):
        print(Dequeue(pseudoq))
issues(queue)
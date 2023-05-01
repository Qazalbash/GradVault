def Enqueue(queue, item, priority):
    qdict = {}
    for i in queue:
        qdict[int(i[1])] = qdict.get(int(i[1]), []) + [i[0]]
    qdict[priority] = qdict.get(priority, []) + [item]
    queue.clear()
    for key in sorted(qdict.keys(), reverse=True):
        for value in sorted(qdict[key]):
            queue.append((value, key))


def Dequeue(queue):
    return queue.pop(0)[0]

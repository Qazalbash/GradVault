def sort(lst):
    for pointer in range(1, len(lst)):
        key, hand = lst[pointer], pointer - 1
        while hand >= 0 and key < lst[hand]:
            lst[hand + 1] = lst[hand]
            hand -= 1
        lst[hand + 1] = key
    return lst


def minimum_distance_pairs(lst):
    distance = {}
    minimum = round(
        ((lst[0][0] - lst[1][0])**2 + (lst[0][1] - lst[1][1])**2)**0.5, 3)
    for i in lst:
        for j in lst:
            dist = round(((i[0] - j[0])**2 + (i[1] - j[1])**2)**0.5, 3)
            try:
                if [i, j] in distance[dist] or [j, i] in distance[dist]:
                    continue
            except:
                pass
            distance[dist] = distance.get(dist, []) + [[i, j]]
            if minimum > dist and dist > 0:
                minimum = dist
    return distance[minimum]


lst = [(0, 0), (0, 2), (1, 1), (2, 0)]
print(minimum_distance_pairs(lst))

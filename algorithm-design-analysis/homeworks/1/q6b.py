def extract(IDs):
    mid = len(IDs) // 2
    first = IDs[:mid]
    second = IDs[mid:]

    unique = []
    if len(IDs) > 1:
        idsFirst = extract(first)
        idsSecond = extract(second)

        for i in idsFirst:
            if i not in unique:  #Assuming this condition is checked in constant time
                unique.append(i)
        for j in idsSecond:
            if j not in unique:  #Assuming this condition is checked in constant time
                unique.append(j)
    else:
        return IDs

    return unique

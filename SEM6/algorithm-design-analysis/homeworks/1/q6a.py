def extract2(IDs):
    mid = len(IDs) // 3
    first = IDs[:mid]
    second = IDs[mid:2 * mid]
    third = IDs[2 * mid:]

    unique = []
    if len(IDs) > 2:
        idsFirst = extract2(first)
        idsSecond = extract2(second)
        idsThird = extract2(third)

        for i in idsFirst:
            if i not in unique:  #Assuming this condition is checked in constant time
                unique.append(i)
        for j in idsSecond:
            if j not in unique:  #Assuming this condition is checked in constant time
                unique.append(j)
        for k in idsThird:
            if k not in unique:  #Assuming this condition is checked in constant time
                unique.append(k)

    else:
        if len(IDs) > 1:
            if IDs[0] == IDs[1]:
                return [IDs[0]]
            else:
                return IDs
        else:
            return IDs

    return unique

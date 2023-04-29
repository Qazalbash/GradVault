def milan(X, Y):
    maximum = 0
    M = []
    for i in range(len(X)):
        r = []
        for j in range(len(Y)):
            r.append(0)
        M.append(r)

    for i in range(len(X)):
        for j in range(len(Y)):
            if i == 0 or j == 0 or X[i] != Y[j]:
                M[i][j] = 0
            else:
                M[i][j] = M[i - 1][j - 1] + 1
            maximum = max(maximum, M[i][j])
    return maximum

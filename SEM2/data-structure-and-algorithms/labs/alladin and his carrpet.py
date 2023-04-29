def optimalPoint(magic, dist):
    if sum(magic) >= sum(dist):
        p, sigma, init = len(magic), 0, 0
        for i in range(p):
            if sigma < 0:
                init, sigma = i, 0
            sigma += magic[i] - dist[i]
        return init
    return -1
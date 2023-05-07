def Horner(A, n, x):
    # A contains the coefficients
    p = 0
    for i in range(n, -1, -1):
        p = A[i] + x * p
    return p

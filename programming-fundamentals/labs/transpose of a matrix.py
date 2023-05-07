def matrix_transpose(A):
    n = len(A)
    m = len(A[0])
    C = [[0 for n in range(n)] for m in range(m)]
    for row in range(m):
        for col in range(n):
            C[row][col] = A[col][row]
    return C
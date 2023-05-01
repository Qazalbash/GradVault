def matrix_addition(A, B):
    if (len(A), len(A[0])) != (len(B), len(B[0])):
        return "Matrices A and B don't have the same dimension required for matrix addition."
    m = len(A)
    n = len(A[0])
    C = [[0 for n in range(n)] for m in range(m)]
    for row in range(m):
        for col in range(n):
            C[row][col] = A[row][col] + B[row][col]
    return C


print(matrix_addition(A, B))

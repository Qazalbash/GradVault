def matrix_multiplication(A,B):
    if len(A[0]) != len(B):
        return "The number of columns in Matrix A does not equal the number of rows in Matrix B required for Matrix Multiplication."
    C = [[0 for m in range(len(B[0]))] for p in range(len(A))]
    p = len(A[0])
    for row in range(len(A)):
        for col in range(len(B[0])):
            C[row][col] = sum([A[row][n]*B[n][col] for n in range(p)])
    return C

print(matrix_multiplication(A,B))

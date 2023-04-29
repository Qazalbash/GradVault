def minorlist(A, r, c):
    Sub = []
    for row in range(r + 1):
        Sub += A[row][:c + 1]
    return Sub


def integral_image(A):
    B = [[0 for i in range(len(A[0]))] for j in range(len(A))]
    for row in range(len(A)):
        for col in range(len(A[0])):
            B[row][col] = sum(minorlist(A, row, col))
    return B


A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

print(integral_image(A))

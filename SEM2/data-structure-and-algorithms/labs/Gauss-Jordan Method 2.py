def matrix_multiplication(A, B):
    if len(A[0]) != len(B):
        return False
    C = [[0 for m in range(len(B[0]))] for p in range(len(A))]
    p = len(A[0])
    for row in range(len(A)):
        for col in range(len(B[0])):
            C[row][col] = sum([A[row][n] * B[n][col] for n in range(p)])
    return C


def support(arr, num, size):
    if size == 2:
        if num == 1:
            return [[1 / arr[0][0], 0], [-arr[1][0] / arr[0][0], 1]]
        elif num == 2:
            return [[1, -arr[0][1] / arr[1][1]], [0, 1 / arr[1][1]]]
    elif size == 3:
        if num == 1:
            return [[1 / arr[0][0], 0, 0], [-arr[1][0] / arr[0][0], 1, 0],
                    [-arr[2][0] / arr[0][0], 0, 1]]
        elif num == 2:
            return [[1, -arr[0][1] / arr[1][1], 0], [0, 1 / arr[1][1], 0],
                    [0, -arr[2][1] / arr[1][1], 1]]
        elif num == 3:
            return [[1, 0, -arr[0][2] / arr[2][2]],
                    [0, 1, -arr[1][2] / arr[2][2]], [0, 0, 1 / arr[2][2]]]
    elif size == 4:
        if num == 1:
            return [[1 / arr[0][0], 0, 0, 0], [-arr[1][0] / arr[0][0], 1, 0, 0],
                    [-arr[2][0] / arr[0][0], 0, 1, 0],
                    [-arr[3][0] / arr[0][0], 0, 0, 1]]
        elif num == 2:
            return [[1, -arr[0][1] / arr[1][1], 0, 0], [0, 1 / arr[1][1], 0, 0],
                    [0, -arr[2][1] / arr[1][1], 1, 0],
                    [0, -arr[3][1] / arr[1][1], 0, 1]]
        elif num == 3:
            return [[1, 0, -arr[0][2] / arr[2][2], 0],
                    [0, 1, -arr[1][2] / arr[2][2], 0], [0, 0, 1 / arr[2][2], 0],
                    [0, 0, -arr[3][2] / arr[2][2], 1]]
        elif num == 4:
            return [[1, 0, 0, -arr[0][3] / arr[3][3]],
                    [0, 1, 0, -arr[1][3] / arr[3][3]],
                    [0, 0, 1, -arr[2][3] / arr[3][3]], [0, 0, 0, 1 / arr[3][3]]]


def Solve(arr):
    if arr[0][0] == 0:
        arr = arr[::-1]
    for i in range(1, len(arr) + 1):
        arr = matrix_multiplication(support(arr, i, len(arr)), arr)
    return [round(b[-1], 2) for b in arr]


coeffs = [[0.3, -1, -0.3, 0, 0], [-1, -3, 4, 0, 20], [-4, 0, 0, 7, -20],
          [7, -2, -1, -4, 0]]
print(Solve(coeffs))

def blur_image(A):
    B = [[0 for j in range(len(A[0]))] for i in range(len(A))]
    for row in range(len(A)):
        for col in range(len(A[0])):
            count, average = 0, 0
            itters = [(row - 1, col), (row, col - 1), (row, col + 1),
                      (row + 1, col), (row, col)]
            for pointer in itters:
                if 0 <= pointer[0] < len(A) and 0 <= pointer[1] < len(A[0]):
                    average += A[pointer[0]][pointer[1]]
                    count += 1
            B[row][col] = round(average / count, 2)
    return B


print(blur_image(A))

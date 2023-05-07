import ast

A = input()
A = ast.literal_eval(A)


def blur_image(A):
    B = [[0 for j in range(len(A[0]))] for i in range(len(A))]
    for row in range(len(A)):
        for col in range(len(A[0])):
            count, average = 0, 0
            itters = [
                (row, col),
                (row - 1, col),
                (row, col - 1),
                (row, col + 1),
                (row + 1, col),
            ]
            for ele in itters:
                if 0 <= ele[0] < len(A) and 0 <= ele[1] < len(A[0]):
                    average += A[ele[0]][ele[1]]
                    count += 1
            B[row][col] = round(average / count, 2)
    return B


print(blur_image(A))

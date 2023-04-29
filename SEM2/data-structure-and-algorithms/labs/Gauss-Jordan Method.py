def Solve(arr):
    for col in range(len(arr[0])-1):
        for row in range(len(arr)):
            if arr[col][col] == 0:
                arr = arr[::-1]
            if row != col:
                arr[row] = [
                    -arr[col][r] * arr[row][col] / arr[col][col] + arr[row][r] for r in range(len(arr[0]))
                ]
                arr[row] = [
                    ele / arr[row][row] for ele in arr[row] if not(arr[row][row] == 0)
                ]
    return [round(r[-1],1) for r in arr]

coeffs = [
    [0.3,-1,-0.3,0,0],
    [-1,-3,4,0,20],
    [-4,0,0,7,-20],
    [7,-2,-1,-4,0]
]
print(Solve(coeffs))

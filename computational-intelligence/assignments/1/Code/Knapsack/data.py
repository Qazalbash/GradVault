file = "Datasets/f8_l-d_kp_23_10000"

openedFile = open(file, "r")
lines = openedFile.read().split("\n")

number_of_items = int(lines[0].split(" ")[0])
knapsack_capacity = int(lines[0].split(" ")[1])

del lines[0]

profits = []
weights = []

for i in range(number_of_items):
    line = lines[i].split(" ")
    profits.append(int(line[0]))
    weights.append(int(line[1]))

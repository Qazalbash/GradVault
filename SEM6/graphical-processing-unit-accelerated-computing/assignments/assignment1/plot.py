from matplotlib import pyplot as plt
import numpy as np

data = np.genfromtxt('output.csv', delimiter=',', skip_header=1)
data = data[data[:, 0].argsort()]
dataset_sizes = np.unique(data[:, 0])

for dataset_size in dataset_sizes:

    data_subset = data[data[:, 0] == dataset_size]
    grid_size = data_subset[:, 1]
    thread_size = data_subset[:, 2]
    gpu_time = data_subset[:, 4]
    cpu_time = data_subset[:, 3]

    plt.axhline(y=cpu_time[0], color='r', linestyle='-')
    plt.text(0, cpu_time[0], 'CPU time: ' + str(cpu_time[0]))
    plt.bar(range(3), gpu_time, width=0.5, align='center')
    plt.title('Dataset size: ' + str(dataset_size))
    plt.xlabel('<<<Grid size, Thread size>>>')
    plt.ylabel('GPU time')
    plt.xticks(range(3), zip(grid_size, thread_size))
    plt.show()
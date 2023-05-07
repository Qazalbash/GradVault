import matplotlib.pyplot as plt
import numpy as np


def img_to_png(path):
    fig = open(f"{path}.img", "rb")
    data = np.fromfile(fig, dtype=np.uint8)
    image = data.reshape((256, 256))
    plt.imshow(image, cmap="gray")
    plt.savefig(f"{path}.png")


output_file_cpu = "../content/lena_out"
output_file_nearest_neighbour = "../content/lena_nearest_neighbour"
output_file_bilinear = "../content/lena_bilinear"
output_file_bicubic = "../content/lena_bicubic"

img_to_png(output_file_cpu)
img_to_png(output_file_nearest_neighbour)
img_to_png(output_file_bilinear)
img_to_png(output_file_bicubic)
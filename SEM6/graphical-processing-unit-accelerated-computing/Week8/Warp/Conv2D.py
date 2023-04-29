import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import cv2 as cv
import wget
import os 

filename = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTjgu2Q2GakpG-bunlNjdcZkeYobs4O4SnoJQ&usqp=CAU"
# Make sure path is opened as root directory of this repository: \CUDA_Spring2023
if not os.path.exists("Week8/Warp/img.png"):
    img = wget.download(filename, out="Week8/Warp/img.png")

I = cv.cvtColor(cv.imread("Week8/Warp/img.png"), cv.COLOR_BGR2GRAY)

wp.init()

kernel = np.array([[1/16,1/8,1/16], [1/8,1/4,1/8], [1/16,1/8,1/16]], dtype=float)

WIDTH = wp.constant(I.shape[0])
HEIGHT = wp.constant(I.shape[1])

a = I
c = np.zeros(shape=(WIDTH.val, HEIGHT.val), dtype=float)

@wp.kernel
def Conv2D(a: wp.array2d(dtype=float), c: wp.array2d(dtype=float), kernel: wp.array2d(dtype=float)):
    i,j = wp.tid()

    Pvalue = float(0)
    for x in range(-1, 2):
        for y in range(-1, 2):
            if i+x >= 0 and i+x < WIDTH and j+y >= 0 and j+y < HEIGHT:
                Pvalue = Pvalue + (a[i+x, j+y] * kernel[x+1,y+1])
               
    c[i,j] = Pvalue

gpu_a = wp.from_numpy(a, dtype=float)
gpu_c = wp.from_numpy(c, dtype=float)
gpu_kernel = wp.from_numpy(kernel, dtype=float)

with wp.ScopedTimer(""):
    wp.launch(kernel=Conv2D,
            dim=(WIDTH.val,HEIGHT.val),
            inputs=[gpu_a, gpu_c, gpu_kernel])

c = np.array(gpu_c.to("cpu"))

f, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(I,cmap='gray')
ax2.imshow(np.array(c, dtype=np.uint8),cmap='gray')
plt.show()


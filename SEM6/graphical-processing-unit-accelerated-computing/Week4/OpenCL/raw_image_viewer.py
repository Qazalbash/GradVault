import numpy as np
import cv2

# lets view our image
fd = open('BaboonFlipped.raw', 'rb')
rows = 512
cols = 512
f = np.fromfile(fd, dtype=np.uint8, count=rows * cols)
im = f.reshape((rows, cols))  #notice row, column format
fd.close()

#display image
cv2.imshow('image', im)
cv2.waitKey(0)
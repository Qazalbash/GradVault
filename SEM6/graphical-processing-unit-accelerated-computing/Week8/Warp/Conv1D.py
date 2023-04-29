import warp as wp
import numpy as np
import time

wp.init()

kernel = wp.array([3,4,5,4,3], dtype=float, device="cuda")

N = wp.constant(16)
a = np.zeros(shape=N.val, dtype=float)
c = np.zeros(shape=N.val, dtype=float)

for i in range(N.val):
    a[i] = i+1

@wp.kernel
def SumKernel(a: wp.array(dtype=float), c: wp.array(dtype=float), kernel: wp.array(dtype=float)):
    i = wp.tid()

    Pvalue = float(0)
    N_start_point = i-(5//2)

    for j in range(0, 5):
        if N_start_point+j >= 0 and N_start_point+j < N:
            Pvalue += a[N_start_point+j] * kernel[j]
    c[i] = Pvalue

gpu_a = wp.from_numpy(a, dtype=float)
gpu_c = wp.from_numpy(c, dtype=float)

with wp.ScopedTimer(""):
    wp.launch(kernel=SumKernel,
            dim=N.val,
            inputs=[gpu_a, gpu_c, kernel])

c = np.array(gpu_c.to("cpu"))

print(c)


'''
Warp does not implement shared memory or synhronization of threads,
so it dosen't make sense to do tiled convolutions, and reductions
are impossible. However, registering external C++ headers directly 
from Warp is on their to-do list!
'''

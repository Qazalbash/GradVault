% %
    cu
#include <stdio.h>
    const int N               = 33 * 1024;
const int     threadsPerBlock = 256;

#define imin(a, b) (a < b ? a : b)

const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

inline cudaError_t checkCudaErr(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
    }
    return err;
}

__global__ void dot(float *a, float *b, float *c) {
    __shared__ float cache[threadsPerBlock];
    int              tid        = threadIdx.x + blockIdx.x * blockDim.x;
    int              cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) c[blockIdx.x] = cache[0];
}

int main() {
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    // allocate memory on the cpu side
    a         = (float *)malloc(N * sizeof(float));
    b         = (float *)malloc(N * sizeof(float));
    partial_c = (float *)malloc(blocksPerGrid * sizeof(float));

    // allocate the memory on the GPU
    checkCudaErr(cudaMalloc((void **)&dev_a, N * sizeof(float)), "cudaMalloc1");
    checkCudaErr(cudaMalloc((void **)&dev_b, N * sizeof(float)), "cudaMalloc2");
    checkCudaErr(cudaMalloc((void **)&dev_partial_c, blocksPerGrid * sizeof(float)), "cudaMalloc3");

    // fill in the host memory with data
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // copy the arrays 'a' and 'b' to the GPU
    checkCudaErr(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemCpy1");
    checkCudaErr(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemCpy2");

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    // copy the array 'c' back from the GPU to the CPU
    checkCudaErr(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemCpy3");

    // finish up on the CPU side

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));

    // free memory on the gpu side
    checkCudaErr(cudaFree(dev_a), "cudaFree1");
    checkCudaErr(cudaFree(dev_b), "cudaFree2");
    checkCudaErr(cudaFree(dev_partial_c), "cudaFree3");

    // free memory on the cpu side
    free(a);
    free(b);
    free(partial_c);

    cudaDeviceReset();
    return 0;
}
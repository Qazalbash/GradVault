#include <stdio.h>
#include <stdlib.h>

inline cudaError_t checkCudaErr(cudaError_t err, const char *msg) {
        if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
        }
        return err;
}

// kernel function definition
__global__ void add(int *a, int *b, int *c) { *c = *a + *b; }

int main() {
        int h_a = 10;
        int h_b = 20;
        int h_c = 10 + 20;

        int *d_a, *d_b, *d_c;

        // allocate memory on device
        cudaMalloc((void **)&d_a, sizeof(int));
        cudaMalloc((void **)&d_b, sizeof(int));
        cudaMalloc((void **)&d_c, sizeof(int));

        // copy host data to device memory
        cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);

        // call kernel
        add<<<1, 1>>>(d_a, d_b, d_c);

        checkCudaErr(cudaDeviceSynchronize(), "Syncronization");
        checkCudaErr(cudaGetLastError(), "GPU");
        printf("Answer (on host): %d\n", h_c);

        // copy device data to host memory
        cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
        printf("Answer (on device): %d", h_c);

        // release GPU memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        return 0;
}
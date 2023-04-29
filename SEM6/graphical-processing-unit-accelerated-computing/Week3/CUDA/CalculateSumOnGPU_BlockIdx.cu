#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__global__ void sum(int* a, int* b, int* c) {
        int i = blockIdx.x;
        c[i]  = a[i] + b[i];
}

void sum_host(int* a, int* b, int* c, const int N) {
        for (int i = 0; i < N; ++i) c[i] = a[i] + b[i];
}

int main() {
        int* h_a = 0;
        int* h_b = 0;
        int* h_c = 0;

        int* d_a = 0;
        int* d_b = 0;
        int* d_c = 0;

        const int N    = 128;
        size_t    size = N * sizeof(int);

        // allocate host memory
        h_a = (int*)malloc(size);
        h_b = (int*)malloc(size);
        h_c = (int*)malloc(size);

        // initialize a, b and c
        for (int i = 0; i < N; ++i) {
                h_a[i] = i + 1;
                h_b[i] = h_a[i] * 2;
                h_c[i] = 0;
        }

        // allocate device memory
        cudaMalloc((void**)&d_a, size);
        cudaMalloc((void**)&d_b, size);
        cudaMalloc((void**)&d_c, size);

        // copy host data to device memory
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

        // calculate on host
        sum_host(h_a, h_b, h_c, N);

        // output result
        printf("Host calculation result: \n");
        for (int i = 0; i < N; ++i) {
                printf("%3d + %3d = %3d\n", h_a[i], h_b[i], h_c[i]);
                // clear host result to ensure that the result of device is actually
                // from the kernel
                h_c[i] = 0;
        }

        // calculate on device
        sum<<<N, 1>>>(d_a, d_b, d_c);

        // copy result from device to host
        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

        // output result
        printf("--------------------------------------\n");
        printf("Device calculation result: \n");
        for (int i = 0; i < N; ++i) {
                printf("%3d + %3d = %3d\n", h_a[i], h_b[i], h_c[i]);
        }
        printf("--------------------------------------\n");

        // delete data allocated on device
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        // delete host memory
        free(h_a);
        free(h_b);
        free(h_c);

        cudaDeviceReset();
        return 0;
}
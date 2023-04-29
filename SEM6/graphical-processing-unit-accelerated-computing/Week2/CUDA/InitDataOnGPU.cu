#include <cuda_runtime.h>
#include <stdio.h>

inline cudaError_t checkCudaErr(cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
        }
        return err;
}

__global__ void InitData(int* data) { data[threadIdx.x] += threadIdx.x; }

int main() {
        const int N = 100;

        int* h_data = (int*)malloc(N * sizeof(int));
        int* d_data;

        // allocate memory on device
        cudaMalloc((void**)&d_data, N * sizeof(int));

        // call kernel
        InitData<<<1, N>>>(d_data);

        checkCudaErr(cudaDeviceSynchronize(), "Syncronization");
        checkCudaErr(cudaGetLastError(), "GPU Error");

        // copy device data to host memory
        checkCudaErr(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy D->H");

        printf("Data (on device): \n");
        for (int i = 0; i < N; ++i) printf(" data[%d] -> %d\n", i, h_data[i]);

        // release GPU memory
        cudaFree(d_data);
        free(h_data);

        return 0;
}
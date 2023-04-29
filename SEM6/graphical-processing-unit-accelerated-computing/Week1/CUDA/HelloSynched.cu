#include <stdio.h>

__global__ void HelloKernel() { printf("\tHello from GPU (device)\n"); }

int main() {
        printf("Hello from CPU (host) before kernel execution\n");

        HelloKernel<<<1, 32>>>();  // this is an async call

        // this function ensures that the CPU execution
        // waits until the GPU finishes its job
        cudaDeviceSynchronize();

        printf("Hello from CPU (host) after kernel execution\n");

        cudaDeviceReset();

        return 0;
}
#include <stdio.h>

__global__ void HelloKernel() { printf("\tHello from GPU (device)\n"); }

int main() {
        printf("Hello from CPU (host) before kernel execution\n");
        HelloKernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        printf("Hello from CPU (host) after kernel execution\n");
        return 0;
}
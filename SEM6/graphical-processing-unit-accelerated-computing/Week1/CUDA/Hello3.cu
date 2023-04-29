#include <stdio.h>

__global__ void HelloKernel() { printf("\tHello from GPU (device)\n"); }

int main() {
        printf("Hello from CPU (host) before kernel execution\n");

        // this is an async call so control returns to CPU immediately
        HelloKernel<<<1, 32>>>();

        printf("Hello from CPU (host) after kernel execution\n");
        cudaDeviceReset();
        return 0;
}
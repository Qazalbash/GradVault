#include <cuda_runtime.h>
#include <stdio.h>

// shared memory size
#define SHMEM_SIZE 250

__global__ void sumReduction(int *v, int *v_r) {
        // Allocate shared memory
        __shared__ int partial_sum[SHARED_MEMORY_SIZE];

        // Calculate thread ID
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Load elements into shared memory

        partial_sum[threadIdx.x] = v[tid];

        __syncthreads();

        // Increase the stride of the access until we exceed the CTA dimensions
        for (int s = 1; s < blockDim.x; s *= 2) {
                // Change the indexing to be sequential threads
                int index = 2 * s * threadIdx.x;

                // Each thread does work unless the index goes off the block
                if (index < blockDim.x) {
                        partial_sum[index] += partial_sum[index + s];
                }

                __syncthreads();
        }

        // Let the thread 0 for this block write its result to main memory
        // Result is indexed by this block
        if (threadIdx.x == 0) {
                v_r[blockIdx.x] = partial_sum[0];
        }
}

int main() {
        // Vector size
        int  N                 = 10000;
        int *initial_host_data = 0;
        int *final_host_data   = 0;

        int size          = N * sizeof(int);
        initial_host_data = (int *)malloc(N * sizeof(int));
        final_host_data   = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) {
                initial_host_data[i] = i + 1;
                final_host_data[i]   = 0;
        }
        int sum = 0;
        for (int i = 0; i < N; i++) {
                sum += initial_host_data[i];
        }
        printf("sum on host %d\n", sum);
        int *initial_device_data, *final_device_data;
        cudaMalloc((void **)&initial_device_data, size);
        cudaMalloc((void **)&final_device_data, size);

        // Copy to device hv to dv
        cudaMemcpy(initial_device_data, initial_host_data, size, cudaMemcpyHostToDevice);
        cudaMemcpy(final_device_data, final_host_data, size, cudaMemcpyHostToDevice);

        // TB Size
        const int TB_SIZE = 250;

        // Grid Size (No padding)
        int GRID_SIZE = N / TB_SIZE;

        // Call kernels
        sumReduction<<<GRID_SIZE, TB_SIZE>>>(initial_device_data, final_device_data);

        sumReduction<<<1, TB_SIZE>>>(final_device_data, final_device_data);

        // Copy to host;
        cudaMemcpy(final_host_data, final_device_data, size, cudaMemcpyDeviceToHost);

        printf("result = %d", final_host_data[0]);

        return 0;
}
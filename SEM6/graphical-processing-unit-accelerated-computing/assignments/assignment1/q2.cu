#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

#define SHARED_MEMORY_SIZE 250

/**
 * @brief Checks for CUDA errors and prints the error message
 *
 * @param err
 * @param msg
 * @return cudaError_t
 */
inline cudaError_t checkCudaErr(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
    }
    return err;
}

/**
 * @brief Calculates the sum of an array using a reduction algorithm
 *
 * @param vector The input vector
 * @param reduced_vector The output vector
 */
__global__ void sumThroughReduction(const int *vector, int *reduced_vector)
{
    __shared__ int partial_sum[SHARED_MEMORY_SIZE]; // Shared memory
    int id = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID
    partial_sum[threadIdx.x] = vector[id];          // Load data into shared memory

    __syncthreads(); // Wait for all threads to load data into shared memory
                     // before starting the reduction

    // Reduction in shared memory (sequential addressing)
    int index;
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        // Each thread does work unless the index goes off the block (s is the
        // stride)
        index = 2 * s * threadIdx.x; // Index of the element to be added

        // Add the element at index to the element at index + s
        if (index < blockDim.x)
            partial_sum[index] += partial_sum[index + s];

        __syncthreads(); // Wait for all threads to finish before starting the
                         // next iteration
    }

    // Write the result for this block to the output vector at the correct index
    if (threadIdx.x == 0)
        reduced_vector[blockIdx.x] = partial_sum[0];
}

/**
 * @brief The main function of the program
 *
 * @return int
 */
int main(void)
{
    float cpu_time_used, gpu_time_used;
    clock_t start, end;
    cudaEvent_t start_gpu, stop_gpu;

    int ARRAY_SIZES[3] = {10000, 100000, 1000000};

    FILE *fp = fopen("output.csv", "w");
    fprintf(fp, "Dataset size,Grid size,Thread size,CPU time,GPU time\n");

    for (int ARRAY_SIZE : ARRAY_SIZES)
    {
        // Size of the array in bytes
        int size = ARRAY_SIZE * sizeof(int);

        // Host data (initial and final) (CPU)
        int *initial_host_data = (int *)malloc(size);
        int *final_host_data = (int *)malloc(size);

        int i = 0;

        // Initialize the host data with some values (1,2,3,...) and set the
        // final host data to 0
        for (; i < ARRAY_SIZE; i++)
        {
            initial_host_data[i] = i + 1;
            final_host_data[i] = 0;
        }

        start = clock();
        int sum = 0;
        // Calculate the sum on the host (CPU)
        for (i = 0; i < ARRAY_SIZE;)
            sum += initial_host_data[i++];
        end = clock();
        cpu_time_used = ((float)(end - start)) / CLOCKS_PER_SEC;
        printf("CPU time used: %f seconds for sum = %d\n", cpu_time_used, sum);

        int GRIDS[] = {ARRAY_SIZE / 250, ARRAY_SIZE / 500, ARRAY_SIZE / 1000};
        int THREADS = 250;

        // Device data (initial and final) (GPU)
        int *initial_device_data, *final_device_data;
        checkCudaErr(cudaMalloc((void **)&initial_device_data, size), "cudaMalloc");
        checkCudaErr(cudaMalloc((void **)&final_device_data, size), "cudaMalloc");

        for (int i = 0; i < 3; i++)
        {
            // Copy data from host to device (CPU -> GPU)
            checkCudaErr(cudaMemcpy(initial_device_data, initial_host_data, size, cudaMemcpyHostToDevice),
                         "cudaMemcpy");
            checkCudaErr(cudaMemcpy(final_device_data, final_host_data, size, cudaMemcpyHostToDevice), "cudaMemcpy");

            // Create events to measure the time taken by the GPU
            checkCudaErr(cudaEventCreate(&start_gpu), "cudaEventCreate");
            checkCudaErr(cudaEventCreate(&stop_gpu), "cudaEventCreate");
            checkCudaErr(cudaEventRecord(start_gpu, 0), "cudaEventRecord");

            // Launch the kernel
            sumThroughReduction<<<GRIDS[i], THREADS>>>(initial_device_data, final_device_data);

            // Launch the kernel again to reduce the final
            // data to a single value (the sum)
            sumThroughReduction<<<1, THREADS>>>(final_device_data, final_device_data);

            // Copy data from device to host (GPU -> CPU)
            checkCudaErr(cudaMemcpy(final_host_data, final_device_data, size, cudaMemcpyDeviceToHost), "cudaMemcpy");

            checkCudaErr(cudaEventRecord(stop_gpu, 0), "cudaEventRecord");
            checkCudaErr(cudaEventSynchronize(stop_gpu), "cudaEventSynchronize");
            checkCudaErr(cudaEventElapsedTime(&gpu_time_used, start_gpu, stop_gpu), "cudaEventElapsedTime");

            // Print the result
            printf("Grid size: %d, Thread size: %d\n", GRIDS[i], THREADS);
            printf("GPU time used: %f seconds for sum = %d\n", gpu_time_used / 1000.0f, final_host_data[0]);

            fprintf(fp, "%d,%d,%d,%f,%f\n", ARRAY_SIZE, GRIDS[i], THREADS, cpu_time_used, gpu_time_used / 1000.0f);
        }

        // Free the memory
        free(initial_host_data);
        free(final_host_data);
        cudaFree(initial_device_data);
        cudaFree(final_device_data);
    }

    fclose(fp);

    return 0;
}
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000

/**
 * @brief Check for CUDA errors
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
 * @brief Initialize the random number generator
 *
 * @param state
 * @param seed
 */
__global__ void init_rand(curandState *state, unsigned long seed)
{
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(seed, id, 0, &state[id]);
}

/**
 * @brief Generate random numbers on GPU
 *
 * @param state
 * @param rand
 */
__global__ void generate_rand(curandState *state, float *rand)
{
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        rand[id] = curand_uniform(&state[id]);
}

int main()
{
        float *rande, *d_rand, cpu_time_used, gpu_time_used, memcpy_time_used;
        int size = N * sizeof(float),             // CPU random number array size
            state_size = N * sizeof(curandState); // GPU random number array size
        curandState *state, *d_state;             // GPU random number generator state array
        clock_t start, end;                       // CPU timers
        cudaEvent_t start_gpu, stop_gpu;          // GPU timers

        rande = (float *)malloc(size);
        state = (curandState *)malloc(state_size);

        checkCudaErr(cudaMalloc((void **)&d_rand, size), "cudaMalloc");
        checkCudaErr(cudaMalloc((void **)&d_state, state_size), "cudaMalloc");

        start = clock();
        for (int i = 0; i < N; i++)
                rande[i] = (float)rand() / (float)RAND_MAX;
        end = clock();
        cpu_time_used = ((float)(end - start)) / CLOCKS_PER_SEC;
        printf("CPU time used:    %10f seconds to generate random numbers on CPU \n", cpu_time_used);

        checkCudaErr(cudaEventCreate(&start_gpu), "cudaEventCreate");
        checkCudaErr(cudaEventCreate(&stop_gpu), "cudaEventCreate");
        checkCudaErr(cudaEventRecord(start_gpu, 0), "cudaEventRecord");

        init_rand<<<N / 256, 256>>>(d_state, time(NULL));
        generate_rand<<<N / 256, 256>>>(d_state, d_rand);

        checkCudaErr(cudaEventRecord(stop_gpu, 0), "cudaEventRecord");
        checkCudaErr(cudaEventSynchronize(stop_gpu), "cudaEventSynchronize");

        checkCudaErr(cudaEventElapsedTime(&gpu_time_used, start_gpu, stop_gpu), "cudaEventElapsedTime");

        printf("GPU time used:    %10f seconds to generate random numbers on GPU \n", gpu_time_used / 1000.0f);

        start = clock();

        checkCudaErr(cudaMemcpy(rande, d_rand, size, cudaMemcpyDeviceToHost), "cudaMemcpy");
        end = clock();
        memcpy_time_used = ((float)(end - start)) / CLOCKS_PER_SEC;
        printf("MEMCPY time used: %10f seconds to copy random numbers from GPU to "
               "CPU\n",
               memcpy_time_used / 1000.0f);

        checkCudaErr(cudaFree(d_rand), "cudaFree");
        checkCudaErr(cudaFree(d_state), "cudaFree");

        free(rande);
        free(state);

        return 0;
}
% %
    cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

    inline cudaError_t
    checkCudaErr(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
    }
    return err;
}

// kernel function to calculate sum on the GPU
__global__ void SumKernel(int* in_array, int* out_array, const int N) {
    unsigned int tid   = threadIdx.x;
    unsigned int idx   = blockIdx.x * blockDim.x + tid;
    int*         idata = in_array + blockIdx.x * blockDim.x;

    if (idx >= N) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) out_array[blockIdx.x] = idata[0];
}

// Neighbored Pair Implementation with divergence
__global__ void reduceNeighbored(int* g_idata, int* g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Neighbored Pair Implementation with less divergence
__global__ void reduceNeighboredLess(int* g_idata, int* g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index
        int index = 2 * stride * tid;

        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved(int* g_idata, int* g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// function to calculate sum on the CPU
void SumHost(int* array, int* sum, const int N) {
    *sum = 0;
    for (int i = 0; i < N; ++i) {
        *sum += array[i];
    }
}

int ReductionSumRecursiveInterleaved(int* array, const int size) {
    if (size == 1) return array[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; ++i) {
        array[i] += array[i + stride];
        array[i + stride] = 0;
    }

    if (stride * 2 < size) {
        array[0] += array[stride * 2];
        array[stride * 2] = 0;
    }
    return ReductionSumRecursiveInterleaved(array, stride);
}

int ReductionSumRecursiveNeighbored(int* array, const int size, int stride) {
    if (stride > size) return array[0];

    int i = 0;
    while ((i + stride) < size) {
        array[i] += array[i + stride];
        array[i + stride] = 0;
        i += stride * 2;
    }

    return ReductionSumRecursiveNeighbored(array, size, stride * 2);
}

void ReductionSumInterleavedHost(int* array, int* sum, const int N) {
    *sum = 0;

    int stride = N / 2;

    while (stride > 0) {
        for (int j = 0; j < stride; ++j) {
            array[j] += array[j + stride];
            array[j + stride] = 0;
        }

        if (stride * 2 < N) {
            array[0] += array[stride * 2];
            array[stride * 2] = 0;
        }
        stride /= 2;
    }
    *sum = array[0];
}

void ReductionSumNeighboredHost(int* array, int* sum, const int N) {
    *sum       = 0;
    int stride = 1;

    while (stride < N) {
        int i = 0;
        while ((i + stride) < N) {
            array[i] += array[i + stride];
            array[i + stride] = 0;
            i += stride * 2;
        }
        stride *= 2;
    }
    *sum = array[0];
}

int main() {
    const int N             = 1000;
    const int SIZE_IN_BYTES = N * sizeof(int);

    int* h_array  = (int*)malloc(SIZE_IN_BYTES);
    int* h_darray = (int*)malloc(SIZE_IN_BYTES);
    int  h_sum    = 0;
    int  h_dsum   = 0;  // device calc sum to be read on host

    // Initialize array on the host
    for (int i = 0; i < N; i++) {
        h_array[i] = i + 1;
    }

    int* d_array;
    int* d_oarray;
    cudaMalloc(&d_array, SIZE_IN_BYTES);
    cudaMalloc(&d_oarray, SIZE_IN_BYTES);

    cudaMemcpy(d_array, h_array, SIZE_IN_BYTES, cudaMemcpyHostToDevice);

    int   numThreadsPerBlock = 8;
    int   numBlocksPerGrid   = ceilf(N / numThreadsPerBlock + 1);
    float gpu_elapsed_time_ms, cpu_elapsed_time_ms, cpu_time_reduction_i_ms, cpu_time_reduction_n_ms,
        cpu_time_reduction_ir_ms, cpu_time_reduction_nr_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);

    SumKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_array, d_oarray, N);

    cudaMemcpy(h_darray, d_oarray, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

    // run a loop for number of blocks and sum on CPU
    for (int i = 0; i < numBlocksPerGrid; ++i) {
        h_dsum += h_darray[i];
    }
    printf("\n");

    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Sum (GPU): %d\t\t\t\t\tTime elapsed (GPU): %f ms.\n", h_dsum, gpu_elapsed_time_ms);

    ///////////////////////////
    // reducedNeighbored GPU //
    ///////////////////////////////////////////////////////////////////////////
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);

    reduceNeighbored<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_array, d_oarray, N);
    cudaMemcpy(h_darray, d_oarray, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("reducedNeighbored (GPU): %d\t\t\t\tTime elapsed (GPU): %f ms.\n", h_dsum, gpu_elapsed_time_ms);

    ///////////////////////////////
    // reducedNeighboredLess GPU //
    ///////////////////////////////////////////////////////////////////////////
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);

    reduceNeighboredLess<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_array, d_oarray, N);
    cudaMemcpy(h_darray, d_oarray, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("reducedNeighboredLess (GPU): %d\t\t\tTime elapsed (GPU): %f ms.\n", h_dsum, gpu_elapsed_time_ms);

    ////////////////////////////
    // reducedInterleaved GPU //
    ///////////////////////////////////////////////////////////////////////////
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);

    reduceInterleaved<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_array, d_oarray, N);
    cudaMemcpy(h_darray, d_oarray, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("reducedInterleaved(GPU): %d\t\t\t\tTime elapsed (GPU): %f ms.\n", h_dsum, gpu_elapsed_time_ms);

    ///////////////////////////
    // start the CPU version //
    ///////////////////////////////////////////////////////////////////////////
    cudaEventRecord(start, 0);
    SumHost(h_array, &h_sum, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Sum (CPU): %d\t\t\t\t\tTime elapsed (CPU): %f ms.\n", h_sum, cpu_elapsed_time_ms);

    ///////////////////////////////////////////////////
    // start the ReductionSumInterleavedHost version //
    ///////////////////////////////////////////////////////////////////////////
    cudaEventRecord(start, 0);
    ReductionSumInterleavedHost(h_array, &h_sum, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time_reduction_i_ms, start, stop);
    printf("Reduction Sum Iterative Interleaved (CPU): %d\tTime elapsed (CPU): %f ms.\n", h_sum,
           cpu_time_reduction_i_ms);

    //////////////////////////////////////////////////
    // start the ReductionSumNeighboredHost version //
    ///////////////////////////////////////////////////////////////////////////
    cudaEventRecord(start, 0);
    ReductionSumNeighboredHost(h_array, &h_sum, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time_reduction_n_ms, start, stop);
    printf("Reduction Sum Iterative Neighbored (CPU): %d\tTime elapsed (CPU): %f ms.\n", h_sum,
           cpu_time_reduction_n_ms);

    ///////////////////////////////////////////////////////
    // start the ReductionSumRecursiveNeighbored version //
    ///////////////////////////////////////////////////////////////////////////
    cudaEventRecord(start, 0);
    h_sum = ReductionSumRecursiveNeighbored(h_array, N, 1);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time_reduction_nr_ms, start, stop);
    printf("Reduction Sum Recursive Neighbored (CPU): %d\tTime elapsed (CPU): %f ms.\n", h_sum,
           cpu_time_reduction_nr_ms);

    ////////////////////////////////////////////////////////
    // start the ReductionSumRecursiveInterleaved version //
    ///////////////////////////////////////////////////////////////////////////
    cudaEventRecord(start, 0);
    h_sum = ReductionSumRecursiveInterleaved(h_array, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time_reduction_ir_ms, start, stop);
    printf("Reduction Sum Recursive Interleaved (CPU): %d\tTime elapsed (CPU): %f ms.\n", h_sum,
           cpu_time_reduction_ir_ms);

    //////////////////////
    // validate results //////////////////
    // validate results computed by GPU //
    ///////////////////////////////////////////////////////////////////////////
    int all_ok = (h_sum == h_dsum) ? 1 : 0;

    // roughly compute speedup
    if (all_ok) {
        printf("All results are correct!!!, speedup (CPU/GPU) = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    } else {
        printf("Incorrect results\n");
    }

    cudaFree(d_array);
    cudaFree(d_oarray);
    free(h_array);

    return 0;
}
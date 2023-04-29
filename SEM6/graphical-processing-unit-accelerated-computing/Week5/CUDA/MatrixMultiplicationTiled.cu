#include <cuda.h>
#include <stdio.h>

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width) {
        // Calculate the row index of the d_Pelement and d_M
        int Row = blockIdx.y * blockDim.y + threadIdx.y;
        // Calculate the column index of d_P and d_N
        int Col = blockIdx.x * blockDim.x + threadIdx.x;
        if ((Row < Width) && (Col < Width)) {
                float Pvalue = 0;
                // each thread computes one element of the block sub-matrix
                for (int k = 0; k < Width; ++k) Pvalue += d_M[Row * Width + k] * d_N[k * Width + Col];

                d_P[Row * Width + Col] = Pvalue;
        }
}

__global__ void MatrixMulKernelTiled(float* d_M, float* d_N, float* d_P, int Width) {
        const int        TILE_WIDTH = 16;
        __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
        int              bx = blockIdx.x;
        int              by = blockIdx.y;
        int              tx = threadIdx.x;
        int              ty = threadIdx.y;

        // Identify the row and column of the d_P element to work on
        int   Row    = by * TILE_WIDTH + ty;
        int   Col    = bx * TILE_WIDTH + tx;
        float Pvalue = 0;

        // Loop over the d_M and d_N tiles required to compute d_P element
        for (int m = 0; m < (TILE_WIDTH + Width - 1) / TILE_WIDTH; ++m) {
                if (m * TILE_WIDTH + tx < Width && Row < Width)
                        Mds[ty][tx] = d_M[Row * Width + m * TILE_WIDTH + tx];
                else
                        Mds[ty][tx] = 0.0;

                if (m * TILE_WIDTH + ty < Width && Col < Width)
                        Nds[ty][tx] = d_N[(m * TILE_WIDTH + ty) * Width + Col];
                else
                        Nds[ty][tx] = 0.0;

                __syncthreads();

                for (int k = 0; k < TILE_WIDTH; ++k) Pvalue += Mds[ty][k] * Nds[k][tx];

                __syncthreads();
        }
        if (Row < Width && Col < Width) d_P[Row * Width + Col] = Pvalue;
}

void MatrixMultHost(float* A, float* B, float* C, int N) {
        for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                        float Pvalue = 0;
                        for (int k = 0; k < N; ++k) Pvalue += A[i * N + k] * B[k * N + j];
                        C[j + i * N] = Pvalue;
                }
        }
}

int main(int argc, char** argv) {
        const int N             = 1000;
        const int SIZE          = N * N;
        const int SIZE_IN_BYTES = SIZE * sizeof(float);

        float* h_A   = (float*)malloc(SIZE_IN_BYTES);
        float* h_B   = (float*)malloc(SIZE_IN_BYTES);
        float* h_C   = (float*)malloc(SIZE_IN_BYTES);
        float* h_CD  = (float*)malloc(SIZE_IN_BYTES);  // device calc res
        float* h_CDT = (float*)malloc(SIZE_IN_BYTES);  // device calc res

        // Initialize matrices on the host
        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        h_A[i * N + j] = (float)(rand() % 1024);
                        h_B[i * N + j] = (float)(rand() % 1024);
                }
        }

        float* d_A;
        float* d_B;
        float* d_C;

        cudaMalloc(&d_A, SIZE_IN_BYTES);
        cudaMalloc(&d_B, SIZE_IN_BYTES);
        cudaMalloc(&d_C, SIZE_IN_BYTES);

        cudaMemcpy(d_A, h_A, SIZE_IN_BYTES, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, SIZE_IN_BYTES, cudaMemcpyHostToDevice);

        dim3 blocksGrid;
        dim3 threadsBlock(16, 16, 1);

        blocksGrid.x = (N + threadsBlock.x - 1) / threadsBlock.x;
        blocksGrid.y = (N + threadsBlock.y - 1) / threadsBlock.y;

        float gpu_elapsed_time_ms, cpu_elapsed_time_ms, gpu_elapsed_time_tiled_ms;

        // some events to count the execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // start to count execution time of GPU version
        cudaEventRecord(start, 0);

        MatrixMulKernel<<<blocksGrid, threadsBlock>>>(d_A, d_B, d_C, N);

        cudaMemcpy(h_CD, d_C, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

        // time counting terminate
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // compute time elapse on GPU computing
        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
        printf("Time elapsed (GPU): %f ms.\n", gpu_elapsed_time_ms);

        // now try the tiled matrix mult kernel
        //  start to count execution time of GPU version
        cudaEventRecord(start, 0);

        MatrixMulKernelTiled<<<blocksGrid, threadsBlock>>>(d_A, d_B, d_C, N);

        cudaMemcpy(h_CDT, d_C, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

        // time counting terminate
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // compute time elapse on GPU computing
        cudaEventElapsedTime(&gpu_elapsed_time_tiled_ms, start, stop);
        printf("Time elapsed (GPU Tiled): %f ms.\n", gpu_elapsed_time_tiled_ms);

        // start the CPU version
        cudaEventRecord(start, 0);
        MatrixMultHost(h_A, h_B, h_C, N);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
        printf("Time elapsed (CPU): %f ms.\n", cpu_elapsed_time_ms);

        // validate results
        //  validate results computed by GPU
        int all_ok = 1;
        for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                        if (h_C[j * N + i] != h_CD[j * N + i]) {
                                all_ok = 0;
                        }
                }
        }

        // roughly compute speedup
        if (all_ok)
                printf("All results are correct!!! (CPU vs GPU)\n");
        else
                printf("incorrect results\n");

        // validate results
        //  validate results computed by GPU Tiled
        all_ok = 1;
        for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                        if (h_C[j * N + i] != h_CDT[j * N + i]) all_ok = 0;

        // roughly compute speedup
        if (all_ok)
                printf("All results are correct!!! (CPU vs GPU Tiled)\n");
        else
                printf("incorrect results (CPU vs GPU Tiled)\n");

        printf("Speedup: GPU (Tiled) vs GPU (Untiled): %3.3f\n", gpu_elapsed_time_ms / gpu_elapsed_time_tiled_ms);

        free(h_A);
        free(h_B);
        free(h_C);
        free(h_CD);

        cudaFree(&d_A);
        cudaFree(&d_B);
        cudaFree(&d_C);

        cudaDeviceReset();
        return 0;
}
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI ((double)3.14159265358979323846264338327950288419716939937510)

const long LENGTH = 256L;
const double ROTATION = 15 * PI / 180.0;
const long REPEAT = 24L;

__constant__ double c;
__constant__ double s;
__constant__ double dx;
__constant__ double dy;
__constant__ long L;

/**
 * @brief Check CUDA error
 *
 * @param err
 * @param msg
 * @return cudaError_t
 */
inline cudaError_t checkCudaErr(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
        fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
    return err;
}

/**
 * @brief Rotate image using nearest neighbour interpolation
 *
 * @param src
 * @param dst
 * @return __global__
 */
__global__ void rotate_nearest_neighbour(const uint8_t *src, uint8_t *dst)
{
    const long x = threadIdx.x + blockIdx.x * blockDim.x; // column
    const long y = threadIdx.y + blockIdx.y * blockDim.y; // row
    const long tx = threadIdx.x;
    const long ty = threadIdx.y;

    __shared__ uint8_t shared_img[16 * 16]; // shared memory

    if (x < L && y < L)
    {
        const long Rxf = (long)(x * c - y * s + dx + 0.5); // column
        const long Ryf = (long)(x * s + y * c + dy + 0.5); // row

        shared_img[ty * 16 + tx] =
            (0L <= Rxf && Rxf < L && 0L <= Ryf && Ryf < L) ? src[Ryf * L + Rxf] : 0; // set pixel
    }

    __syncthreads(); // synchronize

    if (x < L && y < L)
        dst[y * L + x] = shared_img[ty * 16 + tx]; // copy pixel
}

/**
 * @brief Rotate image using bilinear interpolation
 *
 * @param src
 * @param dst
 * @return __global__
 */
__global__ void rotate_bilinear(const uint8_t *src, uint8_t *dst)
{
    const long x = threadIdx.x + blockIdx.x * blockDim.x; // column
    const long y = threadIdx.y + blockIdx.y * blockDim.y; // row
    const long tx = threadIdx.x;
    const long ty = threadIdx.y;

    __shared__ uint8_t shared_img[16 * 16]; // shared memory

    if (x < L && y < L)
    {
        const double Rx = x * c - y * s + dx; // column
        const double Ry = x * s + y * c + dy; // row

        const long Rxf = (long)floor(Rx + 0.5); // column
        const long Ryf = (long)floor(Ry + 0.5); // row

        const double dRx = Rx - Rxf; // column
        const double dRy = Ry - Ryf; // row

        const long index = Ryf * L + Rxf; // index

        shared_img[ty * 16 + tx] = (0L <= Rxf && Rxf < L && 0L <= Ryf && Ryf < L)
                                       ? (1.0 - dRx) * (1.0 - dRy) * src[index] + dRx * (1.0 - dRy) * src[index + 1L] +
                                             (1.0 - dRx) * dRy * src[index + L] + dRx * dRy * src[index + L + 1L]
                                       : 0; // set pixel
    }

    __syncthreads(); // synchronize

    if (x < L && y < L)
        dst[y * L + x] = shared_img[ty * 16 + tx]; // copy pixel
}

/**
 * @brief Bicubic interpolation
 *
 * @param c
 * @param x
 * @return double
 */
__device__ double interpolate(const double *const c, const double x)
{
    return c[1] + 0.5 * x *
                      (c[2] - c[0] +
                       x * (2.0 * c[0] - 5.0 * c[1] + 4.0 * c[2] - c[3] + x * (3.0 * (c[1] - c[2]) + c[3] - c[0])));
}

/**
 * @brief Bicubic interpolation using 4x4 matrix
 *
 * @param p
 * @param u
 * @param v
 * @return double
 */
__device__ double bicubic_interpolate(const double p[4][4], const double u, const double v)
{
    double px[4], py[4]; // temporary arrays
    for (int i = 0; i < 4; i++)
    {
        px[i] = interpolate(p[i], u);     // interpolate in x direction
        py[i] = interpolate(&p[0][i], v); // interpolate in y direction
    }
    return interpolate(px, v); // interpolate in x direction
}

/**
 * @brief Bicubic interpolation
 *
 * @param src
 * @param dst
 * @param w
 * @param h
 * @param theta
 */
__global__ void rotate_bicubic(const uint8_t *const src, uint8_t *dst)
{
    const long x = threadIdx.x + blockIdx.x * blockDim.x; // x value of the thread
    const long y = threadIdx.y + blockIdx.y * blockDim.y; // y value of the thread
    const long tx = threadIdx.x;                          // x value of the thread in the block
    const long ty = threadIdx.y;                          // y value of the thread in the block

    __shared__ double shared_img[16 * 16]; // shared memory

    if (x < L && y < L)
    {
        double p[4][4]; // 4x4 matrix

        const double dx = x - L / 2.0;        // x coordinate of the pixel in the rotated image
        const double dy = y - L / 2.0;        // y coordinate of the pixel in the rotated image
        const double Rx = x * c - y * s + dx; // x coordinate of the pixel in the original image
        const double Ry = x * s + y * c + dy; // y coordinate of the pixel in the original image
        if (0 <= Rx && Rx + 1 < L && 0 <= Ry && Ry + 1 < L)
        { // check if the pixel is inside the image boundaries

            const double Rxf = (long)floor(Rx + 0.5); // x coordinate of the pixel in the original image
            const double Ryf = (long)floor(Ry + 0.5); // y coordinate of the pixel in the original image

            const long dRx = Rx - Rxf; // distance between the pixel and the left pixel
            const long dRy = Ry - Ryf; // distance between the pixel and the top pixel

            for (uint8_t m = 0; m < 4; m++)
            {
                for (uint8_t n = 0; n < 4; n++)
                {
                    int xidx = Rxf + n - 1; // x coordinate of the pixel in the original image
                    int yidx = Ryf + m - 1; // y coordinate of the pixel in the original image

                    p[m][n] =
                        xidx >= 0 && xidx < L && yidx >= 0 && yidx < L ? src[yidx * L + xidx] : 0; // copy the pixel
                }
            }

            double interpolated_value = bicubic_interpolate(p, dRx, dRy); // interpolate the pixel

            shared_img[ty * 16 + tx] = interpolated_value > 255.0 || interpolated_value < 0.0
                                           ? 0
                                           : (uint8_t)round(interpolated_value); // copy the pixel
        }
        else
            shared_img[ty * 16 + tx] = 0; // copy the pixel

        __syncthreads(); // synchronize

        if (x < L && y < L)
            dst[y * L + x] = shared_img[ty * 16 + tx]; // copy the pixel
    }
}

/**
 * @brief Read image from file
 *
 * @param filename
 * @param img
 * @param w
 * @param h
 */
void read_image(const char filename[], uint8_t *img, const long w, const long h)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "cannot open %s for reading\n", filename);
        exit(1);
    }
    for (long i = 0L; i < w * h; i++)
    {
        unsigned char c;
        fread(&c, sizeof(unsigned char), 1, fp);
        img[i] = (double)c;
    }
    fclose(fp);
}

/**
 * @brief Write image to file
 *
 * @param filename
 * @param img
 * @param w
 * @param h
 */
void write_image(const char filename[], uint8_t *img, const long w, const long h)
{
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        fprintf(stderr, "cannot open %s for writing\n", filename);
        exit(1);
    }
    for (long i = 0L; i < w * h; i++)
    {
        unsigned char c = (unsigned char)img[i];
        fwrite(&c, sizeof(unsigned char), 1, fp);
    }
    fclose(fp);
}

/**
 * @brief Repeatedly rotate image using nearest neighbour interpolation
 *
 * @param h_dst
 * @param d_src
 * @param d_dst
 * @param numBlocks
 * @param threadsPerBlock
 */
void repeated_nearest_neighbour(uint8_t *h_dst, uint8_t *d_src, uint8_t *d_dst, const dim3 numBlocks,
                                const dim3 threadsPerBlock)
{
    float gpu_time = 0.0f;           // GPU time
    cudaEvent_t start_gpu, stop_gpu; // GPU timer

    for (long i = 0L; i < REPEAT; i++)
    {                               // repeat rotation
        float gpu_time_used = 0.0f; // GPU time used for each rotation

        checkCudaErr(cudaEventCreate(&start_gpu), "cudaEventCreate");   // create GPU timer
        checkCudaErr(cudaEventCreate(&stop_gpu), "cudaEventCreate");    // create GPU timer
        checkCudaErr(cudaEventRecord(start_gpu, 0), "cudaEventRecord"); // start GPU timer

        rotate_nearest_neighbour<<<numBlocks, threadsPerBlock>>>(d_src, d_dst); // rotate image
        checkCudaErr(cudaDeviceSynchronize(), "cudaDeviceSynchronize");         // synchronize

        checkCudaErr(cudaEventRecord(stop_gpu, 0), "cudaEventRecord");        // stop GPU timer
        checkCudaErr(cudaEventSynchronize(stop_gpu), "cudaEventSynchronize"); // synchronize
        checkCudaErr(cudaEventElapsedTime(&gpu_time_used, start_gpu, stop_gpu),
                     "cudaEventElapsedTime"); // get GPU time

        checkCudaErr(cudaMemcpy(h_dst, d_dst, sizeof(uint8_t) * LENGTH * LENGTH, cudaMemcpyDeviceToHost),
                     "cudaMemcpy"); // copy result to host
        checkCudaErr(cudaMemcpy(d_src, h_dst, sizeof(uint8_t) * LENGTH * LENGTH, cudaMemcpyHostToDevice),
                     "cudaMemcpy"); // copy result to device

        checkCudaErr(cudaEventDestroy(start_gpu), "cudaEventDestroy"); // destroy GPU timer
        checkCudaErr(cudaEventDestroy(stop_gpu), "cudaEventDestroy");  // destroy GPU timer

        gpu_time += gpu_time_used; // add GPU time
    }
    checkCudaErr(cudaMemcpy(h_dst, d_dst, sizeof(uint8_t) * LENGTH * LENGTH, cudaMemcpyDeviceToHost),
                 "cudaMemcpy"); // copy result to host

    printf("nearest neighbour: %f ms\n", gpu_time / REPEAT); // print GPU time
}

/**
 * @brief Repeatedly rotate image using bilinear interpolation
 *
 * @param h_dst
 * @param d_src
 * @param d_dst
 * @param numBlocks
 * @param threadsPerBlock
 */
void repeated_bilinear(uint8_t *h_dst, uint8_t *d_src, uint8_t *d_dst, const dim3 numBlocks,
                       const dim3 threadsPerBlock)
{
    float gpu_time;                  // GPU time
    cudaEvent_t start_gpu, stop_gpu; // GPU timer

    for (long i = 0L; i < REPEAT; i++)
    {                               // repeat rotation
        float gpu_time_used = 0.0f; // GPU time used for each rotation

        checkCudaErr(cudaEventCreate(&start_gpu), "cudaEventCreate");   // create GPU timer
        checkCudaErr(cudaEventCreate(&stop_gpu), "cudaEventCreate");    // create GPU timer
        checkCudaErr(cudaEventRecord(start_gpu, 0), "cudaEventRecord"); // start GPU timer

        rotate_bilinear<<<numBlocks, threadsPerBlock>>>(d_src, d_dst); // rotate image

        checkCudaErr(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // synchronize

        checkCudaErr(cudaEventRecord(stop_gpu, 0), "cudaEventRecord");        // stop GPU timer
        checkCudaErr(cudaEventSynchronize(stop_gpu), "cudaEventSynchronize"); // synchronize
        checkCudaErr(cudaEventElapsedTime(&gpu_time_used, start_gpu, stop_gpu),
                     "cudaEventElapsedTime"); // get GPU time

        checkCudaErr(cudaMemcpy(h_dst, d_dst, sizeof(uint8_t) * LENGTH * LENGTH, cudaMemcpyDeviceToHost),
                     "cudaMemcpy"); // copy result to host
        checkCudaErr(cudaMemcpy(d_src, h_dst, sizeof(uint8_t) * LENGTH * LENGTH, cudaMemcpyHostToDevice),
                     "cudaMemcpy"); // copy result to device

        checkCudaErr(cudaEventDestroy(start_gpu), "cudaEventDestroy"); // destroy GPU timer
        checkCudaErr(cudaEventDestroy(stop_gpu), "cudaEventDestroy");  // destroy GPU timer

        gpu_time += gpu_time_used; // add GPU time
    }
    checkCudaErr(cudaMemcpy(h_dst, d_dst, sizeof(uint8_t) * LENGTH * LENGTH, cudaMemcpyDeviceToHost),
                 "cudaMemcpy"); // copy result to host

    printf("bilinear: %f ms\n", gpu_time / REPEAT); // print GPU time
}

void repeated_bicubic(uint8_t *h_dst, uint8_t *d_src, uint8_t *d_dst, const dim3 numBlocks,
                      const dim3 threadsPerBlock)
{
    float gpu_time;                  // time used on the GPU for all iterations
    cudaEvent_t start_gpu, stop_gpu; // events for timing

    for (long i = 0L; i < REPEAT; i++)
    {                               // repeat the operation REPEAT times
        float gpu_time_used = 0.0f; // time used on the GPU for one iteration

        checkCudaErr(cudaEventCreate(&start_gpu), "cudaEventCreate");   // create the start event
        checkCudaErr(cudaEventCreate(&stop_gpu), "cudaEventCreate");    // create the stop event
        checkCudaErr(cudaEventRecord(start_gpu, 0), "cudaEventRecord"); // record the start event

        rotate_bicubic<<<numBlocks, threadsPerBlock>>>(d_src, d_dst); // call the kernel

        checkCudaErr(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // wait for the kernel to finish

        checkCudaErr(cudaEventRecord(stop_gpu, 0), "cudaEventRecord"); // record the stop event
        checkCudaErr(cudaEventSynchronize(stop_gpu),
                     "cudaEventSynchronize"); // wait for the stop event to be recorded
        checkCudaErr(cudaEventElapsedTime(&gpu_time_used, start_gpu, stop_gpu),
                     "cudaEventElapsedTime"); // compute the time used

        checkCudaErr(cudaMemcpy(h_dst, d_dst, sizeof(uint8_t) * LENGTH * LENGTH, cudaMemcpyDeviceToHost),
                     "cudaMemcpy"); // copy the result back to the host
        checkCudaErr(cudaMemcpy(d_src, h_dst, sizeof(uint8_t) * LENGTH * LENGTH, cudaMemcpyHostToDevice),
                     "cudaMemcpy"); // copy the result back to the device

        checkCudaErr(cudaEventDestroy(start_gpu), "cudaEventDestroy"); // destroy the events
        checkCudaErr(cudaEventDestroy(stop_gpu), "cudaEventDestroy");  // destroy the events

        gpu_time += gpu_time_used; // add the time used for this iteration to the total time
    }
    checkCudaErr(cudaMemcpy(h_dst, d_dst, sizeof(uint8_t) * LENGTH * LENGTH, cudaMemcpyDeviceToHost),
                 "cudaMemcpy");                    // copy the result back to the host
    printf("bicubic: %f ms\n", gpu_time / REPEAT); // print the average time used
}

/**
 * @brief Main function
 *
 * @return int
 */
int main()
{
    const long size = sizeof(uint8_t) * LENGTH * LENGTH; // size of image
    uint8_t *h_src = (uint8_t *)malloc(size);            // host source image
    uint8_t *h_dst = (uint8_t *)malloc(size);            // host destination image

    const double cos_theta = cos(ROTATION), sin_theta = sin(ROTATION); // rotation parameters
    const double co = LENGTH / 2.0;                                    // center of image
    const double delta_x = -co * cos_theta + co * sin_theta + co,
                 delta_y = -co * sin_theta - co * cos_theta + co; // translation parameters

    checkCudaErr(cudaMemcpyToSymbol(c, &cos_theta, sizeof(double)),
                 "cudaMemcpyToSymbol"); // copy rotation parameters to constant memory
    checkCudaErr(cudaMemcpyToSymbol(s, &sin_theta, sizeof(double)),
                 "cudaMemcpyToSymbol"); // copy rotation parameters to constant memory
    checkCudaErr(cudaMemcpyToSymbol(dx, &delta_x, sizeof(double)),
                 "cudaMemcpyToSymbol"); // copy translation parameters to constant memory
    checkCudaErr(cudaMemcpyToSymbol(dy, &delta_y, sizeof(double)),
                 "cudaMemcpyToSymbol"); // copy translation parameters to constant memory
    checkCudaErr(cudaMemcpyToSymbol(L, &LENGTH, sizeof(long)),
                 "cudaMemcpyToSymbol"); // copy image size to constant memory

    const dim3 numBlocks(16, 16);       // number of blocks
    const dim3 threadsPerBlock(16, 16); // number of threads per block

    // input files
    const char input_file[] = "../content/lena.img";

    // output files
    const char output_file_nearest_neighbour[] = "../content/lena_nearest_neighbour.img";
    const char output_file_bilinear[] = "../content/lena_bilinear.img";
    const char output_file_bicubic[] = "../content/lena_bicubic.img";

    read_image(input_file, h_src, LENGTH, LENGTH); // read image

    uint8_t *d_src, *d_dst; // device source and destination images

    checkCudaErr(cudaMalloc((void **)&d_src, size), "cudaMalloc");                      // allocate device memory
    checkCudaErr(cudaMalloc((void **)&d_dst, size), "cudaMalloc");                      // allocate device memory
    checkCudaErr(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice), "cudaMemcpy"); // copy image to device

    // nearest neighbour

    repeated_nearest_neighbour(h_dst, d_src, d_dst, numBlocks, threadsPerBlock); // rotate image

    write_image(output_file_nearest_neighbour, h_dst, LENGTH, LENGTH); // write image

    // bilinear

    checkCudaErr(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice), "cudaMemcpy"); // copy image to device

    repeated_bilinear(h_dst, d_src, d_dst, numBlocks, threadsPerBlock); // rotate image

    write_image(output_file_bilinear, h_dst, LENGTH, LENGTH); // write image

    // bicubic

    checkCudaErr(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice), "cudaMemcpy"); // copy image to device

    repeated_bicubic(h_dst, d_src, d_dst, numBlocks, threadsPerBlock); // rotate image

    write_image(output_file_bicubic, h_dst, LENGTH, LENGTH); // write image

    // free memory
    free(h_src);
    free(h_dst);

    // free device memory
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
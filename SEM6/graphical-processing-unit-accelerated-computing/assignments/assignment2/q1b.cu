#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI ((double)3.14159265358979323846264338327950288419716939937510)

const long LENGTH = 256L;
const double ROTATION = 15 * PI / 180.0;
const long REPEAT = 24L;

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
        fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
    return err;
}

/**
 * @brief Rotate an image using nearest neighbour interpolation
 *
 * @param src
 * @param dst
 * @param w
 * @param theta
 * @return __global__
 */
__global__ void rotate_nearest_neighbour(const uint8_t *src, uint8_t *dst, const long w, const double theta)
{
    const long x = threadIdx.x + blockIdx.x * blockDim.x; // x value of the thread
    const long y = threadIdx.y + blockIdx.y * blockDim.y; // y value of the thread

    if (x < w && y < w)
    {                                 // check if the thread is inside the image boundaries
        const double c = cosf(theta); // cosine of the rotation angle
        const double s = sinf(theta); // sine of the rotation angle

        const double co = (w - 1L) / 2.0; // center of the image

        const double dx = x - co; // x coordinate of the pixel in the rotated image
        const double dy = y - co; // y coordinate of the pixel in the rotated image

        const long Rxf = (long)(dx * c - dy * s + co + 0.5); // x coordinate of the pixel in the original image
        const long Ryf = (long)(dx * s + dy * c + co + 0.5); // y coordinate of the pixel in the original image

        dst[y * w + x] = (0L <= Rxf && Rxf < w && 0L <= Ryf && Ryf < w) ? src[Ryf * w + Rxf] : 0; // copy the pixel
    }
}

/**
 * @brief Rotate an image using bilinear interpolation
 *
 * @param src
 * @param dst
 * @param w
 * @param theta
 * @return __global__
 */
__global__ void rotate_bilinear(const uint8_t *src, uint8_t *dst, const long w, const double theta)
{
    const long x = threadIdx.x + blockIdx.x * blockDim.x; // x value of the thread
    const long y = threadIdx.y + blockIdx.y * blockDim.y; // y value of the thread

    if (x < w && y < w)
    {                                 // check if the thread is inside the image boundaries
        const double c = cosf(theta); // cosine of the rotation angle
        const double s = sinf(theta); // sine of the rotation angle

        const double co = (w - 1L) / 2.0; // center of the image

        const double dx = x - co; // x coordinate of the pixel in the rotated image
        const double dy = y - co; // y coordinate of the pixel in the rotated image

        const double Rx = dx * c - dy * s + co; // x coordinate of the pixel in the original image
        const double Ry = dx * s + dy * c + co; // y coordinate of the pixel in the original image

        const long Rxf = (long)floor(Rx + 0.5); // x coordinate of the pixel in the original image
        const long Ryf = (long)floor(Ry + 0.5); // y coordinate of the pixel in the original image

        const double dRx = Rx - Rxf; // distance between the pixel and the left pixel
        const double dRy = Ry - Ryf; // distance between the pixel and the top pixel

        if (0L <= Rxf && Rxf + 1L < w && 0L <= Ryf &&
            Ryf + 1L < w) // check if the pixel is inside the image boundaries
            dst[y * w + x] = (1.0 - dRx) * (1.0 - dRy) * src[Ryf * w + Rxf] +
                             dRx * (1.0 - dRy) * src[Ryf * w + Rxf + 1L] +
                             (1.0 - dRx) * dRy * src[(Ryf + 1L) * w + Rxf] +
                             dRx * dRy * src[(Ryf + 1L) * w + Rxf + 1L]; // copy the pixel
        else
            dst[y * w + x] = 0; // copy the pixel
    }
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
__device__ double bicubic_interpolate(const double p[4][4], double x, double y)
{
    double arr[4];
    arr[0] = interpolate(p[0], x);
    arr[1] = interpolate(p[1], x);
    arr[2] = interpolate(p[2], x);
    arr[3] = interpolate(p[3], x);
    return interpolate(arr, y);
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
__global__ void rotate_bicubic(const uint8_t *const src, uint8_t *dst, const long w, const double theta)
{
    const long x = threadIdx.x + blockIdx.x * blockDim.x; // x value of the thread
    const long y = threadIdx.y + blockIdx.y * blockDim.y; // y value of the thread

    if (x < w && y < w)
    {
        const double c = cosf(theta);    // cosine of the rotation angle
        const double s = sinf(theta);    // sine of the rotation angle
        const double co = (w - 1) / 2.0; // center of the image

        double p[4][4]; // 4x4 matrix

        const double dx = x - co;               // x coordinate of the pixel in the rotated image
        const double dy = y - co;               // y coordinate of the pixel in the rotated image
        const double Rx = dx * c - dy * s + co; // x coordinate of the pixel in the original image
        const double Ry = dx * s + dy * c + co; // y coordinate of the pixel in the original image
        if (0 <= Rx && Rx + 1 < w && 0 <= Ry && Ry + 1 < w)
        { // check if the pixel is inside the image boundaries

            const long Rxf = (long)floor(Rx + 0.5); // x coordinate of the pixel in the original image
            const long Ryf = (long)floor(Ry + 0.5); // y coordinate of the pixel in the original image

            const double dRx = Rx - Rxf; // distance between the pixel and the left pixel
            const double dRy = Ry - Ryf; // distance between the pixel and the top pixel

            for (uint8_t m = 0; m < 4; m++)
            {
                for (uint8_t n = 0; n < 4; n++)
                {
                    int xidx = Rxf + n - 1; // x coordinate of the pixel in the original image
                    int yidx = Ryf + m - 1; // y coordinate of the pixel in the original image

                    p[m][n] =
                        xidx >= 0 && xidx < w && yidx >= 0 && yidx < w ? src[yidx * w + xidx] : 0; // copy the pixel
                }
            }

            double interpolated_value = bicubic_interpolate(p, dRx, dRy); // interpolate the pixel

            dst[y * w + x] = interpolated_value > 255.0 || interpolated_value < 0.0
                                 ? 0
                                 : (uint8_t)round(interpolated_value); // copy the pixel
        }
        else
            dst[y * w + x] = 0; // copy the pixel
    }
}

/**
 * @brief Read an image from a file
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
 * @brief Write an image to a file
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
 * @brief Repeatedly rotate an image using nearest neighbour interpolation
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
    float gpu_time = 0.0f;           // time used on the GPU for all iterations
    cudaEvent_t start_gpu, stop_gpu; // events for timing

    for (long i = 0L; i < REPEAT; i++)
    {                               // repeat the operation REPEAT times
        float gpu_time_used = 0.0f; // time used on the GPU for one iteration

        checkCudaErr(cudaEventCreate(&start_gpu), "cudaEventCreate");   // create the start event
        checkCudaErr(cudaEventCreate(&stop_gpu), "cudaEventCreate");    // create the stop event
        checkCudaErr(cudaEventRecord(start_gpu, 0), "cudaEventRecord"); // record the start event

        rotate_nearest_neighbour<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, LENGTH, ROTATION); // call the kernel
        checkCudaErr(cudaDeviceSynchronize(), "cudaDeviceSynchronize");                           // wait for the kernel to finish

        checkCudaErr(cudaEventRecord(stop_gpu, 0), "cudaEventRecord");        // record the stop event
        checkCudaErr(cudaEventSynchronize(stop_gpu), "cudaEventSynchronize"); // wait for the stop event to be recorded
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
                 "cudaMemcpy"); // copy the result back to the host

    printf("nearest neighbour: %f ms\n", gpu_time / REPEAT); // print the average time used
}

/**
 * @brief Repeatedly rotate an image using bilinear interpolation
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
    float gpu_time;                  // time used on the GPU for all iterations
    cudaEvent_t start_gpu, stop_gpu; // events for timing

    for (long i = 0L; i < REPEAT; i++)
    {                               // repeat the operation REPEAT times
        float gpu_time_used = 0.0f; // time used on the GPU for one iteration

        checkCudaErr(cudaEventCreate(&start_gpu), "cudaEventCreate");   // create the start event
        checkCudaErr(cudaEventCreate(&stop_gpu), "cudaEventCreate");    // create the stop event
        checkCudaErr(cudaEventRecord(start_gpu, 0), "cudaEventRecord"); // record the start event

        rotate_bilinear<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, LENGTH, ROTATION); // call the kernel

        checkCudaErr(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // wait for the kernel to finish

        checkCudaErr(cudaEventRecord(stop_gpu, 0), "cudaEventRecord");        // record the stop event
        checkCudaErr(cudaEventSynchronize(stop_gpu), "cudaEventSynchronize"); // wait for the stop event to be recorded
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
                 "cudaMemcpy"); // copy the result back to the host

    printf("bilinear: %f ms\n", gpu_time / REPEAT); // print the average time used
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

        rotate_bicubic<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, LENGTH, ROTATION); // call the kernel

        checkCudaErr(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // wait for the kernel to finish

        checkCudaErr(cudaEventRecord(stop_gpu, 0), "cudaEventRecord");        // record the stop event
        checkCudaErr(cudaEventSynchronize(stop_gpu), "cudaEventSynchronize"); // wait for the stop event to be recorded
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
    const long size = sizeof(uint8_t) * LENGTH * LENGTH; // size of the image
    // pointers to the host memory
    uint8_t *h_src = (uint8_t *)malloc(size);
    uint8_t *h_dst = (uint8_t *)malloc(size);

    const dim3 numBlocks(16, 16);       // number of blocks in the grid
    const dim3 threadsPerBlock(16, 16); // number of threads in each block

    // input image
    const char input_file[] = "../content/lena.img";

    // output images
    const char output_file_nearest_neighbour[] = "../content/lena_nearest_neighbour.img";
    const char output_file_bilinear[] = "../content/lena_bilinear.img";
    const char output_file_bicubic[] = "../content/lena_bicubic.img";

    read_image(input_file, h_src, LENGTH, LENGTH); // read the image

    uint8_t *d_src, *d_dst; // pointers to the device memory

    checkCudaErr(cudaMalloc((void **)&d_src, size), "cudaMalloc");                      // allocate memory on the device
    checkCudaErr(cudaMalloc((void **)&d_dst, size), "cudaMalloc");                      // allocate memory on the device
    checkCudaErr(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice), "cudaMemcpy"); // copy the image to the device

    // nearest neighbour

    repeated_nearest_neighbour(h_dst, d_src, d_dst, numBlocks, threadsPerBlock);

    write_image(output_file_nearest_neighbour, h_dst, LENGTH, LENGTH); // write the image

    // bilinear

    checkCudaErr(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice), "cudaMemcpy"); // copy the image to the device

    repeated_bilinear(h_dst, d_src, d_dst, numBlocks, threadsPerBlock);

    write_image(output_file_bilinear, h_dst, LENGTH, LENGTH); // write the image

    // bicubic

    checkCudaErr(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice), "cudaMemcpy"); // copy the image to the device

    repeated_bicubic(h_dst, d_src, d_dst, numBlocks, threadsPerBlock);

    write_image(output_file_bicubic, h_dst, LENGTH, LENGTH); // write the image

    // free the memory
    free(h_src);
    free(h_dst);

    // free the memory on the device
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
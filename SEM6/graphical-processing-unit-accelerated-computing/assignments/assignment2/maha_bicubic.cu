#include <cuda.h>
#include <stdio.h>

const long  BLOCK_SIZE  = 16;
const long  Width       = 256;
const long  Height      = 256;
const long  Size        = Width * Height;
const long  SizeInBytes = Size * sizeof(unsigned char);
const float PI          = 3.14159265f;

inline cudaError_t checkCudaErr(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
    }
    return err;
}

void load_image(char* fname, unsigned char* inputImage) {
    FILE* fp = fopen(fname, "rb");
    if (fp) {
        fread(inputImage, 1, Size, fp);
        fclose(fp);

    } else {
        puts("Cannot open raw image.");
    }
}

void save_image(char* fname, unsigned char* outputImage) {
    FILE* fp = fopen(fname, "wb");
    if (fp) {
        fwrite(outputImage, sizeof(unsigned char), Size, fp);

        fclose(fp);
    }

    else {
        puts("Cannot write raw image.");
    }
}

_device_ double cubicInterpolate(double p[4], double x) {
    return p[1] + 0.5 * x *
                      (p[2] - p[0] +
                       x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}

_device_ double bicubicInterpolate(double p[4][4], double x, double y) {
    double arr[4];
    arr[0] = cubicInterpolate(p[0], y);
    arr[1] = cubicInterpolate(p[1], y);
    arr[2] = cubicInterpolate(p[2], y);
    arr[3] = cubicInterpolate(p[3], y);
    return cubicInterpolate(arr, x);
}

_global_ void rotate(unsigned char* inputImage, unsigned char* outputImage, int height, int width) {
    double theta = 24 * PI / 180.0;
    // compute center of image
    const double cx = width / 2.0;
    const double cy = height / 2.0;

    int tx = blockIdx.y * blockDim.y + threadIdx.y;
    int ty = blockIdx.x * blockDim.x + threadIdx.x;

    if (tx < height && ty < width) {
        float       a11 = cos(theta);
        float       a21 = sin(theta);
        float       a12 = -sin(theta);
        float       a22 = cos(theta);
        const float x   = a11 * (ty - cy) + a21 * (tx - cx) + cx;
        const float y   = a12 * (ty - cy) + a22 * (tx - cx) + cy;
        if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1) {
            outputImage[tx * width + ty] = 0;
        } else {
            int x0 = (int)floor(x + 0.5);
            int y0 = (int)floor(y + 0.5);

            double u = x - x0;
            double v = y - y0;

            double p[4][4];
            for (int m = -1; m <= 2; m++) {
                for (int n = -1; n <= 2; n++) {
                    int xidx = x0 + n;
                    int yidx = y0 + m;
                    if (xidx < 0 || xidx >= width || yidx < 0 || yidx >= height) {
                        p[m + 1][n + 1] = 0;
                    } else {
                        p[m + 1][n + 1] = inputImage[yidx * width + xidx];
                    }
                }
            }
            outputImage[tx * width + ty] = (unsigned char)(int)bicubicInterpolate(p, u, v);
        }
    }
}

int main() {
    float theta = 20.0f * 3.14159265f / 180.0f;

    unsigned char* inputImage  = (unsigned char*)malloc(SizeInBytes);
    unsigned char* outputImage = (unsigned char*)malloc(SizeInBytes);
    unsigned char* d_inputImage;
    unsigned char* d_outputImage;

    checkCudaErr(cudaMalloc(&d_inputImage, SizeInBytes), "malloc1");
    checkCudaErr(cudaMalloc(&d_outputImage, SizeInBytes), "malloc2");

    load_image("/content/lena.img", inputImage);
    checkCudaErr(cudaMemcpy(d_inputImage, inputImage, SizeInBytes, cudaMemcpyHostToDevice), "memcpy1");

    dim3 blockSize(16, 16);
    dim3 gridSize(16, 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 15; i++) {
        rotate<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, 256, 256);
        checkCudaErr(cudaMemcpy(d_inputImage, d_outputImage, SizeInBytes, cudaMemcpyDeviceToDevice), "memcpy2");
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time elapsed: %f ms\n", milliseconds);
    checkCudaErr(cudaMemcpy(outputImage, d_outputImage, SizeInBytes, cudaMemcpyDeviceToHost), "memcpy3");
    save_image("/content/output.img", outputImage);

    free(inputImage);
    free(outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    return 0;
}
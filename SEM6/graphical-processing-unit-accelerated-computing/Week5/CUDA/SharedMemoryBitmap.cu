#include <cuda.h>
#include <stdio.h>

const int WIDTH               = 512;
const int HEIGHT              = 512;
const int SIZE                = WIDTH * HEIGHT;
const int IMAGE_SIZE_IN_BYTES = SIZE * sizeof(unsigned char) * 4;

void load_raw_image(const char* imageName, unsigned char* pData) {
        FILE* fp = fopen(imageName, "rb");
        if (fp) {
                fread(pData, 1, SIZE, fp);
                fclose(fp);
        } else
                puts("Cannot open raw image.");
}

void save_raw_image(const char* imageName, unsigned char* pData) {
        FILE* fp = fopen(imageName, "wb");
        if (fp) {
                fwrite(pData, 4 * sizeof(unsigned char), SIZE, fp);
                fclose(fp);
        } else
                puts("Cannot write raw image.");
}

__global__ void MakeImageKernel(unsigned char* ptr) {
        int              x      = threadIdx.x + blockIdx.x * blockDim.x;
        int              y      = threadIdx.y + blockIdx.y * blockDim.y;
        int              offset = x + y * blockDim.x * gridDim.x;
        __shared__ float shared[16][16];
        const float      period = 128.0f;
        const float      PI     = 3.14159f;
        shared[threadIdx.x][threadIdx.y] =
                255 * (sinf(x * 2.0f * PI / period) + 1.0f) * (sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;
        //__syncthreads(); //uncomment this line to get correct output
        ptr[offset * 4 + 0] = 0;
        ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
        ptr[offset * 4 + 2] = 0;
        ptr[offset * 4 + 3] = 255;
}

int main(int argc, char** argv) {
        unsigned char* host_bitmap = (unsigned char*)malloc(IMAGE_SIZE_IN_BYTES);
        unsigned char* dev_bitmap;

        cudaMalloc(&dev_bitmap, IMAGE_SIZE_IN_BYTES);

        dim3 blocksGrid;
        dim3 threadsBlock(16, 16, 1);
        blocksGrid.x = ceil(WIDTH / 16.0);
        blocksGrid.y = ceil(HEIGHT / 16.0);

        MakeImageKernel<<<blocksGrid, threadsBlock>>>(dev_bitmap);

        cudaMemcpy(host_bitmap, dev_bitmap, IMAGE_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

        save_raw_image("/content/OutputImage.raw", host_bitmap);

        free(host_bitmap);
        cudaFree(&dev_bitmap);
        cudaDeviceReset();
        return 0;
}
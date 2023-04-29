#include <cuda.h>
#include <stdio.h>

const int WIDTH               = 512;
const int HEIGHT              = 512;
const int SIZE                = WIDTH * HEIGHT;
const int IMAGE_SIZE_IN_BYTES = SIZE * sizeof(unsigned char);

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
                fwrite(pData, 1, SIZE, fp);
                fclose(fp);
        } else
                puts("Cannot write raw image.");
}

__global__ void PictureKernel(unsigned char* d_Pin, unsigned char* d_Pout, int n, int m, float brightness = 1) {
        // Calculate the row #
        int Row = blockIdx.y * blockDim.y + threadIdx.y;

        // Calculate the column #
        int Col = blockIdx.x * blockDim.x + threadIdx.x;

        if ((Row < m) && (Col < n)) {
                int offset  = (Row * n) + Col;
                int offset2 = (((n - 1) - Row) * n) + Col;  // this is to flip the output image

                d_Pout[offset2] = d_Pin[offset] * brightness;
        }
}

int main(int argc, char** argv) {
        unsigned char* host_bitmap = (unsigned char*)malloc(IMAGE_SIZE_IN_BYTES);
        unsigned char* dev_bitmap;
        unsigned char* dev_bitmap2;

        cudaMalloc(&dev_bitmap, IMAGE_SIZE_IN_BYTES);
        cudaMalloc(&dev_bitmap2, IMAGE_SIZE_IN_BYTES);

        load_raw_image("Baboon.raw", host_bitmap);

        cudaMemcpy(dev_bitmap, host_bitmap, IMAGE_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

        dim3 blocksGrid;
        dim3 threadsBlock(16, 16, 1);
        blocksGrid.x = ceil(WIDTH / 16.0);
        blocksGrid.y = ceil(HEIGHT / 16.0);

        PictureKernel<<<blocksGrid, threadsBlock>>>(dev_bitmap, dev_bitmap2, WIDTH, HEIGHT, 0.5);

        cudaMemcpy(host_bitmap, dev_bitmap2, IMAGE_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

        save_raw_image("BaboonFlipped.raw", host_bitmap);

        puts("Image saved");

        free(host_bitmap);
        cudaFree(&dev_bitmap);
        cudaFree(&dev_bitmap2);

        return 0;
}
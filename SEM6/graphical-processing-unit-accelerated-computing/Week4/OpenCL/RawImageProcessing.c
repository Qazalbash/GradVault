#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#define MAX_SOURCE_SIZE          (0x100000)

#include <CL/cl.h>
#include <math.h>
#include <stdio.h>

const int WIDTH               = 512;
const int HEIGHT              = 512;
const int SIZE                = WIDTH * HEIGHT;
const int IMAGE_SIZE_IN_BYTES = SIZE * sizeof(uint8_t);

typedef struct {
        size_t x, y, z;
} dim3;

void load_raw_image(const char* imageName, uint8_t* pData) {
        FILE* fp = fopen(imageName, "rb");
        if (fp) {
                fread(pData, 1, SIZE, fp);
                fclose(fp);
        } else
                puts("Cannot open raw image.");
}

void save_raw_image(const char* imageName, uint8_t* pData) {
        FILE* fp = fopen(imageName, "wb");
        if (fp) {
                fwrite(pData, 1, SIZE, fp);
                fclose(fp);
        } else
                puts("Cannot write raw image.");
}

int main(void) {
        unsigned char* host_bitmap = (unsigned char*)malloc(IMAGE_SIZE_IN_BYTES);
        unsigned char* dev_bitmap;
        // unsigned char* dev_bitmap2;

        float brightness = 0.5;

        load_raw_image("Baboon.raw", host_bitmap);

        dim3 blocksGrid;
        blocksGrid.x = WIDTH;
        blocksGrid.y = HEIGHT;

        dim3 threadsBlock;
        threadsBlock.x = 16;
        threadsBlock.y = 16;

        size_t* global_work_size = (size_t*)malloc(2 * sizeof(size_t));
        *global_work_size        = blocksGrid.x;
        *(global_work_size + 1)  = blocksGrid.y;

        size_t* local_work_size = (size_t*)malloc(2 * sizeof(size_t));
        *local_work_size        = threadsBlock.x;
        *(local_work_size + 1)  = threadsBlock.y;

        cl_int err;

        cl_device_id device_id;

        err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not get device.", err);
                return -1;
        }

        cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create context.", err);
                return -1;
        }

        cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create command queue.", err);
                return -1;
        }

        FILE* fp = fopen("kernel.cl", "r");
        fseek(fp, 0, SEEK_END);
        size_t size = ftell(fp);

        if (size == 0) {
                printf("Error: %d. kernel file has no function.", err);
                return -1;
        }

        fseek(fp, 0, SEEK_SET);
        char* source = (char*)malloc(size);
        fread(source, 1, size, fp);
        fclose(fp);

        cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &size, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create program.", err);
                return -1;
        }

        err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not build program.", err);
                return -1;
        }

        cl_kernel kernel = clCreateKernel(program, "PictureKernel", &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create kernel.", err);
                return -1;
        }

        cl_mem dev_bitmap_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_SIZE_IN_BYTES, NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        cl_mem dev_bitmap2_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, IMAGE_SIZE_IN_BYTES, NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        cl_mem n_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        cl_mem m_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        cl_mem brightness_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        err = clEnqueueWriteBuffer(queue, dev_bitmap_mem_obj, CL_TRUE, 0, IMAGE_SIZE_IN_BYTES, host_bitmap, 0, NULL,
                                   NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not write buffer.", err);
                return -1;
        }

        err = clEnqueueWriteBuffer(queue, n_mem_obj, CL_TRUE, 0, sizeof(int), &WIDTH, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not write buffer.", err);
                return -1;
        }

        err = clEnqueueWriteBuffer(queue, m_mem_obj, CL_TRUE, 0, sizeof(int), &HEIGHT, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not write buffer.", err);
                return -1;
        }

        err = clEnqueueWriteBuffer(queue, brightness_mem_obj, CL_TRUE, 0, sizeof(float), &brightness, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not write buffer.", err);
                return -1;
        }

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&dev_bitmap_mem_obj);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&dev_bitmap2_mem_obj);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&n_mem_obj);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&m_mem_obj);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&brightness_mem_obj);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not set kernel arguments.", err);
                return -1;
        }

        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not enqueue kernel.", err);
                return -1;
        }

        err = clEnqueueReadBuffer(queue, dev_bitmap2_mem_obj, CL_TRUE, 0, IMAGE_SIZE_IN_BYTES, host_bitmap, 0, NULL,
                                  NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not read buffer.", err);
                return -1;
        }

        err = clReleaseKernel(kernel);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release kernel.", err);
                return -1;
        }

        err = clReleaseProgram(program);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release program.", err);
                return -1;
        }

        err = clReleaseCommandQueue(queue);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release command queue.", err);
                return -1;
        }

        err = clReleaseContext(context);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release context.", err);
                return -1;
        }

        err = clReleaseMemObject(dev_bitmap_mem_obj);
        err |= clReleaseMemObject(dev_bitmap2_mem_obj);
        err |= clReleaseMemObject(n_mem_obj);
        err |= clReleaseMemObject(m_mem_obj);
        err |= clReleaseMemObject(brightness_mem_obj);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release buffer.", err);
                return -1;
        }

        save_raw_image("BaboonFlipped.raw", host_bitmap);

        free(host_bitmap);

        return 0;
}
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#define MAX_SOURCE_SIZE          (0x100000)

#include <CL/cl.h>
#include <stdio.h>

typedef struct {
        size_t x, y, z;
} dim3;

int main(void) {
        const int num_elements_x = 16;
        const int num_elements_y = 16;

        const int num_bytes = num_elements_x * num_elements_y * sizeof(int);

        int* array = (int*)malloc(num_bytes);

        dim3 block_size;
        block_size.x = 4;
        block_size.y = 4;

        dim3 grid_size;
        grid_size.x = num_elements_x;
        grid_size.y = num_elements_y;

        // block dimension
        size_t* global_work_size = (size_t*)malloc(2 * sizeof(size_t));
        *global_work_size        = grid_size.x;
        *(global_work_size + 1)  = grid_size.y;

        // grid dimension
        size_t* local_work_size = (size_t*)malloc(2 * sizeof(size_t));
        *local_work_size        = block_size.x;
        *(local_work_size + 1)  = block_size.y;

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

        cl_mem array_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_bytes, NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        cl_kernel kernel = clCreateKernel(program, "Simple2D", &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create kernel.", err);
                return -1;
        }

        err = clSetKernelArg(kernel, 0, sizeof(array_mem_obj), (void*)&array_mem_obj);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not set kernel arguments.", err);
                return -1;
        }

        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not enqueue kernel.", err);
                return -1;
        }

        err = clEnqueueReadBuffer(queue, array_mem_obj, CL_TRUE, 0, num_bytes, array, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not read buffer.", err);
                return -1;
        }

        for (int row = 0; row < num_elements_y; ++row) {
                for (int col = 0; col < num_elements_x; ++col) printf("%2d ", array[row * num_elements_x + col]);
                printf("\n");
        }
        printf("\n");

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

        err = clReleaseMemObject(array_mem_obj);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release buffer.", err);
                return -1;
        }

        free(array);
        free(global_work_size);
        free(local_work_size);
        free(source);

        return 0;
}
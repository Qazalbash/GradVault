#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#define MAX_SOURCE_SIZE          (0x100000)

#include <CL/cl.h>
#include <stdio.h>

void sum_host(const int* a, const int* b, int* c, const size_t N) {
        for (size_t i = 0; i < N; ++i) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
        cl_int err;

        const size_t N = 4194304UL;  // 2^22

        int* a = (int*)malloc(N * sizeof(int));
        int* b = (int*)malloc(N * sizeof(int));
        int* c = (int*)malloc(N * sizeof(int));

        for (size_t i = 0; i < N; i++) {
                a[i] = i;
                b[i] = (i + 1) * 2;
                c[i] = 0;
        }

        sum_host(a, b, c, N);

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

        cl_kernel kernel = clCreateKernel(program, "sum", &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create kernel.", err);
                return -1;
        }

        cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(int), NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL couldnot create buffer.", err);
                return -1;
        }

        cl_mem N_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        err = clEnqueueWriteBuffer(queue, a_mem_obj, CL_TRUE, 0, N * sizeof(int), a, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, b_mem_obj, CL_TRUE, 0, N * sizeof(int), b, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, N_mem_obj, CL_TRUE, 0, sizeof(size_t), &N, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not write buffer.", err);
                return -1;
        }

        err = clSetKernelArg(kernel, 0, sizeof(a_mem_obj), (void*)&a_mem_obj);
        err |= clSetKernelArg(kernel, 1, sizeof(b_mem_obj), (void*)&b_mem_obj);
        err |= clSetKernelArg(kernel, 2, sizeof(c_mem_obj), (void*)&c_mem_obj);
        err |= clSetKernelArg(kernel, 3, sizeof(N_mem_obj), (void*)&N_mem_obj);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not set kernel arguments.", err);
                return -1;
        }

        const size_t* global_item_size = N;
        const size_t* local_item_size  = 64;

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, (const size_t**)&global_item_size,
                                     (const size_t**)&local_item_size, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not enqueue kernel.", err);
                return -1;
        }

        err = clEnqueueReadBuffer(queue, c_mem_obj, CL_TRUE, 0, sizeof(int) * N, c, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not read buffer.", err);
                return -1;
        }

        // output results
        for (size_t i = 0; i < N; ++i) printf("%d + %d = %d\n", a[i], b[i], c[i]);

        clFlush(queue);
        clFinish(queue);

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

        err = clReleaseMemObject(a_mem_obj);
        err |= clReleaseMemObject(b_mem_obj);
        err |= clReleaseMemObject(c_mem_obj);
        err |= clReleaseMemObject(N_mem_obj);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release buffer.", err);
                return -1;
        }

        free(source);
        free(a);
        free(b);
        free(c);

        return 0;
}
#ifndef MAT_MUL_GAURD
#define MAT_MUL_GAURD

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#define MAX_SOURCE_SIZE          (0x100000)

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
        srand(time(0));

        cl_int err;

        const int m = 2, p = 4, n = 2;

        float *A = (float *)malloc(m * p * sizeof(float));  // m * p
        float *B = (float *)malloc(p * n * sizeof(float));  // p * n
        float *C = (float *)malloc(m * n * sizeof(float));  // m * n

        size_t i;
        for (i = 0; i < (size_t)(m * p); i++) A[i] = rand() / (float)RAND_MAX;
        for (i = 0; i < (size_t)(p * n); i++) B[i] = rand() / (float)RAND_MAX;

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

        FILE *fp = fopen("kernel.cl", "r");
        fseek(fp, 0, SEEK_END);
        size_t size = ftell(fp);

        if (size == 0) {
                printf("Error: %d. kernel file has no function.", 1);
                return -1;
        }

        fseek(fp, 0, SEEK_SET);
        char *source = (char *)malloc(size);
        fread(source, 1, size, fp);
        fclose(fp);

        cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, &size, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create program.", err);
                return -1;
        }

        err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not build program.", err);
                return -1;
        }

        // Create memory buffers on the device for each vector
        cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, m * p * sizeof(float), NULL, &err);
        cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, p * n * sizeof(float), NULL, &err);
        cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, m * n * sizeof(float), NULL, &err);
        cl_mem m_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
        cl_mem p_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
        cl_mem n_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err);

        // Copy the lists A and B to their respective memory buffers
        err = clEnqueueWriteBuffer(queue, a_mem_obj, CL_TRUE, 0, m * p * sizeof(float), A, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, b_mem_obj, CL_TRUE, 0, p * n * sizeof(float), B, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, m_mem_obj, CL_TRUE, 0, sizeof(int), &m, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, p_mem_obj, CL_TRUE, 0, sizeof(int), &p, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, n_mem_obj, CL_TRUE, 0, sizeof(int), &n, 0, NULL, NULL);

        // Create the OpenCL kernel
        cl_kernel kernel = clCreateKernel(program, "matrix_multiplication", &err);

        // Set the arguments of the kernel
        err = clSetKernelArg(kernel, 0, sizeof(a_mem_obj), (void *)&a_mem_obj);
        err |= clSetKernelArg(kernel, 1, sizeof(b_mem_obj), (void *)&b_mem_obj);
        err |= clSetKernelArg(kernel, 2, sizeof(c_mem_obj), (void *)&c_mem_obj);
        err |= clSetKernelArg(kernel, 3, sizeof(m_mem_obj), (void *)&m_mem_obj);
        err |= clSetKernelArg(kernel, 4, sizeof(p_mem_obj), (void *)&p_mem_obj);
        err |= clSetKernelArg(kernel, 5, sizeof(n_mem_obj), (void *)&n_mem_obj);

        // Execute the OpenCL kernel on the list
        const size_t global_item_size = m * n * p;  // Process the entire lists

        const size_t local_item_size = 8;  // Divide work items into groups of

        // Execute the kernel on the device
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, (const size_t *)&global_item_size,
                                     (const size_t *)&local_item_size,
                                     //  NULL, NULL,
                                     0, NULL, NULL);

        // Read the memory buffer C on the device to the local variable C
        err = clEnqueueReadBuffer(queue, c_mem_obj, CL_TRUE, 0, m * n * sizeof(float), C, 0, NULL, NULL);

        // Display the result to the screen

        for (i = 0; i < m * p; i++) {
                if (i % p == 0) printf("\n");
                printf("%f ", A[i]);
        }
        printf("\n");

        for (i = 0; i < p * n; i++) {
                if (i % n == 0) printf("\n");
                printf("%f ", B[i]);
        }
        printf("\n");

        for (i = 0; i < m * n; i++) {
                if (i % n == 0) printf("\n");
                printf("%f ", C[i]);
        }
        printf("\n");

        // Clean up
        err = clFlush(queue);
        err |= clFinish(queue);
        err |= clReleaseKernel(kernel);
        err |= clReleaseProgram(program);
        err |= clReleaseMemObject(a_mem_obj);
        err |= clReleaseMemObject(b_mem_obj);
        err |= clReleaseMemObject(c_mem_obj);
        err |= clReleaseMemObject(m_mem_obj);
        err |= clReleaseMemObject(p_mem_obj);
        err |= clReleaseMemObject(n_mem_obj);
        err |= clReleaseCommandQueue(queue);
        err |= clReleaseContext(context);

        free(source);
        free(A);
        free(B);
        free(C);

        return 0;
}

#endif
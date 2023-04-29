/*
 * File:   main.c
 * Author: tapin13
 *
 * Created on July 23, 2018, 2:50 AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>

void device_version(const cl_device_id device) {
        char device_version[1000] = {0};
        clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_version), device_version, NULL);
        printf("OpenCL version support: %s\n", device_version);
}

void device_max_mem_alloc_size(const cl_device_id device) {
        cl_ulong device_max_mem_alloc_size;
        clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &device_max_mem_alloc_size, NULL);
        printf("Device max mem alloc size: %lu\n", device_max_mem_alloc_size);
}

int main(int argc, char **argv) {
        cl_uint        num_entries = 1, num_platforms;
        cl_platform_id platforms;

        cl_int ret = clGetPlatformIDs(num_entries, &platforms, &num_platforms);

        if (ret != CL_SUCCESS) {
                fprintf(stderr,
                        "num_entries is equal to zero and platforms is not NULL, or if "
                        "both num_platforms and platforms are NULL\n");
                return (EXIT_FAILURE);
        }

        printf("Number of OpenCL platforms available: %u\n", num_platforms);

        cl_device_id device;
        cl_uint      num_devices;

        ret = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_DEFAULT, num_entries, &device, &num_devices);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clGetDeviceIDs\n");
                return (EXIT_FAILURE);
        }

        printf("The number of OpenCL devices available: %u\n", num_devices);

        device_version(device);
        device_max_mem_alloc_size(device);

        cl_context context = clCreateContext(NULL, num_devices, &device, NULL, NULL, &ret);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clCreateContext\n");
                return (EXIT_FAILURE);
        }

        cl_command_queue command_queue =
                clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ret);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clCreateCommandQueue\n");
                return (EXIT_FAILURE);
        }

        const char file_name[] = "kernel.cl";
        FILE      *fp          = fopen(file_name, "r");

        if (!fp) {
                fprintf(stderr, "file open problem\n");
                return (EXIT_FAILURE);
        }

        fseek(fp, 0, SEEK_END);
        size_t source_size = ftell(fp);
        rewind(fp);

        char *source_str = (char *)malloc(source_size);
        source_size      = fread(source_str, 1, source_size, fp);
        fclose(fp);

        cl_uint count = 1;

        cl_program program = clCreateProgramWithSource(context, count, (const char **)&source_str,
                                                       (const size_t *)&source_size, &ret);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clCreateProgramWithSource\n");
                return (EXIT_FAILURE);
        }

        ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clBuildProgram\n");
                return (EXIT_FAILURE);
        }

        cl_kernel kernel = clCreateKernel(program, "test", &ret);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clCreateKernel\n");
                return (EXIT_FAILURE);
        }

        int     mem_length = 10;
        cl_int *mem        = (cl_int *)malloc(sizeof(cl_int) * mem_length);
        memset(mem, 0, mem_length);

        printf("--- Before ---\n");
        int i;
        for (i = 0; i < mem_length; i++) {
                printf("mem[%d]: %d\n", i, mem[i]);
        }

        cl_mem mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_length * sizeof(cl_int), NULL, &ret);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clCreateBuffer\n");
                return (EXIT_FAILURE);
        }

        ret = clEnqueueWriteBuffer(command_queue, mem_obj, CL_TRUE, 0, mem_length * sizeof(cl_int), mem, 0, NULL, NULL);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clEnqueueWriteBuffer\n");
                return (EXIT_FAILURE);
        }

        cl_uint arg_index = 0;
        size_t  arg_size  = sizeof(cl_mem);

        ret = clSetKernelArg(kernel, arg_index, arg_size, (void *)&mem_obj);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clSetKernelArg\n");
                return (EXIT_FAILURE);
        }

        size_t global_work_size[1] = {10};

        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clEnqueueNDRangeKernel\n");
                return (EXIT_FAILURE);
        }

        ret = clEnqueueReadBuffer(command_queue, mem_obj, CL_TRUE, 0, mem_length * sizeof(float), mem, 0, NULL, NULL);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clEnqueueReadBuffer\n");
                return (EXIT_FAILURE);
        }

        printf("--- After ---\n");
        for (i = 0; i < mem_length; i++) printf("mem[%d]: %d\n", i, mem[i]);

        free(mem);
        free(source_str);

        return (EXIT_SUCCESS);
}

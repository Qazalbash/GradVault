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

int main(int argc, char **argv) {
        cl_int  ret;
        cl_uint count       = 1;
        cl_uint num_entries = 1;
        cl_uint num_devices = 0;

        cl_device_id   device;
        cl_platform_id platforms = NULL;

        cl_context       context;
        cl_command_queue command_queue;
        cl_program       program;
        cl_kernel        kernel;
        cl_mem           mem_obj;

        size_t    mem_length = 20;
        cl_float *mem        = (cl_float *)malloc(sizeof(cl_float) * mem_length);
        memset(mem, 0, mem_length);
        size_t global_work_size[] = {mem_length};

        // get device id of GPU
        ret = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, num_entries, &device, &num_devices);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clGetDeviceIDs\n");
                return (EXIT_FAILURE);
        }

        // create context
        context = clCreateContext(NULL, num_devices, &device, NULL, NULL, &ret);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clCreateContext\n");
                return (EXIT_FAILURE);
        }

        // create command queue
        command_queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ret);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clCreateCommandQueue\n");
                return (EXIT_FAILURE);
        }

        const char file_name[] = "OpenCLtemplate.cl";
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

        // create program
        program = clCreateProgramWithSource(context, count, (const char **)&source_str, (const size_t *)&source_size,
                                            &ret);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clCreateProgramWithSource\n");
                return (EXIT_FAILURE);
        }

        // build program
        ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clBuildProgram\n");
                return (EXIT_FAILURE);
        }

        // create kernel
        kernel = clCreateKernel(program, "test", &ret);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clCreateKernel\n");
                return (EXIT_FAILURE);
        }

        // create memory object
        mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_length * sizeof(cl_int), NULL, &ret);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clCreateBuffer\n");
                return (EXIT_FAILURE);
        }

        // write data to memory object
        ret = clEnqueueWriteBuffer(command_queue, mem_obj, CL_TRUE, 0, mem_length * sizeof(cl_int), mem, 0, NULL, NULL);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clEnqueueWriteBuffer\n");
                return (EXIT_FAILURE);
        }

        cl_uint arg_index = 0;
        size_t  arg_size  = sizeof(cl_mem);

        // set kernel arguments
        ret = clSetKernelArg(kernel, arg_index, arg_size, (void *)&mem_obj);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clSetKernelArg\n");
                return (EXIT_FAILURE);
        }

        // execute kernel
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clEnqueueNDRangeKernel\n");
                return (EXIT_FAILURE);
        }

        // read data from memory object
        ret = clEnqueueReadBuffer(command_queue, mem_obj, CL_TRUE, 0, mem_length * sizeof(float), mem, 0, NULL, NULL);

        if (ret != CL_SUCCESS) {
                fprintf(stderr, "error in clEnqueueReadBuffer\n");
                return (EXIT_FAILURE);
        }

        // print data
        for (int i = 0; i < mem_length; i++) printf("mem[%d]: %f\n", i, mem[i]);

        free(mem);
        free(source_str);

        return (EXIT_SUCCESS);
}

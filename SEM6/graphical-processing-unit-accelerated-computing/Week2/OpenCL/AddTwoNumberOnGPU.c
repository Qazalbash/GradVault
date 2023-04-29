#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#define MAX_SOURCE_SIZE          (0x100000)

#include <CL/cl.h>
#include <stdio.h>
#include <string.h>

int main() {
        cl_uint      numDevices;
        cl_device_id device;
        size_t       globalWorkSize = 1, localWorkSize = 1;
        int          a = 3, b = 2, c;

        cl_int err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);

        if (err != CL_SUCCESS) {
                printf("Failed to create a device group.\n");
                return -1;
        }

        cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if (err != CL_SUCCESS) {
                printf("Failed to create a compute context.\n");
                return -1;
        }

        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

        if (err != CL_SUCCESS) {
                printf("Failed to create a command commands.\n");
                return -1;
        }

        FILE *fp;
        char  fileName[] = "./kernel.cl";

        fp = fopen(fileName, "r");

        if (!fp) {
                fprintf(stderr, "Failed to load kernel.\n");
                exit(1);
        }

        char  *source_str  = (char *)malloc(MAX_SOURCE_SIZE);
        size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);

        cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, &source_size, &err);

        if (err != CL_SUCCESS) {
                printf("Failed to create compute program.\n");
                return -1;
        }

        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Failed to build program executable.\n");
                return -1;
        }

        cl_kernel kernel = clCreateKernel(program, "add", &err);

        cl_mem a_memobj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int), NULL, NULL);

        err = clEnqueueWriteBuffer(queue, a_memobj, CL_TRUE, 0, sizeof(cl_int), &a, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Failed to write to source array a.\n");
                return -1;
        }

        cl_mem b_memobj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int), NULL, NULL);

        err = clEnqueueWriteBuffer(queue, b_memobj, CL_TRUE, 0, sizeof(cl_int), &b, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Failed to write to source array b.\n");
                return -1;
        }

        cl_mem c_memobj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int), NULL, NULL);

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_memobj);
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_memobj);
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_memobj);

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);

        err = clEnqueueReadBuffer(queue, c_memobj, CL_TRUE, 0, sizeof(cl_int), &c, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Failed to read output array.\n");
                return -1;
        }

        printf("%d + %d = %d\n", a, b, c);

        clReleaseMemObject(a_memobj);
        clReleaseMemObject(b_memobj);
        clReleaseMemObject(c_memobj);
        clReleaseProgram(program);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 0;
}

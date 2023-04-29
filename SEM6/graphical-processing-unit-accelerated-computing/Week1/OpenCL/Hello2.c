#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#define MAX_SOURCE_SIZE          (0x100000)

#include <CL/cl.h>
#include <stdio.h>

int main() {
        FILE *fp = fopen("kernel.cl", "r");

        if (!fp) {
                fprintf(stderr, "Failed to load kernel.\n");
                return -1;
        }

        char *kernelSource = (char *)malloc(MAX_SOURCE_SIZE);
        fread(kernelSource, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);

        cl_device_id device;

        clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

        cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

        cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, NULL);

        clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

        cl_kernel kernel = clCreateKernel(program, "greeting", NULL);

        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, (const size_t[]){1}, (const size_t[]){1}, 0, NULL, NULL);

        clFinish(queue);

        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 0;
}
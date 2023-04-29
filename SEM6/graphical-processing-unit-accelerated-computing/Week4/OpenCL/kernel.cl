/*
    get_group_id(uint dimindx)    =   blockIdx.[xyz]
    get_local_size(uint dimindx)  =   blockDim.[xyz]
    get_local_id(uint dimindx)    =   threadIdx.[xyz]
    get_num_groups(uint dimindx)  =   gridDim.[xyz]
*/

__kernel void matrix_multiplication(__global const float *A, __global const float *B, __global float *C,
                                    __global const int *M, __global const int *P, __global const int *N) {
        float sum = 0.0f;

        int tx = get_global_id(0), i = tx / *N, j = tx % *N;
        if (i < *M && j < *N)
                for (int k = 0; k < *P; k++) sum += A[i * *P + k] * B[k * *N + j];

        C[i * *N + j] = sum;
}

__kernel void PictureKernel(__global const unsigned char *d_Pin, __global unsigned char *d_Pout, __global const int *n,
                            __global const int *m, __global const float *brightness) {
        // Calculate the row #
        int Row = get_group_id(1) * get_local_size(1) + get_local_id(1);

        // Calculate the column #
        int Col = get_group_id(0) * get_local_size(0) + get_local_id(0);

        if ((Row < *m) && (Col < *n)) {
                int offset = (Row * *n) + Col;
                // this is to flip the output image
                int offset2 = (((*n - 1) - Row) * *n) + Col;

                d_Pout[offset2] = d_Pin[offset] * *brightness;
        }
}

__kernel void Simple2D(__global int *array) {
        int index_x = get_group_id(0) * get_local_size(0) + get_local_id(0);
        int index_y = get_group_id(1) * get_local_size(1) + get_local_id(1);

        int grid_width = get_num_groups(0) * get_local_size(0);
        int index      = index_y * grid_width + index_x;

        int result = get_group_id(1) * get_num_groups(0) + get_group_id(0);

        array[index] = result;
}
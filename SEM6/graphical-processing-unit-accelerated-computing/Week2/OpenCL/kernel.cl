__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
        int i = get_global_id(0);

        C[i] = A[i] + B[i];
}

__kernel void add(__global const int *A, __global const int *B, __global int *C) { *C = *A + *B; }

__kernel void InitData(__global int *A, __global int *B) {}

__kernel void calculator(__global const int *A, __global const int *B, __global int *C, int op) {
        switch (op) {
                case 0:
                        *C = *A + *B;
                        break;
                case 1:
                        *C = *A - *B;
                        break;
                case 2:
                        *C = *A * *B;
                        break;
                case 3:
                        *C = *A / *B;
                        break;
                default:
                        *C = 0;
                        break;
        }
}
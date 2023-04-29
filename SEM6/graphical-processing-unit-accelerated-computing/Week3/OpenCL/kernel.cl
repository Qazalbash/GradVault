__kernel void sum(__global const int *a, __global const int *b, __global int *c, __global const size_t *N) {
        int i = get_global_id(0);
        if (i < *N) c[i] = a[i] + b[i];
}
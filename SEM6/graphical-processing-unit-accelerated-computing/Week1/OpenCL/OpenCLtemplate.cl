void hello_world(int i) { printf("%d Hello World!\n", i); }

__kernel void test(__global float* message) {
    int gid      = get_global_id(0);
    message[gid] = gid >> 1;
    hello_world(gid);
}

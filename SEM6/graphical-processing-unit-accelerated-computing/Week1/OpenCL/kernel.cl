__kernel void greeting() {
        int id = get_global_id(0);
        printf("%d Hello World! (GPU)\n", id);
}
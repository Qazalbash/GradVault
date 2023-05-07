__global__ void rotate_nearest_neighbour(const uint8_t* const src, uint8_t* dst, const long w, const long h,
                                         const double theta) {
    const long x = threadIdx.x + blockIdx.x * blockDim.x;
    const long y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < w && y < h) {
        const double c = cosf(theta);
        const double s = sinf(theta);

        const double cx = (w - 1L) / 2.0;
        const double cy = (h - 1L) / 2.0;

        const double dx = x - cx;
        const double dy = y - cy;

        const long Rxf = (long)floor(dx * c - dy * s + cx + 0.5);
        const long Ryf = (long)floor(dx * s + dy * c + cy + 0.5);

        dst[y * w + x] = (0 <= Rxf && Rxf < w && 0 <= Ryf && Ryf < h) ? src[Ryf * w + Rxf] : 0;
    }
}

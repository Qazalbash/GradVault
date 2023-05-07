__global__ void rotate_bilinear(const uint8_t* const src, uint8_t* dst, const long w, const long h,
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

        const double Rx = dx * c - dy * s + cx;
        const double Ry = dx * s + dy * c + cy;

        const double Rxf = (long)floor(Rx + 0.5);
        const double Ryf = (long)floor(Ry + 0.5);

        const double dRx = Rx - Rxf;
        const double dRy = Ry - Ryf;

        if (0 <= Rxf && Rxf + 1 < w && 0 <= Ryf && Ryf + 1 < h) {
            dst[y * w + x] = (1 - dRx) * (1 - dRy) * src[Ryf * w + Rxf] + dRx * (1 - dRy) * src[Ryf * w + Rxf + 1] +
                             (1 - dRx) * dRy * src[(Ryf + 1) * w + Rxf] + dRx * dRy * src[(Ryf + 1) * w + Rxf + 1];
        } else {
            dst[y * w + x] = 0;
        }
    }
}

#endif

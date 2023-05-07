__global__ void rotate_bicubic(const uint8_t* const src, uint8_t* dst, const long w, const long h, const double theta) {
    const long x = threadIdx.x + blockIdx.x * blockDim.x;
    const long y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < w && y < h) {
        double       a[4], b[4];
        const double c = cosf(theta);
        const double s = sinf(theta);

        const double cx = (w - 1) / 2.0;
        const double cy = (h - 1) / 2.0;

        const double dx = x - cx;
        const double dy = y - cy;

        const double Rx = dx * c - dy * s + cx;
        const double Ry = dx * s + dy * c + cy;

        const double Rxf = (long)floor(Rx);
        const double Ryf = (long)floor(Ry);

        const double dRx = Rx - Rxf;
        const double dRy = Ry - Ryf;

        if (1 <= Rxf && Rxf + 2 < w && 1 <= Ryf && Ryf + 2 < h) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) a[j] = src[(Ryf - 1 + i) * w + Rxf - 1 + j];

                b[i] = a[0] * (-1.0 / 6.0 * dRx * dRx * dRx + 1.0 / 2.0 * dRx * dRx - 1.0 / 2.0 * dRx + 1.0 / 6.0) +
                       a[1] * (1.0 / 2.0 * dRx * dRx * dRx - dRx * dRx + 2.0 / 3.0) +
                       a[2] * (-1.0 / 2.0 * dRx * dRx * dRx + dRx * dRx + 1.0 / 2.0 * dRx + 1.0 / 3.0) +
                       a[3] * (1.0 / 6.0 * dRx * dRx * dRx - 1.0 / 6.0 * dRx * dRx);
            }
            dst[y * w + x] =
                b[0] * (-1.0 / 6.0 * dRy * dRy * dRy + 1.0 / 2.0 * dRy * dRy - 1.0 / 2.0 * dRy + 1.0 / 6.0) +
                b[1] * (1.0 / 2.0 * dRy * dRy * dRy - dRy * dRy + 2.0 / 3.0) +
                b[2] * (-1.0 / 2.0 * dRy * dRy * dRy + dRy * dRy + 1.0 / 2.0 * dRy + 1.0 / 3.0) +
                b[3] * (1.0 / 6.0 * dRy * dRy * dRy - 1.0 / 6.0 * dRy * dRy);
        } else {
            dst[y * w + x] = 0;
        }
    }
}

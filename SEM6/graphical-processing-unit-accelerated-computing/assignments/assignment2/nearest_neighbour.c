#ifndef NN
#define NN

/**
 * @brief Nearest neighbour interpolation
 *
 * @param src
 * @param dst
 * @param w
 * @param h
 * @param theta
 */
void rotate_nearest_neighbour(const uint8_t* const src, uint8_t* dst, const long w, const long h, const double theta) {
    const double c  = cos(theta);
    const double s  = sin(theta);
    const double cx = (w - 1) / 2.0;
    const double cy = (h - 1) / 2.0;
    double       dx, dy;
    long         Rxf, Ryf, x, y;
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            dx  = x - cx;
            dy  = y - cy;
            Rxf = (long)floor(dx * c - dy * s + cx + 0.5);
            Ryf = (long)floor(dx * s + dy * c + cy + 0.5);

            dst[y * w + x] = (0 <= Rxf && Rxf < w && 0 <= Ryf && Ryf < h) ? src[Ryf * w + Rxf] : 0;
        }
    }
}

#endif
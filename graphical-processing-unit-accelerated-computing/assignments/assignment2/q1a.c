#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI ((double)3.14159265358979323846264338327950288419716939937510)

/**
 * @brief Nearest neighbour interpolation
 *
 * @param src
 * @param dst
 * @param w
 * @param h
 * @param theta
 */
void rotate_nearest_neighbour(const uint8_t *const src, uint8_t *dst, const long w, const long h, const double theta)
{
    const double c = cos(theta);
    const double s = sin(theta);
    const double cx = (w - 1) / 2.0;
    const double cy = (h - 1) / 2.0;
    double dx, dy;
    long Rxf, Ryf, x, y;
    for (y = 0; y < h; y++)
    {
        for (x = 0; x < w; x++)
        {
            dx = x - cx;
            dy = y - cy;
            Rxf = (long)floor(dx * c - dy * s + cx + 0.5);
            Ryf = (long)floor(dx * s + dy * c + cy + 0.5);

            dst[y * w + x] = (0 <= Rxf && Rxf < w && 0 <= Ryf && Ryf < h) ? src[Ryf * w + Rxf] : 0;
        }
    }
}

/**
 * @brief Bilinear interpolation
 *
 * @param src
 * @param dst
 * @param w
 * @param h
 * @param theta
 */
void rotate_bilinear(const uint8_t *const src, uint8_t *dst, const long w, const long h, const double theta)
{
    const double c = cos(theta);
    const double s = sin(theta);
    const double cx = (w - 1) / 2.0;
    const double cy = (h - 1) / 2.0;
    double dx, dy, Rx, Ry, dRx, dRy;
    long Rxf, Ryf, y, x;
    for (y = 0; y < h; y++)
    {
        for (x = 0; x < w; x++)
        {
            dx = x - cx;
            dy = y - cy;
            Rx = dx * c - dy * s + cx;
            Ry = dx * s + dy * c + cy;
            Rxf = (long)floor(Rx + 0.5);
            Ryf = (long)floor(Ry + 0.5);

            dRx = Rx - Rxf;
            dRy = Ry - Ryf;

            if (0 <= Rxf && Rxf + 1 < w && 0 <= Ryf && Ryf + 1 < h)
            {
                dst[y * w + x] = (1 - dRx) * (1 - dRy) * src[Ryf * w + Rxf] + dRx * (1 - dRy) * src[Ryf * w + Rxf + 1] +
                                 (1 - dRx) * dRy * src[(Ryf + 1) * w + Rxf] + dRx * dRy * src[(Ryf + 1) * w + Rxf + 1];
            }
            else
            {
                dst[y * w + x] = 0;
            }
        }
    }
}

/**
 * @brief Bicubic interpolation
 *
 * @param c
 * @param x
 * @return double
 */
double interpolate(const double *const c, const double x)
{
    return c[1] + 0.5 * x *
                      (c[2] - c[0] +
                       x * (2.0 * c[0] - 5.0 * c[1] + 4.0 * c[2] - c[3] + x * (3.0 * (c[1] - c[2]) + c[3] - c[0])));
}

/**
 * @brief Bicubic interpolation using 4x4 matrix
 *
 * @param p
 * @param u
 * @param v
 * @return double
 */
double bicubic_interpolate(const double p[4][4], const double u, const double v)
{
    double px[4], py[4];
    for (int i = 0; i < 4; i++)
    {
        px[i] = interpolate(p[i], u);
        py[i] = interpolate(&p[0][i], v);
    }
    return interpolate(px, v);
}

/**
 * @brief Bicubic interpolation
 *
 * @param src
 * @param dst
 * @param w
 * @param h
 * @param theta
 */
void rotate_bicubic(const uint8_t *const src, uint8_t *dst, const long w, const long h, const double theta)
{
    const double c = cos(theta);
    const double s = sin(theta);
    const double cx = (w - 1) / 2.0;
    const double cy = (h - 1) / 2.0;

    double dx, dy, Rx, Ry, dRx, dRy, interpolated_value;
    long Rxf, Ryf, y, x;

    double p[4][4];

    for (y = 0; y < h; y++)
    {
        for (x = 0; x < w; x++)
        {
            dx = x - cx;
            dy = y - cy;
            Rx = dx * c - dy * s + cx;
            Ry = dx * s + dy * c + cy;
            if (0 <= Rx && Rx + 1 < w && 0 <= Ry && Ry + 1 < h)
            {
                Rxf = (int)floor(Rx + 0.5);
                Ryf = (int)floor(Ry + 0.5);

                dRx = Rx - Rxf;
                dRy = Ry - Ryf;

                for (uint8_t m = 0; m < 4; m++)
                {
                    for (uint8_t n = 0; n < 4; n++)
                    {
                        int xidx = Rxf + n - 1;
                        int yidx = Ryf + m - 1;

                        p[m][n] = xidx >= 0 && xidx < w && yidx >= 0 && yidx < h ? src[yidx * w + xidx] : 0;
                    }
                }

                interpolated_value = bicubic_interpolate(p, dRx, dRy);

                dst[y * w + x] =
                    interpolated_value > 255.0 || interpolated_value < 0.0 ? 0 : (uint8_t)round(interpolated_value);
            }
            else
                dst[y * w + x] = 0;
        }
    }
}

/**
 * @brief Read the image
 *
 * @param filename
 * @param img
 * @param w
 * @param h
 */
void read_image(char *filename, uint8_t *img, const long w, const long h)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "cannot open %s for reading\n", filename);
        exit(1);
    }
    for (long i = 0UL; i < w * h; i++)
    {
        unsigned char c;
        fread(&c, sizeof(unsigned char), 1, fp);
        img[i] = (double)c;
    }
    fclose(fp);
}

/**
 * @brief Write the image
 *
 * @param filename
 * @param img
 * @param w
 * @param h
 */
void write_image(char *filename, uint8_t *img, const long w, const long h)
{
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        fprintf(stderr, "cannot open %s for writing\n", filename);
        exit(1);
    }
    for (long i = 0UL; i < w * h; i++)
    {
        unsigned char c = (unsigned char)img[i];
        fwrite(&c, sizeof(unsigned char), 1, fp);
    }
    fclose(fp);
}

/**
 * @brief Rotate the image according to the given interpolation method
 *
 * @param src
 * @param dst
 * @param method
 * @param width
 * @param height
 * @param theta
 */
void rotate_interpolate_image(const uint8_t *const src, uint8_t *dst, const uint8_t method, const long width,
                              const long height, const double theta)
{
    switch (method)
    {
    case 0:
        rotate_nearest_neighbour(src, dst, width, height, theta);
        break;
    case 1:
        rotate_bilinear(src, dst, width, height, theta);
        break;
    case 2:
        rotate_bicubic(src, dst, width, height, theta);
        break;
    default:
        fprintf(stderr, "unknown method: %d\n", method);
        exit(1);
    }
}

/**
 * @brief Main function
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char *argv[])
{
    if (argc != 7)
    {
        fprintf(stderr, "usage: %s <input> <output> <width> <height> <method> <theta>\n", argv[0]);
        exit(1);
    }

    const int w = atoi(argv[3]);
    const int h = atoi(argv[4]);
    const uint8_t method = atoi(argv[5]);
    const double theta = atof(argv[6]) * PI / 180.0;
    uint8_t *src = (uint8_t *)malloc(sizeof(uint8_t) * w * h);
    uint8_t *dst = (uint8_t *)malloc(sizeof(uint8_t) * w * h);
    uint8_t *tmp;

    read_image(argv[1], src, w, h);

    clock_t start = clock(), end;

    for (int i = 0; i < 4; i++)
    {
        rotate_interpolate_image(src, dst, method, w, h, theta);
        tmp = src;
        src = dst;
        dst = tmp;
    }

    end = clock();

    printf("%f ms\n", (double)(end - start) / (CLOCKS_PER_SEC * 4));

    write_image(argv[2], dst, w, h);

    free(src);
    free(dst);
    return 0;
}

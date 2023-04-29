#ifndef BC
#define BC
// reference https://www.paulinternet.nl/?page=bicubic

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PI          3.14159265358979323846
#define Width       256
#define Height      256
#define Size        Width* Height
#define SizeInBytes Size * sizeof(unsigned char)

void load_image(char* fname, unsigned char* inputImage) {
    FILE* fp = fopen(fname, "rb");
    if (fp) {
        fread(inputImage, 1, Size, fp);
        fclose(fp);

    } else {
        puts("Cannot open raw image.");
    }
}

void save_image(char* fname, unsigned char* outputImage) {
    FILE* fp = fopen(fname, "wb");
    if (fp) {
        fwrite(outputImage, sizeof(unsigned char), Size, fp);

        fclose(fp);
    }

    else {
        puts("Cannot write raw image.");
    }
}

double interpolate(double* c, double x) {
    double result =
        c[1] +
        0.5 * x *
            (c[2] - c[0] + x * (2.0 * c[0] - 5.0 * c[1] + 4.0 * c[2] - c[3] + x * (3.0 * (c[1] - c[2]) + c[3] - c[0])));
    return result;
}

double bicubicInterpolate(double p[4][4], double u, double v) {
    double px[4], py[4];
    for (int i = 0; i < 4; i++) {
        px[i] = interpolate(p[i], u);
        py[i] = interpolate(&p[0][i], v);
    }
    double result = interpolate(px, v);
    return result;
}

void rotate(unsigned char* inputImage, unsigned char* outputImage) {
    double theta = 24 * PI / 180;

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Width; j++) {
            float a11 = cos(theta);
            float a21 = sin(theta);
            float a12 = -sin(theta);
            float a22 = cos(theta);
            float x   = (a11 * (j - Width / 2) + a21 * (i - Height / 2) + Width / 2);
            float y   = (a12 * (j - Width / 2) + a22 * (i - Height / 2) + Height / 2);
            if (x >= 0 && x < Width - 1 && y >= 0 && y < Height - 1) {
                int x0 = (int)floor(x + 0.0);
                int y0 = (int)floor(y + 0.0);

                double intX = x - x0;
                double intY = y - y0;

                double p[4][4];
                // populate neighbours
                for (int m = 0; m < 4; m++) {
                    for (int n = 0; n <= 3; n++) {
                        int xidx = x0 + n - 1;
                        int yidx = y0 + m - 1;
                        if (xidx >= 0 && xidx < Width && yidx >= 0 && yidx < Height) {
                            p[m][n] = inputImage[yidx * Width + xidx];
                        } else {
                            p[m][n] = 0;
                        }
                    }
                }
                float interpolated = bicubicInterpolate(p, intX, intY);
                if (interpolated > 255 || interpolated < 0) {
                    interpolated = 0;
                }
                outputImage[i * Width + j] = (unsigned char)round(interpolated);
            } else {
                outputImage[i * Width + j] = 0;
            }
        }
    }
}

int main() {
    unsigned char* inputImage  = (unsigned char*)malloc(SizeInBytes);
    unsigned char* outputImage = (unsigned char*)malloc(SizeInBytes);

    load_image("/content/lena.img", inputImage);

    clock_t start = clock();
    for (int i = 0; i < 15; i++) {
        rotate(inputImage, outputImage);
        memcpy(inputImage, outputImage, SizeInBytes);
    }
    clock_t end        = clock();
    double  time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", time_spent);

    save_image("/content/output.img", outputImage);

    return 0;
}

#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// nearest neighbour interpolation
void nearest_neighbour(uint8_t *in, uint8_t *out, int width, int height, int new_width, int new_height)
{
    float scale_x = (float)width / new_width;
    float scale_y = (float)height / new_height;

    for (int y = 0; y < new_height; y++)
    {
        for (int x = 0; x < new_width; x++)
        {
            int in_x = (int)(x * scale_x);
            int in_y = (int)(y * scale_y);
            out[y * new_width + x] = in[in_y * width + in_x];
        }
    }
}

// bilinear interpolation
void bilinear(uint8_t *in, uint8_t *out, int width, int height, int new_width, int new_height)
{
    float scale_x = (float)width / new_width;
    float scale_y = (float)height / new_height;

    for (int y = 0; y < new_height; y++)
    {
        for (int x = 0; x < new_width; x++)
        {
            float in_x = x * scale_x;
            float in_y = y * scale_y;
            int in_x0 = (int)in_x;
            int in_y0 = (int)in_y;
            int in_x1 = in_x0 + 1;
            int in_y1 = in_y0 + 1;
            float a = in_x - in_x0;
            float b = in_y - in_y0;
            out[y * new_width + x] = (1 - a) * (1 - b) * in[in_y0 * width + in_x0] +
                                     (1 - a) * b * in[in_y1 * width + in_x0] + a * (1 - b) * in[in_y0 * width + in_x1] +
                                     a * b * in[in_y1 * width + in_x1];
        }
    }
}

// bicubic interpolation
float cubic(float x)
{
    float a = -0.5f;
    if (x < 0)
        x = -x;

    if (x < 1)
        return (a + 2) * x * x * x - (a + 3) * x * x + 1;
    else if (x < 2)
        return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a;
    else
        return 0;
}

void bicubic(uint8_t *in, uint8_t *out, int width, int height, int new_width, int new_height)
{
    float scale_x = (float)width / new_width;
    float scale_y = (float)height / new_height;

    for (int y = 0; y < new_height; y++)
    {
        for (int x = 0; x < new_width; x++)
        {
            const float in_x = x * scale_x;
            const float in_y = y * scale_y;
            const int in_x0 = (int)in_x;
            const int in_y0 = (int)in_y;
            const float a = in_x - in_x0;
            const float b = in_y - in_y0;
            float sum = 0;
            for (int j = -1; j <= 2; j++)
            {
                for (int i = -1; i <= 2; i++)
                {
                    int in_xi = in_x0 + i;
                    int in_yj = in_y0 + j;
                    if (in_xi < 0)
                    {
                        in_xi = 0;
                    }
                    else if (in_xi >= width)
                    {
                        in_xi = width - 1;
                    }
                    if (in_yj < 0)
                    {
                        in_yj = 0;
                    }
                    else if (in_yj >= height)
                    {
                        in_yj = height - 1;
                    }
                    float w = cubic(i - a) * cubic(j - b);
                    sum += w;
                    out[y * new_width + x] += w * in[in_yj * width + in_xi];
                }
            }
            out[y * new_width + x] /= sum;
        }
    }
}

// main
int main(int argc, char *argv[])
{
    if (argc != 8)
    {
        printf(
            "usage: %s <input file> <output file> <width> <height> <interpolation method> <new width> <new "
            "height>\n",
            argv[0]);
        return 1;
    }

    const int WIDTH = atoi(argv[3]);
    const int HEIGHT = atoi(argv[4]);
    const size_t SIZE = WIDTH * HEIGHT;
    const int METHOD = atoi(argv[5]);
    const int NEW_WIDTH = atoi(argv[6]);
    const int NEW_HEIGHT = atoi(argv[7]);
    const size_t NEW_SIZE = NEW_WIDTH * NEW_HEIGHT;

    // read input image

    FILE *fp = fopen(argv[1], "rb");
    if (!fp)
    {
        printf("error: could not open file %s", argv[1]);
        return 1;
    }

    uint8_t *in = (uint8_t *)malloc(SIZE);
    if (!in)
    {
        printf("error: could not allocate memory");
        return 1;
    }

    if (fread(in, 1, SIZE, fp) != SIZE)
    {
        printf("error: could not read file %s", argv[1]);
        return 1;
    }

    fclose(fp);

    // allocate output image

    uint8_t *out = (uint8_t *)malloc(NEW_SIZE);
    if (!out)
    {
        printf("error: could not allocate memory");
        return 1;
    }

    // interpolate

    switch (METHOD)
    {
    case 0:
        nearest_neighbour(in, out, WIDTH, HEIGHT, NEW_WIDTH, NEW_HEIGHT);
        break;
    case 1:
        bilinear(in, out, WIDTH, HEIGHT, NEW_WIDTH, NEW_HEIGHT);
        break;

    case 2:
        bicubic(in, out, WIDTH, HEIGHT, NEW_WIDTH, NEW_HEIGHT);
        break;
    }

    // write output image

    fp = fopen("out.img", "wb");

    if (!fp)
    {
        printf("error: could not open file out.raw");
        return 1;
    }

    if (fwrite(out, 1, NEW_SIZE, fp) != NEW_SIZE)
    {
        printf("error: could not write file out.raw");
        return 1;
    }

    fclose(fp);

    free(in);
    free(out);

    return 0;
}
/*****************************************************************************
 *	Date: January 5, 2009
 *----------------------------------------------------------------------------
 *	This C program is based on the following paper:
 *		P. Thevenaz, T. Blu, M. Unser, "Interpolation Revisited,"
 *		IEEE Transactions on Medical Imaging,
 *		vol. 19, no. 7, pp. 739-758, July 2000.
 *----------------------------------------------------------------------------
 *	Philippe Thevenaz
 *	EPFL/STI/IMT/LIB/BM.4.137
 *	Station 17
 *	CH-1015 Lausanne VD
 *----------------------------------------------------------------------------
 *	phone (CET):	+41(21)693.51.61
 *	fax:			+41(21)693.37.01
 *	RFC-822:		philippe.thevenaz@epfl.ch
 *	X-400:			/C=ch/A=400net/P=switch/O=epfl/S=thevenaz/G=philippe/
 *	URL:			http://bigwww.epfl.ch/
 *----------------------------------------------------------------------------
 *	This file is best viewed with 4-space tabs (the bars below should be aligned)
 *	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|
 *  |...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|
 ****************************************************************************/

/*****************************************************************************
 *	System includes
 ****************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*****************************************************************************
 *	Other includes
 ****************************************************************************/
#include "coeff.h"
#include "interpol.h"
#include "io.h"

/*****************************************************************************
 *	Defines
 ****************************************************************************/
#define PI ((double)3.14159265358979323846264338327950288419716939937510)

/*****************************************************************************
 *	Definition of extern procedures
 ****************************************************************************/
/*--------------------------------------------------------------------------*/
extern int main(int argc, const char *argv[])

{ /* begin main */

    float *ImageRasterArray, *OutputImage;
    float *p;
    double a11, a12, a21, a22;
    double x0, y0, x1, y1;
    double xOrigin, yOrigin;
    double Angle, xShift, yShift;
    long   Width, Height;
    long   SplineDegree;
    long   x, y;
    int    Masking;
    int    Error;

    /* access data samples */
    Error = ReadByteImageRawData(&ImageRasterArray, &Width, &Height);
    if (Error) {
        printf("Failure to import image data\n");
        return (1);
    }

    /* ask for transformation parameters */
    RigidBody(&Angle, &xShift, &yShift, &xOrigin, &yOrigin, &SplineDegree, &Masking);

    /* allocate output image */
    OutputImage = (float *)malloc((size_t)(Width * Height * (long)sizeof(float)));
    if (OutputImage == (float *)NULL) {
        free(ImageRasterArray);
        printf("Allocation of output image failed\n");
        return (1);
    }

    /* convert between a representation based on image samples */
    /* and a representation based on image B-spline coefficients */
    Error = SamplesToCoefficients(ImageRasterArray, Width, Height, SplineDegree);
    if (Error) {
        free(OutputImage);
        free(ImageRasterArray);
        printf("Change of basis failed\n");
        return (1);
    }

    /* prepare the geometry */
    Angle *= PI / 180.0;
    a11    = cos(Angle);
    a12    = -sin(Angle);
    a21    = sin(Angle);
    a22    = cos(Angle);
    x0     = a11 * (xShift + xOrigin) + a12 * (yShift + yOrigin);
    y0     = a21 * (xShift + xOrigin) + a22 * (yShift + yOrigin);
    xShift = xOrigin - x0;
    yShift = yOrigin - y0;

    /* visit all pixels of the output image and assign their value */
    p = OutputImage;
    for (y = 0L; y < Height; y++) {
        x0 = a12 * (double)y + xShift;
        y0 = a22 * (double)y + yShift;
        for (x = 0L; x < Width; x++) {
            x1 = x0 + a11 * (double)x;
            y1 = y0 + a21 * (double)x;
            if (Masking) {
                if ((x1 <= -0.5) || (((double)Width - 0.5) <= x1) || (y1 <= -0.5) || (((double)Height - 0.5) <= y1)) {
                    *p++ = 0.0F;
                } else {
                    *p++ = (float)InterpolatedValue(ImageRasterArray, Width, Height, x1, y1, SplineDegree);
                }
            } else {
                *p++ = (float)InterpolatedValue(ImageRasterArray, Width, Height, x1, y1, SplineDegree);
            }
        }
    }

    /* save output */
    Error = WriteByteImageRawData(OutputImage, Width, Height);
    if (Error) {
        free(OutputImage);
        free(ImageRasterArray);
        printf("Failure to export image data\n");
        return (1);
    }

    free(OutputImage);
    free(ImageRasterArray);
    printf("Done\n");
    return (0);
} /* end main */

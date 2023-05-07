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
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*****************************************************************************
 *	Other includes
 ****************************************************************************/
#include "interpol.h"

/*****************************************************************************
 *	Definition of extern procedures
 ****************************************************************************/
/*--------------------------------------------------------------------------*/
extern double InterpolatedValue(float *Bcoeff,      /* input B-spline array of coefficients */
                                long   Width,       /* width of the image */
                                long   Height,      /* height of the image */
                                double x,           /* x coordinate where to interpolate */
                                double y,           /* y coordinate where to interpolate */
                                long   SplineDegree /* degree of the spline model */
)

{ /* begin InterpolatedValue */

    float *p;
    double xWeight[10], yWeight[10];
    double interpolated;
    double w, w2, w4, t, t0, t1;
    long   xIndex[10], yIndex[10];
    long   Width2 = 2L * Width - 2L, Height2 = 2L * Height - 2L;
    long   i, j, k;

    /* compute the interpolation indexes */
    if (SplineDegree & 1L) {
        i = (long)floor(x) - SplineDegree / 2L;
        j = (long)floor(y) - SplineDegree / 2L;
        for (k = 0L; k <= SplineDegree; k++) {
            xIndex[k] = i++;
            yIndex[k] = j++;
        }
    } else {
        i = (long)floor(x + 0.5) - SplineDegree / 2L;
        j = (long)floor(y + 0.5) - SplineDegree / 2L;
        for (k = 0L; k <= SplineDegree; k++) {
            xIndex[k] = i++;
            yIndex[k] = j++;
        }
    }

    /* compute the interpolation weights */
    switch (SplineDegree) {
        case 2L:
            /* x */
            w          = x - (double)xIndex[1];
            xWeight[1] = 0.75 - w * w;
            xWeight[2] = 0.5 * (w - xWeight[1] + 1.0);
            xWeight[0] = 1.0 - xWeight[1] - xWeight[2];
            /* y */
            w          = y - (double)yIndex[1];
            yWeight[1] = 0.75 - w * w;
            yWeight[2] = 0.5 * (w - yWeight[1] + 1.0);
            yWeight[0] = 1.0 - yWeight[1] - yWeight[2];
            break;
        case 3L:
            /* x */
            w          = x - (double)xIndex[1];
            xWeight[3] = (1.0 / 6.0) * w * w * w;
            xWeight[0] = (1.0 / 6.0) + 0.5 * w * (w - 1.0) - xWeight[3];
            xWeight[2] = w + xWeight[0] - 2.0 * xWeight[3];
            xWeight[1] = 1.0 - xWeight[0] - xWeight[2] - xWeight[3];
            /* y */
            w          = y - (double)yIndex[1];
            yWeight[3] = (1.0 / 6.0) * w * w * w;
            yWeight[0] = (1.0 / 6.0) + 0.5 * w * (w - 1.0) - yWeight[3];
            yWeight[2] = w + yWeight[0] - 2.0 * yWeight[3];
            yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];
            break;
        default:
            printf("Invalid spline degree\n");
            return (0.0);
    }

    /* apply the mirror boundary conditions */
    for (k = 0L; k <= SplineDegree; k++) {
        xIndex[k] = (Width == 1L) ? (0L)
                                  : ((xIndex[k] < 0L) ? (-xIndex[k] - Width2 * ((-xIndex[k]) / Width2))
                                                      : (xIndex[k] - Width2 * (xIndex[k] / Width2)));
        if (Width <= xIndex[k]) {
            xIndex[k] = Width2 - xIndex[k];
        }
        yIndex[k] = (Height == 1L) ? (0L)
                                   : ((yIndex[k] < 0L) ? (-yIndex[k] - Height2 * ((-yIndex[k]) / Height2))
                                                       : (yIndex[k] - Height2 * (yIndex[k] / Height2)));
        if (Height <= yIndex[k]) {
            yIndex[k] = Height2 - yIndex[k];
        }
    }

    /* perform interpolation */
    interpolated = 0.0;
    for (j = 0L; j <= SplineDegree; j++) {
        p = Bcoeff + (ptrdiff_t)(yIndex[j] * Width);
        w = 0.0;
        for (i = 0L; i <= SplineDegree; i++) {
            w += xWeight[i] * p[xIndex[i]];
        }
        interpolated += yWeight[j] * w;
    }

    return (interpolated);
} /* end InterpolatedValue */

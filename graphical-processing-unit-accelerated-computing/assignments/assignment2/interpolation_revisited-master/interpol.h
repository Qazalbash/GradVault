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

/*--------------------------------------------------------------------------*/
extern double InterpolatedValue(float *Bcoeff,      /* input B-spline array of coefficients */
                                long   Width,       /* width of the image */
                                long   Height,      /* height of the image */
                                double x,           /* x coordinate where to interpolate */
                                double y,           /* y coordinate where to interpolate */
                                long   SplineDegree /* degree of the spline model */
);

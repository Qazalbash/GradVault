#ifndef IMAGE_HPP
#define IMAGE_HPP

/**
   This file declares the Image class which is a utility for storing image
   pixels and writing them to file in PPM format.

   x increases to the right, and y to the bottom.
*/

#include <fstream>
#include <string>

#include "../world/ViewPlane.hpp"
#include "BBox.hpp"
#include "RGBColor.hpp"

class RGBColor;
class ViewPlane;

class Image {
private:

    int        hres, vres;  // resolution.
    RGBColor **colors;      // pixel colors.

public:

    // Constructors.
    Image(int hres, int vres);   // initialize blank image of given size.
    Image(const ViewPlane &vp);  // initialize blank image with size of vp.

    // Destructor.
    ~Image();  // free memory.

    // Set pixel color. Convert to integer values.
    void set_pixel(int x, int y, const RGBColor &color);
    void set_pixel(int x, int y, const RGBColor &color, int samples);

    // Write image to file in PPM format.
    void write_ppm(std::string path) const;
};

#endif  // IMAGE_HPP
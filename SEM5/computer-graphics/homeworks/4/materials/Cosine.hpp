#ifndef COSINE_HPP
#define COSINE_HPP

/**
   This file declares the Cosine class which represents a simple cosine
   material.

   It assigns a shade according to the cosine of the angle between the incoming
   ray direction and the normal at the hit point.

   Courtesy Kevin Suffern.
*/

#include "../utilities/RGBColor.hpp"
#include "../utilities/ShadeInfo.hpp"
#include "Material.hpp"

class Cosine : public Material {
protected:

    RGBColor color;  // the color of the material.

public:

    // Constructors.
    Cosine();                           // set color to (0, 0, 0).
    Cosine(float c);                    // set color to (c, c, c).
    Cosine(float r, float g, float b);  // set color to (r, g, b).
    Cosine(const RGBColor &c);          // set color to c.

    // Copy constuctor and assignment operator.
    Cosine(const Cosine &other);
    Cosine &operator=(const Cosine &other);

    /* Returned shade is: color * cos \theta.
       \theta is the angle between the normal at the hit pont and the ray.
       Assuming unit vectors, cos \theta = dot product of normal and -ray.dir.
    */
    virtual RGBColor shade(const ShadeInfo &sinfo) const override;
};

#endif  // COSINE_HPP
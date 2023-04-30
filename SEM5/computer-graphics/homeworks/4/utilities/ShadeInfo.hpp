#ifndef SHADEINFO_HPP
#define SHADEINFO_HPP

/**
   This file declares the class ShadeInfo which contains all the relevant
   information for shading a point.

   Courtesy Kevin Suffern.
*/

#include <cstdlib>

#include "Constants.hpp"
#include "Point3D.hpp"
#include "Ray.hpp"
#include "Vector3D.hpp"

class Material;
class World;

class ShadeInfo {
public:

    bool hit;  // did the ray hit an object?
    Material
           *material_ptr;  // pointer to the material of the nearest hit object.
    Point3D hit_point;     // coordinates of intersection.
    Vector3D     normal;   // normal at hit point.
    Ray          ray;      // the ray that hit.
    int          depth;    // recursion depth.
    float        t;        // ray parameter at hit point.
    const World *w;        // pointer to the world.

public:

    // Constructor.
    ShadeInfo(const World &wr);  // set the world.

    // Copy constructor.
    ShadeInfo(const ShadeInfo &sr) = default;

    // Destructor.
    ~ShadeInfo() = default;
};

#endif  // SHADEINFO_HPP
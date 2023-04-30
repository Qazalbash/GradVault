#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include "../materials/Material.hpp"
#include "../utilities/BBox.hpp"
#include "../utilities/Ray.hpp"
#include "../utilities/ShadeInfo.hpp"

/**
   This file declares the Geometry class which is an abstract class from which
   other concrete geometric objects will inherit.

   Courtesy Kevin Suffern.
*/

#include <string>

class BBox;
class Material;
class Ray;
class ShadeInfo;

class Geometry {
protected:

    Material *material_ptr;  // this object's material.

public:

    // Constructors.
    Geometry();  // sets material_ptr to NULL.

    // Copy constructor and assignment operator.
    Geometry(const Geometry &object)         = default;
    Geometry &operator=(const Geometry &rhs) = default;

    // Destructor.
    virtual ~Geometry() = default;

    // String representation.
    virtual std::string to_string() const = 0;

    // Get/set material.
    Material *get_material() const;
    void      set_material(Material *mPtr);

    // Ray intersection. Set t and sinfo as per intersection with this object.
    virtual bool hit(const Ray &ray, float &t, ShadeInfo &sinfo) const = 0;

    // Get bounding box.
    virtual BBox getBBox() const = 0;
};

#endif  // GEOMETRY_HPP
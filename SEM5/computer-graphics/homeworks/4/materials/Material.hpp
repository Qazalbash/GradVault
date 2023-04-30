#ifndef MATERIAL_HPP
#define MATERIAL_HPP

/**
   This file declares the Material class which is an abstract class for concrete
   materials to inherit from.

   Courtesy Kevin Suffern.
*/

class RGBColor;
class ShadeInfo;

class Material {
public:

    // Constructors.
    Material() = default;  // does nothing.

    // Copy constuctor and assignment operator.
    Material(const Material &other)            = default;
    Material &operator=(const Material &other) = default;

    // Desctructor.
    virtual ~Material() = default;

    // Get color.
    virtual RGBColor shade(const ShadeInfo &sinfo) const = 0;
};

#endif  // MATERIAL_HPP
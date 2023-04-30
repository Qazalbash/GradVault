#ifndef RGBCOLOR_HPP
#define RGBCOLOR_HPP
/**
   This file declares the class RGBColor which represents an RGB color.

   Each of the r, g, b components is clamped within [0,1], i.e. if a component
   becomes less than 0 or greater than 1, e.g. due to an operation, it is made
   equal to 0 or 1 respectively.

  Courtesy Kevin Suffern.
*/

#include <string>

class RGBColor {
public:

    float r, g, b;  // red, green, and blue components in [0,1].

public:

    // Constructors.
    RGBColor();                              // set color to (0, 0, 0).
    RGBColor(float c);                       // set color to (c, c, c).
    RGBColor(float _r, float _g, float _b);  // set color to (_r, _g, _b).

    // Destructor.
    ~RGBColor() = default;

    // Copy constructor and assignment operator.
    RGBColor(const RGBColor &c)              = default;
    RGBColor &operator=(const RGBColor &rhs) = default;

    // String representation.
    std::string to_string() const;

    // Operations.
    RGBColor  operator+(const RGBColor &c) const;  // addition.
    RGBColor &operator+=(const RGBColor &c);       // compound addition.
    RGBColor  operator*(const float a) const;      // multiplication by a float.
    RGBColor &operator*=(const float a);  // compound multiplication by a float.
    RGBColor  operator/(const float a) const;  // division by a float.
    RGBColor &operator/=(const float a);       // compound division by a float.

    RGBColor operator*(
        const RGBColor &c) const;  // component-wise multiplication.
    bool     operator==(const RGBColor &c) const;  // equality.
    RGBColor powc(float p) const;  // raise components to a power.
    float    average() const;      // the average of the components.
};

// Multiplication by a float.
RGBColor operator*(const float a, const RGBColor &c);

#endif  // RGBCOLOR_HPP
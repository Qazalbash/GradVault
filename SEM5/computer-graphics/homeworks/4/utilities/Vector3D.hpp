/**
   This file declares the class Vector3D which represents a 3D vector.

   Courtesy Kevin Suffern.
*/

#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <iostream>
#include <string>

class Point3D;

class Vector3D {
public:

    double x = 0, y = 0, z = 0;  // the components.

public:

    // Constructors.
    Vector3D();                                 // set vector to (0, 0, 0).
    Vector3D(double c);                         // set vector to (c, c, c).
    Vector3D(double _x, double _y, double _z);  // set vector to (_x, _y, _z).
    Vector3D(const Point3D &p);                 // construct from a point.

    // Copy constructor and assignment operator.
    Vector3D(const Vector3D &v)              = default;
    Vector3D &operator=(const Vector3D &rhs) = default;
    // Assign vector from other entitites.
    Vector3D &operator=(const Point3D &rhs);  // from a point.

    // Destructor.
    ~Vector3D() = default;

    // String representation.
    std::string to_string() const;

    // Arithmetic.
    Vector3D  operator-() const;                   // unary minus
    Vector3D  operator+(const Vector3D &v) const;  // addition.
    Vector3D &operator+=(const Vector3D &v);       // compound addition.
    Vector3D  operator-(const Vector3D &v) const;  // subtraction.
    Vector3D &operator-=(const Vector3D &v);       // compound subtraction.

    // Scaling.
    Vector3D operator*(const double a) const;
    Vector3D operator/(const double a) const;
    void     normalize();  // nomalize - convert to a unit vector.

    // Length.
    double length() const;       // length.
    double len_squared() const;  // square of the length.

    // Vector products.
    double   operator*(const Vector3D &b) const;  // dot product.
    Vector3D operator^(const Vector3D &v) const;  // cross product.
};

// Scaling.
Vector3D operator*(const double a, const Vector3D &v);

#endif  // VECTOR_HPP
#include "Cosine.hpp"

#include <cmath>

/**
 * @brief Construct a new Cosine::Cosine object
 *
 */
Cosine::Cosine() { this->color = RGBColor(); }

/**
 * @brief Construct a new Cosine::Cosine object
 *
 * @param c
 */
Cosine::Cosine(float c) { this->color = RGBColor(c); }

/**
 * @brief Construct a new Cosine::Cosine object
 *
 * @param r
 * @param g
 * @param b
 */
Cosine::Cosine(float r, float g, float b) { this->color = RGBColor(r, g, b); }

/**
 * @brief Construct a new Cosine::Cosine object
 *
 * @param c
 */
Cosine::Cosine(const RGBColor &c) { this->color = c; }

/**
 * @brief Construct a new Cosine::Cosine object
 *
 * @param other
 */
Cosine::Cosine(const Cosine &other) { this->color = other.color; }

/**
 * @brief Assignment operator overloaded
 *
 * @param other
 * @return Cosine&
 */
Cosine &Cosine::operator=(const Cosine &other) {
    this->color = other.color;
    return *this;
}

/**
 * @brief Returns the shade
 *
 * @param sinfo
 * @return RGBColor
 *
 * @details Returned shade is: color * cos \theta. \theta is the angle between
 * the normal at the hit pont and the ray. Assuming unit vectors, cos \theta =
 * dot product of normal and -ray.dir.
 */
RGBColor Cosine::shade(const ShadeInfo &sinfo) const {
    Vector3D a = -sinfo.ray.d;
    a.normalize();
    Vector3D b = sinfo.normal;
    b.normalize();
    return this->color * (a * b);
}

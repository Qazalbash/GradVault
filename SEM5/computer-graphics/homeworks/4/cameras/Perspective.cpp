#include "Perspective.hpp"

/**
 * @brief Construct a new Perspective::Perspective object
 *
 */
Perspective::Perspective() : pos(Point3D()) {}

/**
 * @brief Construct a new Perspective::Perspective object
 *
 * @param c
 */
Perspective::Perspective(float c) : pos(Point3D(c)) {}

/**
 * @brief Construct a new Perspective::Perspective object
 *
 * @param x
 * @param y
 * @param z
 */
Perspective::Perspective(float x, float y, float z) : pos(Point3D(x, y, z)) {}

/**
 * @brief Construct a new Perspective::Perspective object
 *
 * @param pt
 */
Perspective::Perspective(const Point3D &pt) : pos(pt) {}

/**
 * @brief Construct a new Perspective::Perspective object
 *
 * @param camera
 */
Perspective::Perspective(const Perspective &camera) : pos(camera.pos) {}

/**
 * @brief Assignment operator overloaded
 *
 * @param other
 * @return Perspective&
 */
Perspective &Perspective::operator=(const Perspective &other) {
    this->pos = other.pos;
    return *this;
}

/**
 * @brief Getter for direction
 *
 * @param p
 * @return Vector3D
 */
Vector3D Perspective::get_direction(const Point3D &p) const {
    Vector3D projVec = p - this->pos;
    return projVec;
}

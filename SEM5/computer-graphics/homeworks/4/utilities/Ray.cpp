#include "Ray.hpp"

/**
 * @brief Construct a new Ray:: Ray object
 *
 */
Ray::Ray() : o(Point3D()), d(Vector3D()), w(1) {}

/**
 * @brief Construct a new Ray:: Ray object
 *
 * @param origin
 * @param dir
 */
Ray::Ray(const Point3D &origin, const Vector3D &dir)
    : o(origin), d(dir), w(1) {}

/**
 * @brief Convert the Ray to a string
 *
 * @return std::string
 */
std::string Ray::to_string() const {
    return this->o.to_string() + " " + this->d.to_string() + " " +
           std::to_string(this->w);
}
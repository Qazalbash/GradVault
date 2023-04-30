#include "Parallel.hpp"

/**
 * @brief Construct a new Parallel::Parallel object parallel to
 * negative z axis
 *
 */
Parallel::Parallel() : dir(Vector3D(0.0, 0.0, -1.0)) {}

/**
 * @brief Construct a new Parallel::Parallel object parallel to the given
 * vector
 *
 * @param c
 */
Parallel::Parallel(float c) {
    this->dir = Vector3D(c);
    dir.normalize();
}

/**
 * @brief Construct a new Parallel::Parallel object parallel to the vector <x,
 * y, z>
 *
 * @param x
 * @param y
 * @param z
 */
Parallel::Parallel(float x, float y, float z) {
    this->dir = Vector3D(x, y, z);
    dir.normalize();
}

/**
 * @brief Construct a new Parallel::Parallel object parallel to d
 *
 * @param d
 */
Parallel::Parallel(const Vector3D &d) {
    this->dir = d;
    dir.normalize();
}

/**
 * @brief Construct a new Parallel::Parallel object parallel to direction of the
 * camera
 *
 * @param camera
 */
Parallel::Parallel(const Parallel &camera) : dir(camera.dir) {}

/**
 * @brief Assignment operator overloaded
 *
 * @param other
 * @return Parallel&
 */
Parallel &Parallel::operator=(const Parallel &other) {
    this->dir = other.dir;
    return *this;
}

/**
 * @brief Getter for direction
 *
 * @details In parallel mode all vertices have the same dir vector.
 *
 * @param p
 * @return Vector3D
 */
Vector3D Parallel::get_direction(const Point3D &p) const { return dir; }

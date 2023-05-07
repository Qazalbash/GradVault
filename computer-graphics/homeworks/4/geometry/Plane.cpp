#include "Plane.hpp"

#include <string>

/**
 * @brief Construct a new Plane::Plane object
 *
 */
Plane::Plane() : a(Point3D()), n(Vector3D(0, 1, 0)) {}

/**
 * @brief Construct a new Plane::Plane object
 *
 * @param pt
 * @param n_
 */
Plane::Plane(const Point3D& pt, const Vector3D& n_) : a(pt), n(n_) {
    n.normalize();
}

/**
 * @brief Construct a new Plane::Plane object
 *
 * @param object
 */
Plane::Plane(const Plane& object) : a(object.a), n(object.n) {}

/**
 * @brief Assignment operator overloaded
 *
 * @param rhs
 * @return Plane&
 */
Plane& Plane::operator=(const Plane& rhs) {
    a = rhs.a;
    n = rhs.n;
    return *this;
}

/**
 * @brief converting this to string
 *
 * @return std::string
 */
std::string Plane::to_string() const {
    return "Plane { a: " + a.to_string() + ", n: " + n.to_string() + " }";
}

/**
 * @brief Checks if ray hitted the plane
 *
 * @param ray
 * @param tmin
 * @param sr
 * @return true
 * @return false
 *
 * @details book listing 3.5
 */
bool Plane::hit(const Ray& ray, float& tmin, ShadeInfo& sr) const {
    double t = (a - ray.o) * n / (ray.d * n);
    if (t > kEpsilon) {
        tmin            = t;
        sr.ray          = ray;
        sr.hit          = true;
        sr.normal       = this->n;
        sr.material_ptr = this->material_ptr;
        sr.hit_point    = ray.o + ray.d * tmin;
        sr.hit_point    = ray.o + t * ray.d;
        sr.t            = tmin;
        return true;
    }
    return false;
}

/**
 * @brief Bounding box of the plane
 *
 * @return BBox
 */
BBox Plane::getBBox() const { return BBox(); }

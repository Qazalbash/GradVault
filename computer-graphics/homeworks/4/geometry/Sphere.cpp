#include "Sphere.hpp"

#include <cmath>

/**
 * @brief Construct a new Sphere::Sphere object
 *
 */
Sphere::Sphere() : c(Point3D()), r(0) {}

/**
 * @brief Construct a new Sphere::Sphere object
 *
 * @param center
 * @param radius
 */
Sphere::Sphere(const Point3D &center, float radius) : c(center), r(radius) {}

/**
 * @brief Construct a new Sphere::Sphere object
 *
 * @param object
 */
Sphere::Sphere(const Sphere &object) : c(object.c), r(object.r) {}

/**
 * @brief Assignment operator overloaded
 *
 * @param rhs
 * @return Sphere&
 */
Sphere &Sphere::operator=(const Sphere &rhs) {
    c = rhs.c;
    r = rhs.r;
    return *this;
}

/**
 * @brief Converts this to string
 *
 * @return std::string
 */
std::string Sphere::to_string() const {
    return "Sphere { c: " + c.to_string() + ", r: " + std::to_string(r) + " }";
}

/**
 * @brief Checks if ray hitted the sphere
 *
 * @param ray
 * @param tmin
 * @param sr
 * @return true
 * @return false
 *
 * @details book listing 3.5
 */
bool Sphere::hit(const Ray &ray, float &tmin, ShadeInfo &sr) const {
    Vector3D temp = ray.o - c;
    double a = ray.d * ray.d, b = 2.0 * temp * ray.d, c_ = temp * temp - r * r,
           disc = b * b - 4.0 * a * c_;

    if (disc < 0.0) return false;

    double e = sqrt(disc), invDenom = 0.5 / a,
           t = -(b + e) * invDenom;  // smaller root

    if (t > kEpsilon) {
        tmin            = t;
        sr.hit          = true;
        sr.material_ptr = this->material_ptr;
        sr.ray          = ray;
        sr.hit_point    = ray.o + ray.d * tmin;
        sr.normal       = sr.hit_point - this->c;
        sr.normal.normalize();
        sr.t = tmin;
        return true;
    }
    t = (e - b) * invDenom;  // larger root
    if (t > kEpsilon) {
        tmin            = t;
        sr.hit          = true;
        sr.material_ptr = this->material_ptr;
        sr.ray          = ray;
        sr.hit_point    = ray.o + ray.d * tmin;
        sr.normal       = sr.hit_point - this->c;
        sr.normal.normalize();
        sr.t = tmin;
        return true;
    }

    return false;
}

/**
 * @brief Bounding box of the sphere
 *
 * @return BBox
 */
BBox Sphere::getBBox() const { return BBox(c - Vector3D(r), c + Vector3D(r)); }

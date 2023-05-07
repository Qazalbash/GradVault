#include "Triangle.hpp"

/**
 * @brief Construct a new Triangle::Triangle object
 *
 */
Triangle::Triangle() : v0(Point3D()), v1(Point3D()), v2(Point3D()) {}

/**
 * @brief Construct a new Triangle::Triangle object
 *
 * @param v0_
 * @param v1_
 * @param v2_
 */
Triangle::Triangle(const Point3D &v0_, const Point3D &v1_, const Point3D &v2_)
    : v0(v0_), v1(v1_), v2(v2_) {}

/**
 * @brief Construct a new Triangle::Triangle object
 *
 * @param object
 */
Triangle::Triangle(const Triangle &object)
    : v0(object.v0), v1(object.v1), v2(object.v2) {}

/**
 * @brief Assignment operator overloaded
 *
 * @param rhs
 * @return Triangle&
 */
Triangle &Triangle::operator=(const Triangle &rhs) {
    v0 = rhs.v0;
    v1 = rhs.v1;
    v2 = rhs.v2;
    return *this;
}

/**
 * @brief Converts this to string
 *
 * @return std::string
 */
std::string Triangle::to_string() const {
    return "Vertex 0:" + v0.to_string() + "\n" + "Vertex 1:" + v1.to_string() +
           "\n" + "Vertex 2:" + v2.to_string() + "\n";
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
 * @details book implementation with slight modification
 */
bool Triangle::hit(const Ray &ray, float &t, ShadeInfo &sr) const {
    float a = v0.x - v1.x, b = v0.x - v2.x, c = ray.d.x, d = v0.x - ray.o.x;
    float e = v0.y - v1.y, f = v0.y - v2.y, g = ray.d.y, h = v0.y - ray.o.y;
    float i = v0.z - v1.z, j = v0.z - v2.z, k = ray.d.z, l = v0.z - ray.o.z;

    float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
    float q = g * i - e * k, s = e * j - f * i;

    float inv_denom = 1.0f / (a * m + b * q + c * s);

    float e1   = d * m - b * n - c * p;
    float beta = e1 * inv_denom;

    if (beta < 0.0f) return false;

    float r     = e * l - h * i;
    float e2    = a * n + d * q + c * r;
    float gamma = e2 * inv_denom;

    if (gamma < 0.0f || beta + gamma > 1.0f) return false;

    float e3 = a * p - b * r + d * s;
    float t_ = e3 * inv_denom;

    if (t_ < kEpsilon) return false;

    t            = t_;
    sr.hit_point = ray.o + t_ * ray.d;
    sr.ray       = ray;
    sr.t         = t_;
    sr.hit       = true;
    sr.normal    = (this->v1 - this->v0) ^ (this->v2 - this->v0);
    sr.normal.normalize();
    sr.material_ptr = this->material_ptr;
    return true;
}

/**
 * @brief Bounding box of the triangle
 *
 * @return BBox
 */
BBox Triangle::getBBox() const {
    Point3D PMIN = min(v0, min(v1, v2));
    Point3D PMAX = max(v0, max(v1, v2));
    return BBox(PMIN, PMAX);
}

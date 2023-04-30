#include "Point3D.hpp"

#include <cmath>
#include <iostream>

/**
 * @brief Construct a new Point 3 D:: Point 3 D object
 *
 */
Point3D::Point3D() : x(0), y(0), z(0) {}
Point3D::Point3D(float c) : x(c), y(c), z(c) {}
Point3D::Point3D(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

/**
 * @brief Convert the Point3D to a string
 *
 * @return std::string
 */
std::string Point3D::to_string() const {
    return std::to_string(this->x) + " " + std::to_string(this->y) + " " +
           std::to_string(this->z);
}

/**
 * @brief unary operator - for Point3D
 *
 * @return Point3D
 */
Point3D Point3D::operator-() const {
    return Point3D(-this->x, -this->y, -this->z);
}

/**
 * @brief binary operator - for Point3D
 *
 * @param p
 * @return Vector3D
 */
Vector3D Point3D::operator-(const Point3D& p) const {
    return Vector3D(this->x - p.x, this->y - p.y, this->z - p.z);
}

/**
 * @brief binary operator + for Point3D and Vector3D
 *
 * @param v
 * @return Point3D
 */
Point3D Point3D::operator+(const Vector3D& v) const {
    return Point3D(this->x + v.x, this->y + v.y, this->z + v.z);
}

/**
 * @brief binary operator - for Point3D and Vector3D
 *
 * @param v
 * @return Point3D
 */
Point3D Point3D::operator-(const Vector3D& v) const {
    return Point3D(this->x - v.x, this->y - v.y, this->z - v.z);
}

/**
 * @brief binary operator * for Point3D and float
 *
 * @param s
 * @return Point3D
 */
Point3D Point3D::operator*(const float s) const {
    return Point3D(this->x * s, this->y * s, this->z * s);
}

/**
 * @brief norm squared of the Point3D
 *
 * @param p
 * @return float
 */
float Point3D::d_squared(const Point3D& p) const {
    return pow((x - p.x), 2) + pow((y - p.y), 2) + pow((z - p.z), 2);
}

/**
 * @brief norm of the Point3D
 *
 * @param p
 * @return float
 */
float Point3D::distance(const Point3D& p) const {
    return sqrt(Point3D::d_squared(p));
}

/**
 * @brief binary operator * for float and Point3D
 *
 * @param a
 * @param pt
 * @return Point3D
 */
Point3D operator*(const float a, const Point3D& pt) {
    return Point3D(a * pt.x, a * pt.y, a * pt.z);
}

/**
 * @brief minimum of two Point3D
 *
 * @param a
 * @param b
 * @return Point3D
 */
Point3D min(const Point3D& a, const Point3D& b) {
    return Point3D(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

/**
 * @brief maximum of two Point3D
 *
 * @param a
 * @param b
 * @return Point3D
 */
Point3D max(const Point3D& a, const Point3D& b) {
    return Point3D(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

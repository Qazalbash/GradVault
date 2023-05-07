#include "Vector3D.hpp"

#include <cmath>

/**
 * @brief Construct a new Vector3D::Vector3D object
 *
 */
Vector3D::Vector3D() : x(0.0f), y(0.0f), z(0.0f) {}

/**
 * @brief Construct a new Vector3D::Vector3D object
 *
 * @param c
 */
Vector3D::Vector3D(double c) : x(c), y(c), z(c) {}

/**
 * @brief Construct a new Vector3D::Vector3D object
 *
 * @param _x
 * @param _y
 * @param _z
 */
Vector3D::Vector3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

/**
 * @brief Convert the Vector3D to a string
 *
 * @return std::string
 */
std::string Vector3D::to_string() const {
    return std::to_string(this->x) + " " + std::to_string(this->y) + " " +
           std::to_string(this->z);
}

/**
 * @brief unary operator - for Vector3D
 *
 * @return Vector3D
 */
Vector3D Vector3D::operator-() const {
    return Vector3D(-this->x, -this->y, -this->z);
}

/**
 * @brief binary operator + for Vector3D
 *
 * @param v
 * @return Vector3D
 */
Vector3D Vector3D::operator+(const Vector3D& v) const {
    return Vector3D(this->x + v.x, this->y + v.y, this->z + v.z);
}

/**
 * @brief Compound assignment operator += for Vector3D
 *
 * @param v
 * @return Vector3D&
 */
Vector3D& Vector3D::operator+=(const Vector3D& v) {
    this->x += v.x;
    this->y += v.y;
    this->z += v.z;
    return (*this);
}

/**
 * @brief binary operator - for Vector3D
 *
 * @param v
 * @return Vector3D
 */
Vector3D Vector3D::operator-(const Vector3D& v) const {
    return Vector3D(this->x - v.x, this->y - v.y, this->z - v.z);
}

/**
 * @brief Compound assignment operator -= for Vector3D
 *
 * @param v
 * @return Vector3D&
 */
Vector3D& Vector3D::operator-=(const Vector3D& v) {
    this->x -= v.x;
    this->y -= v.y;
    this->z -= v.z;
    return (*this);
}

/**
 * @brief binary operator * for Vector3D and double
 *
 * @param a
 * @return Vector3D
 */
Vector3D Vector3D::operator*(const double a) const {
    return Vector3D(this->x * a, this->y * a, this->z * a);
}

/**
 * @brief Compound assignment operator *= for Vector3D and double
 *
 * @param a
 * @return Vector3D&
 */
Vector3D Vector3D::operator/(const double a) const {
    return Vector3D(this->x / a, this->y / a, this->z / a);
}

/**
 * @brief Compound assignment operator /= for Vector3D and double
 *
 * @param a
 * @return Vector3D&
 */
void Vector3D::normalize() {
    double len = Vector3D::length();
    if (len > 0) {
        len = 1 / len;
        this->x *= len;
        this->y *= len;
        this->z *= len;
    }
}

/**
 * @brief Get the length of the Vector3D
 *
 * @return double
 */
double Vector3D::length() const { return sqrt(Vector3D::len_squared()); }

/**
 * @brief Get the squared length of the Vector3D
 *
 * @return double
 */
double Vector3D::len_squared() const {
    return (pow(this->x, 2) + pow(this->y, 2) + pow(this->z, 2));
}

/**
 * @brief Get the dot product of two Vector3D
 *
 * @param b
 * @return double
 */
double Vector3D::operator*(const Vector3D& b) const {
    return (this->x * b.x + this->y * b.y + this->z * b.z);
}

/**
 * @brief Get the cross product of two Vector3D
 *
 * @param v
 * @return Vector3D
 */
Vector3D Vector3D::operator^(const Vector3D& v) const {
    return Vector3D((this->y * v.z) - (this->z * v.y),
                    (this->z * v.x) - (this->x * v.z),
                    (this->x * v.y) - (this->y * v.x));
}

/**
 * @brief Get the distance between two Vector3D
 *
 * @param v
 * @return double
 */
Vector3D operator*(const double a, const Vector3D& v) {
    return Vector3D(a * v.x, a * v.y, a * v.z);
}

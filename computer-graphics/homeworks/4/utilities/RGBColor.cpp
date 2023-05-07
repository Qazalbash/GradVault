#include "RGBColor.hpp"

#include <cmath>

/**
 * @brief Construct a new RGBColor::RGBColor object
 *
 */
RGBColor::RGBColor() : r(0.0f), g(0.0f), b(0.0f) {}

/**
 * @brief Construct a new RGBColor::RGBColor object
 *
 * @param c
 */
RGBColor::RGBColor(float c) : r(c), g(c), b(c) {}

/**
 * @brief Construct a new RGBColor::RGBColor object
 *
 * @param _r
 * @param _g
 * @param _b
 */
RGBColor::RGBColor(float _r, float _g, float _b) : r(_r), g(_g), b(_b) {}

/**
 * @brief Convert the RGBColor to a string
 *
 * @return std::string
 */
std::string RGBColor::to_string() const {
    return std::to_string(this->r) + " " + std::to_string(this->g) + " " +
           std::to_string(this->b);
}

/**
 * @brief binary operator + for RGBColor
 *
 * @param c
 * @return RGBColor
 */
RGBColor RGBColor::operator+(const RGBColor &c) const {
    return RGBColor(this->r + c.r, this->g + c.g, this->b + c.b);
}

/**
 * @brief compund addition for RGBColor
 *
 * @param c
 * @return RGBColor&
 */
RGBColor &RGBColor::operator+=(const RGBColor &c) {
    this->r += c.r;
    this->g += c.g;
    this->b += c.b;
    return *this;
};

/**
 * @brief binary operator * for RGBColor and float
 *
 * @param a
 * @return RGBColor
 */
RGBColor RGBColor::operator*(const float a) const {
    return RGBColor(this->r * a, this->g * a, this->b * a);
}

/**
 * @brief compound multiplication for RGBColor and float
 *
 * @param a
 * @return RGBColor&
 */
RGBColor &RGBColor::operator*=(const float a) {
    this->r *= a;
    this->g *= a;
    this->b *= a;
    return *this;
};

/**
 * @brief binary operator / for RGBColor and float
 *
 * @param a
 * @return RGBColor
 */
RGBColor RGBColor::operator/(const float a) const {
    return RGBColor(this->r / a, this->g / a, this->b / a);
}

/**
 * @brief compound division for RGBColor and float
 *
 * @param a
 * @return RGBColor&
 */
RGBColor &RGBColor::operator/=(const float a) {
    this->r /= a;
    this->g /= a;
    this->b /= a;
    return *this;
}

/**
 * @brief binary operator * for two RGBColor
 *
 * @param c
 * @return RGBColor
 */
RGBColor RGBColor::operator*(const RGBColor &c) const {
    return RGBColor(this->r * c.r, this->g * c.g, this->b * c.b);
}

/**
 * @brief binary operator == for two RGBColor
 *
 * @param c
 * @return true
 * @return false
 */
bool RGBColor::operator==(const RGBColor &c) const {
    return (this->r == c.r && this->g == c.g && this->b == c.b);
}

/**
 * @brief powc function for RGBColor
 *
 * @param p
 * @return RGBColor
 */
RGBColor RGBColor::powc(float p) const {
    return RGBColor(pow(this->r, p), pow(this->g, p), pow(this->b, p));
}

/**
 * @brief average function for RGBColor
 *
 * @return float
 */
float RGBColor::average() const {
    return ((this->r + this->g + this->b) * 0.3333333333f);
}

/**
 * @brief binary operator * for float and RGBColor
 *
 * @param a
 * @param c
 * @return RGBColor
 */
RGBColor operator*(const float a, const RGBColor &c) {
    return RGBColor(a * c.r, a * c.g, a * c.b);
}

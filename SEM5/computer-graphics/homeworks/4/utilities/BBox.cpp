#include "BBox.hpp"

/**
 * @brief Construct a new BBox::BBox object
 *
 * @param min
 * @param max
 */
BBox::BBox(const Point3D& min, const Point3D& max) : pmin(min), pmax(max) {}

/**
 * @brief Converts this to string
 *
 * @return std::string
 */
std::string BBox::to_string() const {
    return this->pmin.to_string() + " " + this->pmax.to_string();
}

/**
 * @brief Checks if the ray hit the bounding box
 *
 * @param ray
 * @param t_enter
 * @param t_exit
 * @return true
 * @return false
 */
bool BBox::hit(const Ray& ray, float& t_enter, float& t_exit) const {
    float txmin, tymin, tzmin, txmax, tymax, tzmax;

    // Checking for entering and exiting on each slab

    if (1 / ray.d.x >= 0) {
        txmin = (this->pmin.x - ray.o.x) * (1 / ray.d.x);
        txmax = (this->pmax.x - ray.o.x) * (1 / ray.d.x);
    } else {
        txmin = (this->pmax.x - ray.o.x) * (1 / ray.d.x);
        txmax = (this->pmin.x - ray.o.x) * (1 / ray.d.x);
    }

    // Checking if the ray is in the negatve region of y
    if (1 / ray.d.y >= 0) {
        tymin = (this->pmin.y - ray.o.y) * (1 / ray.d.y);
        tymax = (this->pmax.y - ray.o.y) * (1 / ray.d.y);
    } else {
        tymin = (this->pmax.y - ray.o.y) * (1 / ray.d.y);
        tymax = (this->pmin.y - ray.o.y) * (1 / ray.d.y);
    }

    // Checking if the ray is in the negatve region of z
    if (1 / ray.d.z >= 0) {
        tzmin = (this->pmin.z - ray.o.z) * (1 / ray.d.z);
        tzmax = (this->pmax.z - ray.o.z) * (1 / ray.d.z);
    } else {
        tzmin = (this->pmax.z - ray.o.z) * (1 / ray.d.z);
        tzmax = (this->pmin.z - ray.o.z) * (1 / ray.d.z);
    }

    // Checking if smallest entry point is less than largest exit point.
    if (max_element(txmin, tymin, tzmin) < min_element(txmax, tymax, tzmax) &&
        min_element(tzmax, tymax, tzmax) > kEpsilon) {
        // If conditions are true, the ray hits bbox. Therefore, define entry
        // and exit
        t_enter = max_element(txmin, tymin, tzmin);
        t_exit  = min_element(txmax, tymax, tzmax);

        return true;
    }

    return false;
};

/**
 * @brief extends the bounding box
 *
 * @param g
 */
void BBox::extend(Geometry* g){};

/**
 * @brief extends the bounding box
 *
 * @param b
 */
void BBox::extend(const BBox& b){};

/**
 * @brief Returns true if p is inside the bounding box
 *
 * @param p
 * @return true
 * @return false
 */
bool BBox::contains(const Point3D& p) {
    // Check if point is within the bounds of the bbox
    return (pmin.x < p.x && pmin.y < p.y && pmin.z < p.z && pmax.x > p.x &&
            pmax.y > p.y && pmax.z > p.z);
};

/**
 * @brief Returns true is if g and this overlaps
 *
 * @param g
 * @return true
 * @return false
 */
bool BBox::overlaps(Geometry* g) {
    // Create const bbox from geometry object
    const BBox& gbbox = g->getBBox();
    // Run BBox method overlap on self through abstraction
    return BBox::overlaps(gbbox);
};

/**
 * @brief Returns true if b and this overlaps
 *
 * @param b
 * @return true
 * @return false
 */
bool BBox::overlaps(const BBox& b) {
    return (pmax.x > b.pmin.x && pmax.y > b.pmin.y && pmax.z > b.pmin.z);
};

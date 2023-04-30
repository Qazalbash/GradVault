#ifndef BBOX_HPP
#define BBOX_HPP

/**
   This file declares BBox class which represents an axis-aligned bounding box
   defined by 2 points containing the min and max values respectively of x, y,
   and z.
*/

#include "../geometry/Geometry.hpp"
#include "Constants.hpp"
#include "Point3D.hpp"
#include "Ray.hpp"

class Geometry;
class Ray;

class BBox {
public:

    // Diagonally opposite points.
    Point3D pmin;  // min coordinates.
    Point3D pmax;  // max coordinates.

public:

    // Constructors.
    BBox() = default;                              // both points at origin.
    BBox(const Point3D& min, const Point3D& max);  // set points.

    // Copy constructor and assignment operator.
    BBox(const BBox& b)              = default;
    BBox& operator=(const BBox& rhs) = default;

    // Destructor.
    ~BBox() = default;

    // String representation.
    std::string to_string() const;

    // Does ray hit bbox? If so, set entering and leaving t values for ray.
    bool hit(const Ray& ray, float& t_enter, float& t_exit) const;

    // Extend this bbox, if necessary, to include g or b.
    void extend(Geometry* g);
    void extend(const BBox& b);

    // Does this BBox contain p? True even when p lies on a boundary.
    bool contains(const Point3D& p);

    // Does this BBox overlap with g or b?
    bool overlaps(Geometry* g);
    bool overlaps(const BBox& b);
};

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

inline float max_element(float a, float b, float c) {
    return MAX(MAX(a, b), c);
}

inline float min_element(float a, float b, float c) {
    return MIN(MIN(a, b), c);
}
#endif  // BBOX_HPP
#ifndef VIEWPLANE_HPP
#define VIEWPLANE_HPP

/**
   This file declares the ViewPlane class which represents a view plane.

   The view plane shares the world's coordinate system. x increases rightward,
   and y upward.

   Courtesy Kevin Suffern.
*/

#include "../utilities/Point3D.hpp"
#include "../utilities/Vector3D.hpp"

class ViewPlane {
public:

    Point3D  top_left;      // top left corner of the view plane.
    Point3D  bottom_right;  // bottom right corner of the view plane.
    Vector3D normal;        // normal to the plane.
    int      hres;          // horizontal resolution
    int      vres;          // vertical resolution

    // Constructors.
    ViewPlane();  // 640 x 480 view plane at (-320, 240)

    // Copy constructor and assignment operator.
    ViewPlane(const ViewPlane &other)          = default;
    ViewPlane &operator=(const ViewPlane &rhs) = default;

    // Get/set resolution.
    int  get_hres() const;
    void set_hres(int);
    int  get_vres() const;
    void set_vres(int);

    // Destructor.
    ~ViewPlane() = default;
};

#endif  // VIEWPLANE_HPP
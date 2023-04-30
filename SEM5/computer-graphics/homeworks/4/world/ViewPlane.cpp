#include "ViewPlane.hpp"

/**
 * @brief Construct a new View Plane:: View Plane object
 *
 */
ViewPlane::ViewPlane() {
    top_left     = Point3D(-320, 240, 0);
    bottom_right = Point3D(top_left.x + hres, top_left.y + vres, 0);
    normal       = Vector3D(0, 0, -1);
    hres         = 640;
    vres         = 480;
    // hres = 2 * 640;
    // vres = 2 * 480;
}

/**
 * @brief Get the horizontal resolution
 *
 * @return int
 */
int ViewPlane::get_hres() const { return hres; }

/**
 * @brief Set the horizontal resolution
 *
 * @param hresVal
 */
void ViewPlane::set_hres(int hresVal) { hres = hresVal; }

/**
 * @brief Get the vertical resolution
 *
 * @return int
 */
int ViewPlane::get_vres() const { return vres; }

/**
 * @brief Set the vertical resolution
 *
 * @param vresVal
 */
void ViewPlane::set_vres(int vresVal) { vres = vresVal; }
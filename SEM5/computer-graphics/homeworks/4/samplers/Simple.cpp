#include "Simple.hpp"

/**
 * @brief Construct a new Simple::Simple object
 *
 * @param c_ptr
 * @param v_ptr
 */
Simple::Simple(Camera *c_ptr, ViewPlane *v_ptr) {
    this->camera_ptr    = c_ptr;
    this->viewplane_ptr = v_ptr;
}

/**
 * @brief Construct a new Simple::Simple object
 *
 * @param camera
 */
Simple::Simple(const Simple &camera) {
    this->camera_ptr    = camera.camera_ptr;
    this->viewplane_ptr = camera.viewplane_ptr;
}

/**
 * @brief Assignment operator overloaded
 *
 * @param other
 * @return Simple&
 */
Simple &Simple::operator=(const Simple &other) {
    this->camera_ptr    = other.camera_ptr;
    this->viewplane_ptr = other.viewplane_ptr;
    return *this;
}

/**
 * @brief Get rays reflected
 *
 * @param px
 * @param py
 * @return std::vector<Ray>
 */
std::vector<Ray> Simple::get_rays(int px, int py) const {
    float pixelH = (viewplane_ptr->bottom_right.y - viewplane_ptr->top_left.y) /
                   viewplane_ptr->vres,
          pixelW = (viewplane_ptr->bottom_right.x - viewplane_ptr->top_left.x) /
                   viewplane_ptr->hres;

    Point3D point;

    point.x = (px + 0.5f) * pixelW + viewplane_ptr->top_left.x;
    point.y = (py + 0.5f) * pixelH + viewplane_ptr->top_left.y;
    point.z = viewplane_ptr->top_left.z;

    Vector3D dir = camera_ptr->get_direction(point);

    Ray              r(point, dir);
    std::vector<Ray> ray;
    ray.push_back(r);
    return ray;
}

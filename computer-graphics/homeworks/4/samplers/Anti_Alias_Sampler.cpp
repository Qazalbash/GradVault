#include "Anti_Alias_Sampler.hpp"

// Returns a random real in [0,1).
inline double random_double() { return rand() / (RAND_MAX + 1.0); }

/**
 * @brief Construct a new Simple::Simple object
 *
 * @param c_ptr
 * @param v_ptr
 */

Anti_Alias_Sampler::Anti_Alias_Sampler(Camera *c_ptr, ViewPlane *v_ptr) {
    this->camera_ptr    = c_ptr;
    this->viewplane_ptr = v_ptr;
}

/**
 * @brief Construct a new Simple::Simple object
 *
 * @param camera
 */
Anti_Alias_Sampler::Anti_Alias_Sampler(const Anti_Alias_Sampler &camera) {
    this->camera_ptr    = camera.camera_ptr;
    this->viewplane_ptr = camera.viewplane_ptr;
}

/**
 * @brief Assignment operator overloaded
 *
 * @param other
 * @return Simple&
 */
Anti_Alias_Sampler &Anti_Alias_Sampler::operator=(
    const Anti_Alias_Sampler &other) {
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
std::vector<Ray> Anti_Alias_Sampler::get_rays(int px, int py) const {
    std::vector<Ray> ray;
    for (int i = 0; i < 100; i++) {
        float pixelH =
                  (viewplane_ptr->bottom_right.y - viewplane_ptr->top_left.y) /
                  viewplane_ptr->vres,
              pixelW =
                  (viewplane_ptr->bottom_right.x - viewplane_ptr->top_left.x) /
                  viewplane_ptr->hres;
        Point3D point;

        point.x = (px + random_double()) * pixelW + viewplane_ptr->top_left.x;
        point.y = (py + random_double()) * pixelH + viewplane_ptr->top_left.y;
        point.z = viewplane_ptr->top_left.z;
        Vector3D dir = camera_ptr->get_direction(point);
        Ray      r(point, dir);
        ray.push_back(r);
    }

    return ray;
}

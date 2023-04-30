#include "../samplers/Sampler.hpp"

/**
 * @brief Construct a new Sampler::Sampler object
 *
 */
Sampler::Sampler() {
    this->camera_ptr    = nullptr;
    this->viewplane_ptr = nullptr;
}

/**
 * @brief Construct a new Sampler::Sampler object
 *
 * @param c_ptr
 * @param v_ptr
 */
Sampler::Sampler(Camera *c_ptr, ViewPlane *v_ptr) {
    this->camera_ptr    = c_ptr;
    this->viewplane_ptr = v_ptr;
}

/**
 * @brief Destroy the Sampler:: Sampler object
 *
 */
Sampler::~Sampler() {
    this->camera_ptr    = nullptr;
    this->viewplane_ptr = nullptr;
}
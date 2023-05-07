#include "World.hpp"
// copied from build function
#include "../cameras/Perspective.hpp"
#include "../geometry/Plane.hpp"
#include "../geometry/Sphere.hpp"
#include "../geometry/Triangle.hpp"
#include "../materials/Cosine.hpp"
#include "../samplers/Simple.hpp"
#include "../utilities/Constants.hpp"
#include "../world/World.hpp"

/**
 * @brief Construct a new World:: World object
 *
 */
World::World() {
    vplane      = ViewPlane();
    bg_color    = RGBColor();
    camera_ptr  = nullptr;
    sampler_ptr = nullptr;
}

/**
 * @brief Destroy the World:: World object
 *
 */
World::~World() {
    delete camera_ptr;
    delete sampler_ptr;
    for (auto &object : geometry) delete object;
}

/**
 * @brief add a geometry object to the world
 *
 * @param geom_ptr
 */
void World::add_geometry(Geometry *geom_ptr) { geometry.push_back(geom_ptr); }

/**
 * @brief set the camera for the world
 *
 * @param c_ptr
 */
void World::set_camera(Camera *c_ptr) { camera_ptr = c_ptr; }

/**
 * @brief hit objects in the world
 *
 * @param ray
 * @return ShadeInfo
 */
ShadeInfo World::hit_objects(const Ray &ray) {
    ShadeInfo sr(*this), sr_temp(*this);
    float     tmin = kHugeValue, t;

    for (auto &object : geometry)
        if (object->hit(ray, t, sr_temp) && (t < tmin)) {
            tmin = t;
            sr   = sr_temp;
        }
    return (sr);
}


/**
  This builds a simple scene that consists of a sphere, a triangle, and a
  plane.
  Parallel viewing is used with a single sample per pixel.
*/
#include <time.h>

#include "../cameras/Perspective.hpp"
#include "../geometry/Plane.hpp"
#include "../geometry/Sphere.hpp"
#include "../geometry/Triangle.hpp"
#include "../materials/Cosine.hpp"
#include "../samplers/Anti_Alias_Sampler.hpp"
#include "../utilities/Constants.hpp"
#include "../world/World.hpp"

void World::build(void) {
    time_t now = time(0);
    srand(now);
    // View plane  .
    vplane.top_left.x     = -10;
    vplane.top_left.y     = 10;
    vplane.top_left.z     = 10;
    vplane.bottom_right.x = 10;
    vplane.bottom_right.y = -10;
    vplane.bottom_right.z = 10;
    vplane.hres           = 400;
    vplane.vres           = 400;

    // Background color.
    bg_color = black;

    // Camera and sampler.
    set_camera(new Perspective(0, 0, 20));
    sampler_ptr = new Anti_Alias_Sampler(camera_ptr, &vplane);

    for (int i = 0; i < 100; i++) {
        int a = rand() % 3;
        int r = rand() % 256;
        int g = rand() % 256;
        int b = rand() % 256;
        if (a == 0) {
            int     x          = rand() % 41 - 20;
            int     y          = rand() % 41 - 20;
            int     c          = rand() % 10;
            Sphere* sphere_ptr = new Sphere(Point3D(x, y, -20), c);
            sphere_ptr->set_material(new Cosine(RGBColor(r, g, b)));
            add_geometry(sphere_ptr);

        } else if (a == 1) {
        } else {
        }
    }
}

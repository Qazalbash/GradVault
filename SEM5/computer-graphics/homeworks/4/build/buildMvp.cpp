//  Credit: Matthew Calligaro <matthewcalligaro@hotmail.com>

#include "../cameras/Parallel.hpp"
#include "../geometry/Plane.hpp"
#include "../geometry/Sphere.hpp"
#include "../geometry/Triangle.hpp"
#include "../materials/Cosine.hpp"
#include "../samplers/Simple.hpp"
#include "../utilities/Constants.hpp"
#include "../world/World.hpp"

void World::build(void) {
    // View plane  .
    int n                 = 10;
    vplane.top_left.x     = -n;
    vplane.top_left.y     = n;
    vplane.top_left.z     = 10;
    vplane.bottom_right.x = n;
    vplane.bottom_right.y = -n;
    vplane.bottom_right.z = 10;
    vplane.hres           = 400;
    vplane.vres           = 400;

    // colors
    RGBColor yellow(1, 1, 0);             // yellow
    RGBColor brown(0.71, 0.40, 0.16);     // brown
    RGBColor darkGreen(0.0, 0.41, 0.41);  // darkGreen
    RGBColor orange(1, 0.75, 0);          // orange
    RGBColor green(0, 0.6, 0.3);          // green
    RGBColor lightGreen(0.65, 1, 0.30);   // light green
    RGBColor darkYellow(0.61, 0.61, 0);   // dark yellow
    RGBColor lightPurple(0.65, 0.3, 1);   // light purple
    RGBColor darkPurple(0.5, 0, 1);       // dark purple
    RGBColor grey(0.3, 0.35, 0.3);        // grey

    // Background color.
    bg_color = grey;

    // Camera and sampler.
    set_camera(new Parallel(0, 0, -1));
    sampler_ptr = new Simple(camera_ptr, &vplane);

    for (int x = -8; x <= 8; x += 2) {
        for (int y = -8; y <= 8; y += 2) {
            Sphere* s = new Sphere(Point3D(x, y, 0), 1);
            s->set_material(new Cosine((x + 8.0) / 16.0, 0, (y + 8.0) / 16.0));
            add_geometry(s);
        }
    }
}

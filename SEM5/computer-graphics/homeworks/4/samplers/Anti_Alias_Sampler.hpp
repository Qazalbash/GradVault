#ifndef SIMPLE_HPP
#define SIMPLE_HPP

/**
   This file declares the Simple class which represents a simple sampler.

   It shoots a single ray of weight 1 through a pixel.

   Courtesy Kevin Suffern.
*/

#include "Sampler.hpp"

class Anti_Alias_Sampler : public Sampler {
protected:

    // add members to cache values to avoid recomputation in get_rays().

public:

    // Constructors.
    Anti_Alias_Sampler() = default;  // initializes members to NULL.
    Anti_Alias_Sampler(Camera *c_ptr, ViewPlane *v_ptr);  // set members.

    // Copy constuctor and assignment operator.
    Anti_Alias_Sampler(const Anti_Alias_Sampler &camera);
    Anti_Alias_Sampler &operator=(const Anti_Alias_Sampler &other);

    // Shoot a ray of weight 1 through the center of the pixel.
    std::vector<Ray> get_rays(int px, int py) const override;
    // std::vector<std::vector<Ray>> get_rays_1(int px, int py) const override;
};

#endif  // SIMPLE_HPP
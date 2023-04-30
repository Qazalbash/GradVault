#ifndef CAMERA_HPP
#define CAMERA_HPP

/**
   This file declares the Camera class which is an abstract class for concrete
   cameras to inherit from. A camera views the world through a view plane.

   Courtesy Kevin Suffern.
*/

class Point3D;
class Vector3D;

class Camera {
public:

    // Constructors.
    Camera() = default;  // does nothing.

    // Copy constuctor and assignment operator.
    Camera(const Camera &camera)           = default;
    Camera &operator=(const Camera &other) = default;

    // Desctructor.
    virtual ~Camera() = default;

    // Get direction of projection for a point.
    virtual Vector3D get_direction(const Point3D &p) const = 0;
};

#endif  // CAMERA_HPP
#include "Geometry.hpp"

/**
 * @brief Construct a new Geometry::Geometry object
 *
 */
Geometry::Geometry() : material_ptr(nullptr) {}

/**
 * @brief Getter for material pointer
 *
 * @return Material*
 */
Material* Geometry::get_material() const { return this->material_ptr; }

/**
 * @brief Setter for material pointer
 *
 * @param mPtr
 */
void Geometry::set_material(Material* mPtr) { this->material_ptr = mPtr; }

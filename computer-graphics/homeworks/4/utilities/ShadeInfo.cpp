#include "ShadeInfo.hpp"

/**
 * @brief Construct a new Shade Info:: Shade Info object
 *
 * @param wr
 */
ShadeInfo::ShadeInfo(const World &wr) : hit(false), t(kHugeValue), w(&wr) {}
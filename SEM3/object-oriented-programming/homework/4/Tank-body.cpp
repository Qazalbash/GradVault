#include "Tank-body.hpp"

TankBody::TankBody(SDL_Renderer *rend, SDL_Texture *ast, int x, int y)
    : Unit(rend, ast) {
    mover = {x, y, 50, 50};
    src   = {0, 13, 427, 281};
}

void TankBody::draw() {
    Unit::draw(src, mover);  // Drawing tank body.
}
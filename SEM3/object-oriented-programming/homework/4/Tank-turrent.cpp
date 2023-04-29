#include "Tank-turrent.hpp"
using namespace std;
#include <iostream>

TankTurrent::TankTurrent(SDL_Renderer *rend, SDL_Texture *ast, int x, int y)
    : Unit(rend, ast) {
    mover     = {x, y, 50, 50};
    src       = {603, 0, 507, 152};
    direction = -10;
}

void TankTurrent::draw() {
    Unit::draw(src, mover);  // Drawing Turrent.
}

void TankTurrent::fire() {
    mover.x += direction;  // Moving turrent forward.
}
void TankTurrent::unfire() {
    mover.x -= direction;  // Moving turrent backward.
}

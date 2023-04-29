#include "Unit.hpp"

class TankBody : public Unit {
    SDL_Rect src, mover;

public:

    TankBody(SDL_Renderer* rend, SDL_Texture* ast, int x, int y);
    void draw();
};
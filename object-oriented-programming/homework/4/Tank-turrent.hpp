#include "Unit.hpp"

class TankTurrent : public Unit {
    SDL_Rect src, mover;
    int      direction;

public:

    TankTurrent(SDL_Renderer *rend, SDL_Texture *ast, int x, int y);
    void draw();
    void fire();
    void unfire();
};
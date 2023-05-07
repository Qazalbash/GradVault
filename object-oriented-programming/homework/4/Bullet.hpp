#include "Unit.hpp"
class Bullet : public Unit {
    SDL_Rect src, mover;
    int      animation, posx;
    bool     complete;
    bool     animate;

public:

    Bullet(SDL_Renderer *rend, SDL_Texture *ast, int x, int y);
    void draw();
};
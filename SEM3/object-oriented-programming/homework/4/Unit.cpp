#include "Unit.hpp"

// Unit class is well implemented, no need to change it

Unit::Unit(SDL_Renderer *rend, SDL_Texture *ast)
    : gRenderer(rend), assets(ast) {}

void Unit::draw(SDL_Rect srcRect, SDL_Rect moverRect) {
    SDL_RenderCopy(gRenderer, assets, &srcRect, &moverRect);
}
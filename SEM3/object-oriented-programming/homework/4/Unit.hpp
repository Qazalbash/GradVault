#pragma once
#include <SDL.h>
// Unit class is well implemented, no need to change it

class Unit {
    SDL_Renderer* gRenderer;
    SDL_Texture*  assets;

public:

    Unit(SDL_Renderer*, SDL_Texture*);
    void draw(SDL_Rect srcRect, SDL_Rect moverRect);
};
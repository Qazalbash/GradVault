#include "egg.hpp"
using namespace std;

Egg::Egg() {
    // src coorinates from assets.png file, they have been found using
    // spritecow.com
    srcRect   = {228, 24, 132, 174};
    moverRect = {30, 40, 80, 90};
}
Egg::Egg(int _x, int _y) {
    // Overloaded constructor defined to insert egg where mouse is clicked
    srcRect   = {228, 24, 132, 174};
    moverRect = {_x, _y, 30, 50};
}

void Egg::drop() {
    // Code to change image of egg when it hits the ground.
    int ground = 450;
    if (moverRect.y > ground) {
        srcRect = {207, 244, 231, 186};
    } else {
        moverRect.y += 10;
    }
}

void Egg::draw(SDL_Renderer* gRenderer, SDL_Texture* assets) {
    SDL_RenderCopy(gRenderer, assets, &srcRect, &moverRect);
    drop();
}
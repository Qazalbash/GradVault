#include "Nest.hpp"
// pigeon implementation will go here.

void Nest::draw(SDL_Renderer *gRenderer, SDL_Texture *assets) {
    SDL_RenderCopy(gRenderer, assets, &srcRect, &moverRect);
    // Code to continously change the image of nest selected from assets.png
    if (n == 0) {
        srcRect.x = 489;
        srcRect.y = 171;
        srcRect.w = 141;
        srcRect.h = 99;
        n++;
    } else if (n == 1) {
        srcRect.x = 494;
        srcRect.y = 308;
        srcRect.w = 141;
        srcRect.h = 107;
        n++;
    } else if (n == 2) {
        srcRect.x = 489;
        srcRect.y = 0;
        srcRect.w = 156;
        srcRect.h = 145;
        n         = 0;
    }
}
Nest::Nest() {
    // src coorinates from assets.png file, they have been found using
    // spritecow.com
    srcRect = {489, 0, 156, 145};

    // it will display nest on x = 30, y = 40 location, the size of nest is 50
    // width, 60 height
    moverRect = {30, 40, 100, 100};
}
Nest::Nest(int _x, int _y) {
    // Another constructor to create object at the position of the click.
    srcRect   = {489, 0, 156, 145};
    moverRect = {_x, _y, 100, 80};
}
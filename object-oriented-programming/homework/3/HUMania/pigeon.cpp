#include "pigeon.hpp"
using namespace std;
// pigeon implementation will go here.

void Pigeon::draw(SDL_Renderer *gRenderer, SDL_Texture *assets) {
    SDL_RenderCopy(gRenderer, assets, &srcRect, &moverRect);
    fly();

    // continously changing coordinates used to snip image of pigeons from
    // assets.png
    if (i == 0) {
        srcRect.x = 0;
        srcRect.y = 164;
        srcRect.w = 153;
        srcRect.h = 83;
        i++;
    } else if (i == 1) {
        srcRect.x = 2;
        srcRect.y = 288;
        srcRect.w = 159;
        srcRect.h = 123;
        i++;
    } else if (i == 2) {
        srcRect.x = 0;
        srcRect.y = 0;
        srcRect.w = 160;
        srcRect.h = 133;
        i         = 0;
    }
}
void Pigeon::fly() {
    // speed with which the pigeon flies
    int speed = 10;

    // As the pigeon reaches the end of the screen it changes direction
    if (direction == true) {
        moverRect.x += speed;
        if ((moverRect.x + moverRect.w) > 1000) {
            direction = false;
        }
    } else if (direction == false) {
        moverRect.x -= speed;
        if (moverRect.x <= 0) {
            direction = true;
        }
    }
}

Pigeon::Pigeon() {
    // src coorinates from assets.png file, they have been found using
    // spritecow.com
    srcRect = {0, 0, 160, 133};
    // it will display pigeon on x = 30, y = 40 location, the size of pigeon is
    // 100 width, 100 height
    moverRect = {30, 40, 100, 100};
}
Pigeon::Pigeon(int _x, int _y) {
    // Second constructor to place pigeon where the mouse is clicked.
    srcRect   = {0, 0, 160, 133};
    moverRect = {_x, _y, 100, 100};
}
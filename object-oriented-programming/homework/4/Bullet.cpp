#include "Bullet.hpp"

Bullet::Bullet(SDL_Renderer *rend, SDL_Texture *ast, int x, int y)
    : Unit(rend, ast) {
    mover     = {x + 49, y, 50, 50};
    src       = {39, 394, 92, 95};
    animation = 2;
    complete  = false;  // For checking if bullet has reached end.
    animate   = true;   // To stop drawing.
}

void Bullet::draw() {
    if (animate ==
        true)  // Once our bullet reaches the end we will not draw it.
    {
        Unit::draw(src, mover);
        if (mover.x + mover.w >= 1000 and
            animation > 8)  // To cause explostion at the beginning and end.
        {
            mover.x -= 30;
            mover.y -= 24;
            mover.w   = 50;
            mover.h   = 50;
            src       = {39, 362, 162, 165};
            animation = 1;
            complete  = true;
        }

        switch (animation)  // Case 1 till 7 is for perfect explosion animation
                            // and the rest is for animating bullet.
        {
            case 1:
                src = {39, 394, 92, 95};
                break;
            case 2:
                src = {189, 373, 133, 141};
                break;
            case 3:
                src = {339, 362, 162, 165};
                break;
            case 4:
                src = {506, 362, 162, 165};
                break;
            case 5:
                src = {681, 362, 154, 165};
                break;
            case 6:
                src = {847, 362, 154, 165};
                break;
            case 7:
                src = {1010, 362, 158, 165};
                break;
            case 8:
                if (complete == true) {
                    animate = false;  // To stop animating after explosion.
                    break;
                }
                src     = {616, 201, 302, 96};
                mover.w = 30;
                mover.h = 10;
                mover.x += 30;
                mover.y += 24;
                break;
            default:
                mover.x += 10;
        }
        animation += 1;
    }
}

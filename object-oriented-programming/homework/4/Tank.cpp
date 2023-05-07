#include "Tank.hpp"

Tank::Tank(SDL_Renderer *rend, SDL_Texture *ast, int x, int y)
    : body(rend, ast, x - 15, y), turrent(rend, ast, x, y), posx(x), posy(y) {}

void Tank::draw() {
    // Drawing tank objects.
    body.draw();
    turrent.draw();
    for (auto *bullet : bullets) {  // Drawing bullets.
        bullet->draw();
    }
}

void Tank::fire(SDL_Renderer *rend, SDL_Texture *ast) {
    turrent.fire();  // Animating turrent.
    bullets.push_back(
        new Bullet(rend, ast, posx, posy));  // Creating new bullet.
}

void Tank::unfire() {
    turrent.unfire();  // Animating turrent.
}

Tank::~Tank() {
    for (auto *bullet : bullets)  // Drawing bullets.
        delete bullet;
    bullets.clear();  // Clearing bullets list from memory.
}
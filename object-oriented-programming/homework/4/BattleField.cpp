#include "BattleField.hpp"

#include <iostream>

// In my code I've added explosions at both creation of bullet and collision
// with wall to make it realistic.
void BattleField::drawObjects() {
    if (fireall ==
        true)  // Will call functions to create explosion and bullets.
    {
        for (auto *it : tanks) {
            it->fire(gRenderer, assets);
        }
    }
    for (auto *tank : tanks)  // Drawing tanks.
    {
        tank->draw();
    }
    if (fireall == true)  // Will call function to make sure turrent moves back
                          // to original position.
    {
        for (auto *it : tanks) {
            it->unfire();
        }
        fireall = false;
    }
}

void BattleField::createObject(int x, int y) {
    tanks.push_back(
        new Tank(gRenderer, assets, x,
                 y));  // Adding new tank object dynamically to list.

    std::cout << "Mouse clicked at: " << x << " -- " << y << std::endl;
}

BattleField::BattleField(SDL_Renderer *renderer, SDL_Texture *asst)
    : gRenderer(renderer), assets(asst) {
    fireall = false;
}

BattleField::~BattleField() {
    for (auto *tank : tanks)  // Deleting Tanks.
        delete tank;
    tanks.clear();  // Clearing tanks from memory.
}

void BattleField::fire() {
    cout << "F key is pressed" << endl;
    fireall = true;  // To make sure all tanks fire.
}

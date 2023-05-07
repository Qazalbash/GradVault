#include "HUMania.hpp"

#include <iostream>
using namespace std;

void HUMania::drawObjects() {
    // Calling the draw functions of all the objects({Pigeons, Eggs, Nests) here
    for (int i = 0; i < pigeons.size(); i++) {
        pigeons[i].draw(gRenderer, assets);
    }
    for (int i = 0; i < nests.size(); i++) {
        nests[i].draw(gRenderer, assets);
    }
    for (int i = 0; i < eggs.size(); i++) {
        eggs[i].draw(gRenderer, assets);
    }
}

void HUMania::createObject(int x, int y) {
    cout << "Mouse clicked at: " << x << " -- " << y << endl;
    // Randomly inserting any of the three objects.-- appending it in their
    // respective vectors.
    int obj_generator = rand() % 3 + 1;

    switch (obj_generator) {
        case 1:
            pigeons.push_back(Pigeon(x, y));
            break;
        case 2:
            nests.push_back(Nest(x, y));
            break;
        case 3:
            eggs.push_back(Egg(x, y));
            break;
    }
}

HUMania::HUMania(SDL_Renderer *renderer, SDL_Texture *asst)
    : gRenderer(renderer), assets(asst) {}

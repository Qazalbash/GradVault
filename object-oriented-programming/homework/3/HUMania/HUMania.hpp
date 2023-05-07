#include <SDL.h>

#include "MyVector.cpp"
#include "Nest.hpp"
#include "egg.hpp"
#include "pigeon.hpp"
using namespace std;

class HUMania {
    SDL_Renderer *gRenderer;
    SDL_Texture  *assets;

    // Creating vectors of pigeons, eggs, and nests

    MyVector<Pigeon> pigeons;
    MyVector<Nest>   nests;
    MyVector<Egg>    eggs;

public:

    HUMania(SDL_Renderer *, SDL_Texture *);
    void drawObjects();
    void createObject(int, int);
};
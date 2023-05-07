#include <SDL.h>

#include <list>

#include "Tank.hpp"

using namespace std;
class BattleField {
    SDL_Renderer *gRenderer;
    SDL_Texture  *assets;
    list<Tank *>  tanks;
    bool          fireall;

public:

    BattleField(SDL_Renderer *, SDL_Texture *);
    void drawObjects();
    void createObject(int, int);
    void fire();
    ~BattleField();
};
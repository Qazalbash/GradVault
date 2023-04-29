// Make composition of TankBody and TankTurret objects in Tank class
#include <iostream>
#include <list>

#include "Bullet.hpp"
#include "Tank-body.hpp"
#include "Tank-turrent.hpp"
using namespace std;
class Tank {
    TankBody       body;
    TankTurrent    turrent;
    int            posx;
    int            posy;
    list<Bullet *> bullets;

public:

    Tank(SDL_Renderer *rend, SDL_Texture *ast, int x, int y);
    void draw();
    void fire(SDL_Renderer *rend, SDL_Texture *ast);
    void unfire();
    ~Tank();
};
#include <SDL.h>
using namespace std;

class Egg {
    SDL_Rect srcRect, moverRect;

public:

    // add the fly function here as well.
    void draw(SDL_Renderer*, SDL_Texture* assets);
    void drop();
    Egg();
    // overloaded constructor added here.
    Egg(int, int);
};

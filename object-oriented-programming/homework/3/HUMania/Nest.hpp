#include <SDL.h>
using namespace std;

class Nest {
    SDL_Rect srcRect, moverRect;
    int      n = 0;

public:

    // add the fly function here as well.
    void draw(SDL_Renderer*, SDL_Texture* assets);
    Nest();
    // overloaded constructor
    Nest(int, int);
};

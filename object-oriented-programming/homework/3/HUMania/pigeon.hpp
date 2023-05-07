#include <SDL.h>
using namespace std;

class Pigeon {
    SDL_Rect srcRect, moverRect;
    // private variables initialized.
    int  i         = 0;
    bool direction = true;

public:

    // add the fly function here as well.
    void draw(SDL_Renderer *, SDL_Texture *assets);
    void fly();
    Pigeon();
    // overloaded constructor added here.
    Pigeon(int, int);
};

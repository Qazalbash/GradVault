#include <SDL.h>
#include <SDL_image.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <string>

class Game {
    // Screen dimension constants
    const int SCREEN_WIDTH  = 1000;
    const int SCREEN_HEIGHT = 600;

    // The window we'll be rendering to
    SDL_Window* gWindow = NULL;

    // The window renderer
    SDL_Renderer* gRenderer = NULL;

    // Current displayed texture
    SDL_Texture* gTexture = NULL;
    // global reference to png image sheets
    SDL_Texture* assets = NULL;

public:

    bool         init();
    bool         loadMedia();
    void         close();
    SDL_Texture* loadTexture(std::string path);
    void         run();
};

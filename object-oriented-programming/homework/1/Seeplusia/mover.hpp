#include <SDL.h>

void moveSouth();
void moveNorth();
void moveEast();
void moveWest();
void status(SDL_Renderer *gRenderer, SDL_Texture *assets);
void update(SDL_Renderer *gRenderer, SDL_Texture *assets);
void moveWarrior(SDL_Renderer *gRenderer, SDL_Texture *assets, SDL_Keycode key);
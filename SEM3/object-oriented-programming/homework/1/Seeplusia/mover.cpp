#include "mover.hpp"

#include "seeplusia.hpp"

const int xJump = 200, yJump = 160;

SDL_Rect bg           = {0, 0, 800, 600};
SDL_Rect bgSrc        = {0, 350, 770, 600};
SDL_Rect warriorSrc   = {340, 120, 240, 220};
SDL_Rect warriorMover = {430, 160, 100, 100};

// Screen dimension constants
const int SCREEN_WIDTH  = 800;
const int SCREEN_HEIGHT = 600;

void moveSouth() {
    if (warriorMover.y + yJump < SCREEN_HEIGHT) warriorMover.y += yJump;
}
void moveNorth() {
    if (warriorMover.y - yJump > 0)
        warriorMover.y -= yJump;
    else
        warriorMover.y = 0;
}
void moveEast() {
    if (warriorMover.x + xJump < SCREEN_WIDTH) warriorMover.x += xJump;
}
void moveWest() {
    if (warriorMover.x - xJump > 0) warriorMover.x -= xJump;
}

void status(SDL_Renderer *gRenderer, SDL_Texture *assets) {
    SDL_Rect statusSrc, statusMover;

    // Apples
    statusSrc   = {634, 38, 122, 139};
    statusMover = {30, 420, 20, 20};

    for (int i = 0; i < applesLeft; i++) {
        if (i % 10 == 0) {  // go to new line
            statusMover.x = 30;
            statusMover.y += 30;
        }
        SDL_RenderCopy(gRenderer, assets, &statusSrc, &statusMover);
        statusMover.x += 20;
    }

    // Diamonds
    statusSrc   = {635, 215, 145, 120};
    statusMover = {30, 550, 30, 30};

    for (int i = 0; i < nCrystalsFound; i++) {
        SDL_RenderCopy(gRenderer, assets, &statusSrc, &statusMover);
        statusMover.x += 30;
    }

    statusMover = {400, 500, 300, 60};
    if (gameState == "Running") {
        statusSrc = {0, 0, 585, 54};
        SDL_RenderCopy(gRenderer, assets, &statusSrc, &statusMover);
    } else if (gameState == "Won") {
        statusSrc = {0, 270, 225, 60};
        SDL_RenderCopy(gRenderer, assets, &statusSrc, &statusMover);
        cout << "Game Won";
    } else if (gameState == "Lost") {
        statusSrc = {0, 136, 275, 55};
        SDL_RenderCopy(gRenderer, assets, &statusSrc, &statusMover);
    }
}

void moveWarrior(SDL_Renderer *gRenderer, SDL_Texture *assets,
                 SDL_Keycode key) {
    if (key == SDLK_UP) {
        makeMove("North");
    } else if (key == SDLK_DOWN) {
        makeMove("South");
    } else if (key == SDLK_RIGHT) {
        makeMove("East");
    } else if (key == SDLK_LEFT) {
        makeMove("West");
    } else {
        std::cout << "hello";
    }
    update(gRenderer, assets);
}

void update(SDL_Renderer *gRenderer, SDL_Texture *assets) {
    SDL_RenderClear(gRenderer);
    SDL_RenderCopy(gRenderer, assets, &bgSrc, &bg);
    status(gRenderer, assets);
    SDL_RenderCopy(gRenderer, assets, &warriorSrc, &warriorMover);
    SDL_RenderPresent(gRenderer);
    SDL_Delay(5);
}
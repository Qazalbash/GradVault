#include <vector>

#include "game.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    Game game;
    // Test t1;
    // for(int i=0;i<5;i++)
    //     t1.add(i);
    srand(time(NULL));
    if (!game.init())
    {
        printf("Failed to initialize!\n");
        return 0;
    }
    // Load media
    if (!game.loadMedia())
    {
        printf("Failed to load media!\n");
        return 0;
    }
    game.run();
    game.close();

    return 0;
}
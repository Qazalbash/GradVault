#include <stdlib.h>
#include <time.h>

#include <fstream>
#include <iostream>

// Structure to keep track of various locations
struct Point
{
    int x, y;
    Point(const int &_x, const int &_y) : x(_x), y(_y) {}
};

// Structure for the Player object
struct Player
{
    int food, health;
    bool alive;
    int x, y;

    Player() : food(64), health(10), alive(true) {}

    void loseHealth()
    {
        if (health > 0)
            --health;
        if (health == 0)
            alive = false;
    }

    void gainHealth()
    {
        if (health < 10)
            ++health;
    }
};

int random(const int &size) { return rand() % size; }

typedef const int cint; // Google typedef to see what this means
typedef const Point cPoint;

char *CreateDungeon(int, int, Point &, Point &); // Creates the dungeon
void Traversal(char *, Point &, cPoint &, cint,
               cint);       // Used for moving inside dungeon
void Combat(Player &, int); // Used for simulating combat
void TrapStatements();      // 3 statements that show at random when the player
                            // activates a trap
void FoodStatements();      // 3 statements that show at random when the player
                            // finds food
void HitStatements();       // 3 statements that show at random when the player hits
                            // enemies
void GetHitStatements();    // 3 statements that show at random when the player
                            // gets hit
void titleScreen();
void lostScreen();
void winScreen();

int main()
{
    srand(time(0));
    titleScreen();

    int width, height;
    std::cout << "Enter width: ";
    std::cin >> width;
    std::cout << std::endl;

    std::cout << "Enter height: ";
    std::cin >> height;
    std::cout << std::endl;

    Point startPoint(2, 3);
    Point exitPoint(height - 3, width - 5);

    char *dungeon = nullptr;

    dungeon = CreateDungeon(width, height, startPoint, exitPoint);

    std::cout
        << "After being captured by the raid of some robbers on your carvan,"
        << std::endl;
    std::cout << "you find yourself in a dark dungeon. With nothing but your"
              << std::endl;
    std::cout << "wits, you choose to take a step..." << std::endl;

    Traversal(dungeon, startPoint, exitPoint, width, height);

    char bye;
    std::cout << "Press any key and then press enter to exit.....   ";
    std::cin >> bye;

    return 0;
}

void TrapStatements()
{
    std::cout << std::endl;
    int randNo = rand() % 3;
    switch (randNo)
    {
    case 1:
        std::cout << "Oh no you fell in a trap!" << std::endl;
        break;
    case 2:
        std::cout << "Trapped" << std::endl;
        break;
    case 3:
        std::cout << "You are Stuck in the trap" << std::endl;
        break;
    }
}

void FoodStatements()
{
    // cout<<std::endl;
    int randNo = rand() % 3;
    switch (randNo)
    {
    case 0:
        std::cout << "You found a plate of Biryani! :)    ";
        break;
    case 1:
        std::cout << "You found a plate of Nihari! :)     ";
        break;
    case 2:
        std::cout << "You found a Bun-Kabab! :)           ";
        break;
    }
}

void HitStatements()
{
    std::cout << std::endl;
    int randNo = rand() % 3;
    switch (randNo)
    {
    case 0:
        std::cout << "You made an Epic move !! -->" << std::endl;
        break;
    case 1:
        std::cout << "Fantastic Shot !! -->" << std::endl;
        break;
    case 2:
        std::cout << "You Yeeted an Enemy !! -->" << std::endl;
        break;
    }
}

void GetHitStatements()
{
    std::cout << std::endl;
    int randNo = rand() % 3;
    switch (randNo)
    {
    case 0:
        std::cout
            << "                                         <--YOU GOT HIT !!"
            << std::endl;
        break;
    case 1:
        std::cout << "                                         <--THE "
                     "ENEMY HIT YOU !!"
                  << std::endl;
        break;
    case 2:
        std::cout << "                                         <--BONK !!"
                  << std::endl;
        break;
    }
}

void Traversal(char *dungeon, Point &startPoint, cPoint &exitPoint, cint width,
               cint height)
{
    Player P;
    P.x = startPoint.x;
    P.y = startPoint.y;
    char direction;
    int location, delta_x, delta_y;
    std::cout << std::endl;
    std::cout << "***************" << std::endl;
    std::cout << "* HOW TO PLAY *" << std::endl;
    std::cout << "*-------------*" << std::endl;
    std::cout << "* Down => D   *" << std::endl
              << "* Left => L   *" << std::endl
              << "* Right => R  *" << std::endl
              << "* Up => U     *" << std::endl
              << "* Exit => X   *" << std::endl;
    std::cout << "***************" << std::endl;
    std::cout << std::endl;

    while (true)
    {
        std::cout << std::endl;
        std::cout << "Food: " << P.food << std::endl
                  << "Health: " << P.health << std::endl;

        std::cout << "Enter the direction: ";
        std::cin >> direction;

        if (direction == 'D' || direction == 'd')
        {
            location = (P.y + 1) * width + P.x;
        }
        else if (direction == 'L' || direction == 'l')
        {
            location = P.y * width + P.x - 1;
        }
        else if (direction == 'R' || direction == 'r')
        {
            location = P.y * width + P.x + 1;
        }
        else if (direction == 'U' || direction == 'u')
        {
            location = (P.y - 1) * width + P.x;
        }
        else if (direction == 'X' || direction == 'x')
        {
            break;
        }

        delta_x = (location == P.y * width + P.x + 1) -
                  (location == P.y * width + P.x - 1); // branchless statement
        delta_y =
            (location == (P.y + 1) * width + P.x) -
            (location == (P.y - 1) * width + P.x); // branchless statement

        if (dungeon[location] == 'H')
        {
            if (P.health == 10)
            {
                std::cout << "< PLAYER HEALTH AT MAX - NO HEALTH GAINED >"
                          << std::endl;
            }
            else
            {
                std::cout << "< YOU GAINED A HEALTH : + 1 >" << std::endl;
                P.gainHealth();
                *(dungeon + P.y * width + P.x) = ' ';
                P.x += delta_x;
                P.y += delta_y;
                *(dungeon + P.y * width + P.x) = 'P';
            }
        }
        else if (dungeon[location] == 'T')
        {
            *(dungeon + P.y * width + P.x) = ' ';
            P.loseHealth();
            TrapStatements();
            P.x += delta_x;
            P.y += delta_y;
            *(dungeon + P.y * width + P.x) = 'P';
        }
        else if (dungeon[location] == 'F')
        {
            int food = 4 + rand() % 5;
            std::cout << std::endl
                      << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
            std::cout << "$     YOU GOT " << food << " DAYS OF FOOD          $"
                      << std::endl;
            std::cout << "---------------------------------------" << std::endl;
            for (int i = 0; i < food; i++)
            {
                std::cout << "$ ";
                FoodStatements();
                std::cout << "$" << std::endl;
            }
            std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;

            P.food += food;
            *(dungeon + P.y * width + P.x) = ' ';
            P.x += delta_x;
            P.y += delta_y;
            *(dungeon + P.y * width + P.x) = 'P';
        }
        else if (dungeon[location] == 'X')
        {
            winScreen();
            P.x += delta_x;
            P.y += delta_y;
            *(dungeon + P.y * width + P.x) = 'P';
            P.alive = false;
        }
        else if (dungeon[location] == 'W')
        {
            std::cout << std::endl
                      << "< YOU BUMPED INTO A WALL >" << std::endl;
        }
        else if (dungeon[location] == 'E')
        {
            int enemies = rand() % 4 + 2;
            Combat(P, enemies);
            *(dungeon + P.y * width + P.x) = ' ';
            P.x += delta_x;
            P.y += delta_y;
            *(dungeon + P.y * width + P.x) = 'P';
        }
        else
        {
            *(dungeon + P.y * width + P.x) = ' ';
            P.x += delta_x;
            P.y += delta_y;
            *(dungeon + P.y * width + P.x) = 'P';
        }

        if (P.food == 0)
        {
            std::cout << "< YOU RAN OUT OF FOOD AND STARVED TO DEATH>"
                      << std::endl;
            lostScreen();
            break;
        }
        if (!P.alive)
        {
            break;
        }
        P.food--;
    }
}

void Combat(Player &player, int enemies)
{
    int randomNum;
    std::cout << "< YOU ENCOUNTERED " << enemies << " ENEMIES!!! >" << std::endl
              << " . . . " << std::endl;
    std::cout
        << "<<<<<<<<<<<<<<<<<<<<< ! PREPARE TO FIGHT ! >>>>>>>>>>>>>>>>>>>>>"
        << std::endl;
    while (player.alive == true && enemies > 0)
    {
        std::cout << std::endl;
        std::cout << "Player Health : " << player.health
                  << "                         "
                  << "Enemies Remaining : " << enemies << std::endl;
        std::cout
            << "###############################################################"
            << std::endl;
        std::cout << std::endl;
        randomNum = random(100);
        if (randomNum <= 30)
        {
            HitStatements();
            enemies--;
        }
        else
        {
            std::cout << "~You missed the attack~..." << std::endl;
        }
        randomNum = random(100);
        if (randomNum <= 10)
        {
            GetHitStatements();
            player.loseHealth();
        }
        else
        {
            std::cout << "                                         ~The enemy "
                         "missed the attack~..."
                      << std::endl;
        }
    }
    std::cout << "Player Health : " << player.health
              << "                         "
              << "Enemies Remaining : " << enemies << std::endl;
    std::cout
        << "###############################################################"
        << std::endl;

    if (player.alive == true && enemies < 1)
    {
        std::cout << "< YOU DEFEATED ALL THE ENEMIES >" << std::endl;
        std::cout << "You can move now . . . ." << std::endl;
    }
    else if (player.alive == false && enemies > 0)
    {
        std::cout << "< YOU HAVE BEEN DEFEATED >" << std::endl;
        lostScreen();
    }
}

char *CreateDungeon(int width, int height, Point &startPoint,
                    Point &exitPoint)
{
    char *dungeon = new char[width * height];
    for (int i = 0; i < (height * width); i += width)
    {
        dungeon[i] = 'W';
        dungeon[i + (width - 1)] = 'W';
    }

    for (int i = 0; i < width; i++)
    {
        dungeon[i] = 'W';
        dungeon[(height - 1) * width + i] = 'W';
    }

    int randomNumber;
    int j = 0;
    for (int i = 0; i < height; i++)
    {
        for (; j < (width * (1 + i)); j++)
        {
            if (dungeon[j] != 'W')
            {
                randomNumber = random(100);
                if (randomNumber <= 20)
                {
                    randomNumber = random(100);

                    if (randomNumber <= 15)
                    {
                        dungeon[j] = 'E';
                    }
                    else if (randomNumber > 15 && randomNumber <= 30)
                    {
                        dungeon[j] = 'H';
                    }
                    else if (randomNumber > 30 && randomNumber <= 45)
                    {
                        dungeon[j] = 'T';
                    }
                    else if (randomNumber > 45 && randomNumber <= 60)
                    {
                        dungeon[j] = 'F';
                    }
                    else
                    {
                        dungeon[j] = 'W';
                    }
                }
                else
                {
                    dungeon[j] = ' ';
                }
            }
        }
    }
    dungeon[startPoint.y * width + startPoint.x] = 'P';
    dungeon[exitPoint.y * width + exitPoint.x] = 'X';

    return dungeon;
}

void titleScreen()
{
    std::cout << R"(
 THE       
'||''|.       |     '||''|.   '||'  |'                                           
 ||   ||     |||     ||   ||   || .'                                             
 ||    ||   |  ||    ||''|'    ||'|.                                             
 ||    ||  .''''|.   ||   |.   ||  ||                                            
.||...|'  .|.  .||. .||.  '|' .||.  ||.            

         '||''|.   '||'  '|' '|.   '|'  ..|'''.|  '||''''|   ..|''||   '|.   '|' 
          ||   ||   ||    |   |'|   |  .|'     '   ||  .    .|'    ||   |'|   |  
          ||    ||  ||    |   | '|. |  ||    ....  ||''|    ||      ||  | '|. |  
          ||    ||  ||    |   |   |||  '|.    ||   ||       '|.     ||  |   |||  
         .||...|'    '|..'   .|.   '|   ''|...'|  .||.....|  ''|...|'  .|.   '|  
                                                                                 
                                                                                 
                         
                                                                                 
   )" << '\n';
}

void lostScreen()
{
    std::cout << R"(

 _      _   ___   .     .      .       ___     _____  _______
  `.   /  .'   `. /     /      /     .'   `.  (      '   /   
    `./   |     | |     |      |     |     |   `--.      |   
    ,'    |     | |     |      |     |     |      |      |   
 _-'       `.__.'  `._.'       /---/  `.__.' \___.'      /   
 ------------------------GAME OVER------------------------                                                            
                                                             
                                                             
                                          
                                                                                                                                     
   )" << '\n';
}

void winScreen()
{
    std::cout << R"(
 **    **   *******   **     **     **       **   *******   ****     **
//**  **   **/////** /**    /**    /**      /**  **/////** /**/**   /**
 //****   **     //**/**    /**    /**   *  /** **     //**/**//**  /**
  //**   /**      /**/**    /**    /**  *** /**/**      /**/** //** /**
   /**   /**      /**/**    /**    /** **/**/**/**      /**/**  //**/**
   /**   //**     ** /**    /**    /**** //****//**     ** /**   //****
   /**    //*******  //*******     /**/   ///** //*******  /**    //***
   //      ///////    ///////      //       //   ///////   //      /// 
 ----------------YOU MANAGED TO ESCAPE THE DUNGEON--------------------                                                                      
                                                                       
                                                                       
                                              
                                                                       
                                                                       
   )" << '\n';
}
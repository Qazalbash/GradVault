#include <iostream>
#include <string>

void Game_state(int days, std::string location, int c1, int c2, int c3) {
    std::cout << "Days: " << days << std::endl
              << "Location: " << location << std::endl;  // Game state
    std::cout << "Number of crystals collected: " << c1 + c2 + c3 << std::endl;
    std::cout << "In which direction you want to move? ";
}
int Crystals(std::string location, int c1, int c2, int c3) {
    if (c1 == 0 &&
        location.compare("Marsh of the Undead") == 0) {  // 1st Crystal
        std::cout << "Crystal Collected!" << std::endl << std::endl;
        return 1;
    } else if (c2 == 0 &&
               location.compare("Werewolf Hill") == 0) {  // 2nd Crystal
        std::cout << "Crystal Collected!" << std::endl << std::endl;
        return 2;
    } else if (c3 == 0 &&
               location.compare("Elven Waterfall") == 0) {  // 3rd Crystal
        std::cout << "Crystal Collected!" << std::endl << std::endl;
        return 3;
    } else {
        return 4;
    }
}
int Game_End(std::string location, int days) {
    if (location.compare("Sands of Quick") ==
        0) {  // Game over when we go to sands of quick
        std::cout << "You are struck in Sands of Quick" << std::endl
                  << "Game Over!" << std::endl;
        return 1;
    } else if (location.compare("Wizard's Castle") == 0) {
        std::cout << "You Won!" << std::endl;
        return 1;
    } else {
        return 0;
    }
}
int main() {
    int         days = 1;
    int         c1   = 0;  //
    int         c2   = 0;  // Show which crystal is collected
    int         c3   = 0;  //
    int         a;
    int         b;
    std::string location = "Enchanted Forest";
    std::string direction;
    while (days <= 30) {
        a = Crystals(location, c1, c2, c3);
        if (a == 1) {
            c1 = 1;
        } else if (a == 2) {
            c2 = 1;
        } else if (a == 3) {
            c3 = 1;
        }
        b = Game_End(location, days);
        if (b == 1) {
            break;
        }
        Game_state(days, location, c1, c2, c3);
        std::cin >> direction;
        if (direction.compare("north") != 0 &&
            direction.compare("south") != 0 && direction.compare("east") != 0 &&
            direction.compare("west") != 0) {
            std::cout << "Invalid Direction!" << std::endl;
            days += 1;
        } else if (location.compare("Enchanted Forest") ==
                   0) {  // When in Enchanted Forest
            if (direction.compare("north") == 0) {
                location = "Marsh of the Undead";
                days += 1;
            } else if (direction.compare("east") == 0) {
                location = "Swamps of Despair";
                days += 2;
            } else if (direction.compare("south") == 0) {
                location = "Vampire Cove";
                days += 3;
            } else if (direction.compare("west") == 0) {
                location = "Bridge of Death";
                days += 1;
            } else {
                std::cout << "Invalid Move!" << std::endl;
                days += 1;
            }
        } else if (location.compare("Marsh of the Undead") ==
                   0) {  // When in Marsh of Undead
            if (direction.compare("east") == 0) {
                location = "Sands of Quick";
                days += 1;
            } else if (direction.compare("south") == 0) {
                location = "Enchanted Forest";
                days += 1;
            } else {
                std::cout << "Invalid Move!" << std::endl;
                days += 1;
            }
        } else if (location.compare("Swamps of Despair") ==
                   0) {  // When in Swamps of Despair
            if (direction.compare("north") == 0) {
                location = "Sands of Quick";
                days += 1;
            } else if (direction.compare("south") == 0) {
                location = "Elven Waterfall";
                days += 1;
            } else if (direction.compare("west") == 0) {
                location = "Enchanted Forest";
                days += 2;
            } else {
                std::cout << "Invaid Move!" << std::endl;
                days += 1;
            }
        } else if (location.compare("Elven Waterfall") ==
                   0) {  // When in Elven Waterfall
            if (direction.compare("north") == 0) {
                location = "Swamps of Despair";
                days += 1;
            } else {
                std::cout << "Invalid Move!" << std::endl;
                days += 1;
            }
        } else if (location.compare("Vampire Cove") ==
                   0) {  // When in Vampire Cove
            if (direction.compare("north") == 0) {
                location = "Enchanted Forest";
                days += 3;
            } else if (direction.compare("south") == 0) {
                location = "Werewolf Hill";
                days += 3;
            } else {
                std::cout << "Invalid Move!" << std::endl;
                days += 1;
            }
        } else if (location.compare("Werewolf Hill") ==
                   0) {  // When in Werewolf Hill
            if (direction.compare("north") == 0) {
                location = "Vampire Cove";
                days += 3;
            } else {
                std::cout << "Invalid Move!" << std::endl;
                days += 1;
            }
        } else if (location.compare("Bridge of Death") ==
                   0) {  // When on Bridge of Death
            if (direction.compare("east") == 0) {
                location = "Enchanted Forest";
                days += 1;
            } else if (direction.compare("west") == 0) {
                if (c1 + c2 + c3 ==
                    3) {  // Check if player has all three crystals
                    location = "Wizard's Castle";
                    days += 5;
                } else {
                    std::cout << "You don't have enought crystals" << std::endl;
                }
            } else {
                std::cout << "Invalid Move!" << std::endl;
                days += 1;
            }
        }
        std::cout << std::endl;
    }
    if (days > 30) {
        std::cout << "Days: " << days << std::endl
                  << "Location: " << location << std::endl
                  << "You die of starvation!" << std::endl
                  << "Game Over!" << std::endl;
    }
}

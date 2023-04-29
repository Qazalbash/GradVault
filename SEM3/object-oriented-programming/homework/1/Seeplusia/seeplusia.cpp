#include "seeplusia.hpp"

#include "mover.hpp"

int    applesLeft = 20, nCrystalsFound = 0, Diamond[4] = {1, 1, 1, 1};
string gameState = "Running", position = "vampire cove";

// This is a demo implementation of makeMove function
// It doesn't follow the game rules at all
// You have to implement it according to game logic

void makeMove(string direction) {
    if (direction == "East") {
        cout << "Provide East move implementation" << endl;
        if (position == "vampire cove" && applesLeft >= 1) {
            moveEast();  // Call this function only if warrior needs to be moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft--;           // taking one apple
            nCrystalsFound +=
                Diamond[0];  // Diamond[0] = marsh of undead's diamond
            Diamond[0] = 0;  // after picking diamonds setting them to be zero
            position   = "marsh of undead";
        } else if (position == "werewolf hill" && applesLeft >= 1) {
            moveEast();  // Call this function only if warrior needs to be moved
            applesLeft--;  // taking one apple
            position  = "sands of quick";
            gameState = "Lost";  // Game over
        } else if (position == "enchanted forest" && applesLeft >= 1) {
            moveEast();  // Call this function only if warrior needs to be moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft--;           // taking one apple
            nCrystalsFound +=
                Diamond[1];  // Diamond[1] = swapms of despair's diamond
            Diamond[1] = 0;  // after picking diamonds setting them to be zero
            position   = "swamps of despair";
        } else if (position == "bridge of death" && applesLeft >= 2) {
            moveEast();  // Call this function only if warrior needs to be moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft -= 2;        // taking two apples
            position = "enchanted forest";
        } else if (position == "elvin waterfall" && applesLeft >= 2) {
            moveEast();  // Call this function only if warrior needs to be moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft -= 2;        // taking two apples
            position = "werewolf hill";
        } else if (position == "eisten tunnel" && applesLeft >= 2) {
            moveEast();  // Call this function only if warrior needs to be moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft -= 2;        // taking two apples
            position = "elvin waterfall";
        } else if (position == "apples orchid" && applesLeft >= 1) {
            moveEast();  // Call this function only if warrior needs to be moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft--;           // taking one apple
            position = "vampire cove";
        } else {
            cout << "Invalid move" << endl;
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft--;           // taking one apple as penalty
        }
    } else if (direction == "West") {
        cout << "Provide West move implementation" << endl;
        if (position == "bridge of death" && nCrystalsFound == 4 &&
            applesLeft >= 5) {
            moveWest();  // Call this function only if warrior needs to be moved
            gameState = "Won";  // Game won
            applesLeft -= 5;    // taking five apples
            position = "wizard castle";
        } else if (position == "enchanted forest" && applesLeft >= 2) {
            moveWest();  // Call this function only if warrior needs to be moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft -= 2;        // taking two apples
            position = "bridge of death";
        } else if (position == "swamps of despair" && applesLeft >= 1) {
            moveWest();  // Call this function only if warrior needs to be moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft--;           // taking one apple
            position = "enchanted forest";
        } else if (position == "marsh of undead") {
            moveWest();  // Call this function only if warrior needs to be moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft--;           // taking one apple
            position = "vampire cove";
        } else if (position == "vampire cove" && applesLeft >= 1) {
            moveWest();  // Call this function only if warrior needs to be moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft += 5;
            position = "apples orchid";
        } else if (position == "werewolf hill" && applesLeft >= 2) {
            moveWest();  // Call this function only if warrior needs to be moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft -= 2;        // taking two apples
            position = "elvin waterfall";
        } else if (position == "elvin waterfall" && applesLeft >= 2) {
            moveWest();  // Call this function only if warrior needs to be moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft -= 2;        // taking two apples
            nCrystalsFound +=
                Diamond[2];  // Diamond[2] = eisten tunnel's diamond
            Diamond[2] = 0;  // after picking diamonds setting them to be zero
            position   = "eisten tunnel";
        } else {
            cout << "Invalid move" << endl;
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft--;           // taking one apple as penalty
        }
    } else if (direction == "North") {
        cout << "Provide North move implementation" << endl;
        if (position == "werewolf hill" && applesLeft >= 3) {
            moveNorth();  // Call this function only if warrior needs to be
                          // moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft -= 3;        // taking three apples
            position = "vampire cove";
        } else if (position == "vampire cove" && applesLeft >= 3) {
            moveNorth();  // Call this function only if warrior needs to be
                          // moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft -= 3;        // taking three apples
            position = "enchanted forest";
        } else if (position == "eisten tunnel" && applesLeft >= 10 &&
                   nCrystalsFound == 4) {
            moveNorth();       // Call this function only if warrior needs to be
                               // moved
            moveNorth();       // Call this function only if warrior needs to be
                               // moved
            applesLeft -= 10;  // taking ten apples
            position  = "wizard castle";
            gameState = "Won";  // Game won
        } else {
            cout << "Invalid move" << endl;
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft--;           // taking one apple as penalty
        }
    } else if (direction == "South") {
        cout << "Provide South move implementation" << endl;
        if (position == "enchanted forest" && applesLeft >= 3) {
            moveSouth();  // Call this function only if warrior needs to be
                          // moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft -= 3;        // taking three apples
            position = "vampire cove";
        } else if (position == "vampire cove" && applesLeft >= 3) {
            moveSouth();  // Call this function only if warrior needs to be
                          // moved
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft -= 3;        // taking three apples
            nCrystalsFound +=
                Diamond[3];  // Diamond[3] = werewolf hill's diamond
            Diamond[3] = 0;  // after picking diamonds setting them to be zero
            position   = "werewolf hill";
        } else if (position == "marsh of undead") {
            moveSouth();  // Call this function only if warrior needs to be
                          // moved
            gameState = "Lost";  // Game over
            applesLeft--;        // taking one apple
            position = "sands of quick";
        } else {
            cout << "Invalid move" << endl;
            gameState = "Running";  // Set this gameState when the game is lost
            applesLeft--;           // taking one apple as penalty
        }
    } else if (applesLeft <= 0) {
        gameState = "Lost";  // Game over
    }
}
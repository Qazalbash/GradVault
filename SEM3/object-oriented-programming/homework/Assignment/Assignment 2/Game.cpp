#include "Game.hpp"

#include <iostream>

#include "Location.hpp"
#include "Map.hpp"
#include "Player.hpp"
#include "Road.hpp"

Game::Game() {
    m_days      = 0;
    m_game_over = false;
    m_player.set_location(m_map.get_start_location());
}

void Game::execute_special(Location *m_location) {
    if (m_location->get_special() == "crystal") {
        m_player.add_crystal();
        m_location->remove_special("crystal");
        std::cout << "You have found a CRYSTAL!!!!" << std::endl;
    } else if (m_location->get_special() == "prince") {
        m_location->remove_special("prince");
        std::cout << "You have rescued the PRINCE!!!!" << std::endl
                  << "You WON" << std::endl;
        m_game_over = true;
    } else if (m_location->get_special() == "death") {
        m_game_over = true;
        std::cout << "You are STUCK!!!!" << std::endl
                  << "GAME OVER" << std::endl;
    }
}

void Game::show_state()  // cout the game state and check if  location has any
                         // special
{
    std::cout << "Days Passed: " << m_days << std::endl;
    execute_special(m_player.get_location());
    std::cout << "Current Location: " << m_player.get_location()->get_name()
              << std::endl;
    std::cout << "Crystals Collected: " << m_player.get_crystals() << std::endl;
}

void Game::run() {
    std::string direction_inp;
    while (m_game_over == false)  // runs while player loses or wins
    {
        show_state();
        if (m_game_over == true) {
            break;
        }
        std::cout << "In which direction you want to move?     ";
        std::getline(std::cin, direction_inp);
        if (direction_inp != "north" && direction_inp != "south" &&
            direction_inp != "east" &&
            direction_inp != "west") {  // checks if the direction entered by
                                        // the user if valid
            m_days += 1;
            std::cout << "INVALID DIRECTION" << std::endl;
        } else {
            // if a valid direction is entered
            Road *path_road;
            path_road = m_map.get_road(m_player.get_location(), direction_inp);
            if (path_road == nullptr) {  // checks if the move in the specified
                                         // valid direction is possible
                m_days += 1;
                std::cout << "Move not POSSIBLE!!!!" << std::endl;
            } else if (path_road != nullptr &&
                       (path_road->get_end(m_player.get_location())
                                ->get_special() != "prince" ||
                        m_player.get_crystals() == 3)) {
                m_days += path_road->get_days();  // increments days
                m_player.set_location(path_road->get_end(
                    m_player.get_location()));  // changes location
            } else  // if player moves to the prince's location without enough
                    // crystals.
            {
                std::cout << "You don't have ENOUGH CRYSTALS!!!" << std::endl;
            }
        }
        std::cout << std::endl;
        if (m_days > 30 && m_game_over == false)  // after 30 days.
        {
            std::cout << "You DIED of HUNGER!!!!" << std::endl
                      << "GAME OVER" << std::endl;
            m_game_over = true;
        }
    }
}
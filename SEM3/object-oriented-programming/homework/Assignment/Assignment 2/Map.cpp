#include "Map.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include "Location.hpp"
#include "Player.hpp"
#include "Road.hpp"
using namespace std;

Map::Map() {
    set_map();  // setting up the entire map
}

Map::~Map() {
    delete[] Map_locations;  // deleting the array
}

void Map::set_map() {
    ifstream input_file;
    string   temp;
    int      location_count = -1;
    string   direction;
    int      direction_id;
    string   map_n = "";
    while (map_n != "standard" && map_n != "custom")  // asking user for map
    {
        cout << "Which map you want to play? (standard/custom)      ";
        getline(cin, map_n);
    }
    cout << endl;
    if (map_n == "standard") {
        input_file.open("seeplusia.txt");
    } else {
        input_file.open("custom.txt");
    }
    while (getline(input_file, temp))  // opening text file
    {
        if (temp == "special : start")  // sets start of map
        {
            m_start = Map_locations + location_count;
        } else if (location_count == -1)  // declaring array of Locations
        {
            n_locations    = std::stoi(temp);
            Map_locations  = new Location[n_locations];
            location_count = 0;
        } else if (temp.substr(0, 3) == "id ")  // location id
        {
            location_count = std::stoi(temp.substr(5, 5));
        } else if (temp.substr(0, 3) == "nam")  // location name
        {
            Map_locations[location_count].set_name(temp.substr(7));
        } else if (temp.substr(0, 3) == "nor" &&
                   temp != "north :")  // north road
        {
            direction    = "north";
            direction_id = std::stoi(temp.substr(8, 8));
        } else if (temp.substr(0, 3) == "sou" &&
                   temp != "south :")  // south road
        {
            direction    = "south";
            direction_id = std::stoi(temp.substr(8, 8));
        } else if (temp.substr(0, 3) == "eas" && temp != "east :")  // east road
        {
            direction    = "east";
            direction_id = std::stoi(temp.substr(7, 7));
        } else if (temp.substr(0, 3) == "wes" && temp != "west :")  // west road
        {
            direction    = "west";
            direction_id = std::stoi(temp.substr(7, 7));
        } else if (temp.substr(0, 3) == "day" &&
                   temp != "days :")  // days taken while moving on the road
        {
            if (direction == "north")  // setting up north road
            {
                (Map_locations + location_count)
                    ->set_north(Map_locations + direction_id,
                                std::stoi(temp.substr(7, 7)));
            } else if (direction == "south")  // setting up south road
            {
                (Map_locations + location_count)
                    ->set_south(Map_locations + direction_id,
                                std::stoi(temp.substr(7, 7)));
            } else if (direction == "east")  // setting up east road
            {
                (Map_locations + location_count)
                    ->set_east(Map_locations + direction_id,
                               std::stoi(temp.substr(7, 7)));
            } else if (direction == "west")  // setting up west road
            {
                (Map_locations + location_count)
                    ->set_west(Map_locations + direction_id,
                               std::stoi(temp.substr(7, 7)));
            }
        } else if (temp.substr(0, 3) == "spe" &&
                   temp != "special :")  // special of location
        {
            if (temp.substr(10) ==
                "start")  // if special is start, declaring start of map.
            {
                m_start = Map_locations + location_count;
            } else {
                (Map_locations + location_count)->set_special(temp.substr(10));
            }
        }
    }
}

Road *Map::get_road(Location *location_, std::string direction) {
    // get the road for a specific direction
    if (direction == "north") {
        return location_->get_road_north();
    } else if (direction == "south") {
        return location_->get_road_south();
    } else if (direction == "east") {
        return location_->get_road_east();
    } else if (direction == "west") {
        return location_->get_road_west();
    }
    return nullptr;  // incase function is called with invalid direction
}

Location *Map::get_start_location() { return m_start; }
#pragma once

class Map;
class Road;
class Player;

class Location {
private:

    std::string m_name;  // the name of this location.

    /* The special characteristic of this location, one of
       - start: this is the start location,
       - crystal: a crystal is present here,
       - death: getting here results in the player's death
       - prince: the prince is imprisoned here.
       The string is empty if this location has no special charactersitic.
    */
    std::string m_special;

    // Outgoing roads. A road is set to NULL if there is travel is not possible
    // in that directions.
    Road *m_north, *m_south, *m_east, *m_west;

public:

    Location(std::string name,
             std::string special);  // initialize name and special.
    Location();

    std::string get_name();           // returns this location's name.
    std::string get_special();        // returns this location's special
                                      // characteristic.
    void set_name(std::string name);  // set name of the location
    void set_special(
        std::string special_);  // set special characteristic of a location.
    void remove_special(std::string characteristic);  // removes the given
                                                      // special characteristic.

    // Set a neighbor by setting the corresponding road appropriately.
    void set_north(Location *place, int days_required);
    void set_south(Location *place, int days_required);
    void set_east(Location *place, int days_required);
    void set_west(Location *place, int days_required);

    // Get the road in the specified direction - NULL if no such road exists.
    Road *get_road_north();
    Road *get_road_south();
    Road *get_road_east();
    Road *get_road_west();

    // Get a neighbor - NULL if travel is not possible in that direction.
    Location *get_north();
    Location *get_south();
    Location *get_east();
    Location *get_west();
};

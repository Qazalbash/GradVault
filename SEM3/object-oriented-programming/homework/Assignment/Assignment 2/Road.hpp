#pragma once

class Map;
class Location;
class Player;

class Road {
private:

    Location *m_a, *m_b;  // endpoints of this road.
    int       m_days;     // the number of days to travel this road.

public:

    Road(Location *a, Location *b,
         int days_required);  // initialize attributes.

    Location *get_end(Location *initial);  // get the other end of this road.
    int       get_days();  // get the number of days to travel this road.
};

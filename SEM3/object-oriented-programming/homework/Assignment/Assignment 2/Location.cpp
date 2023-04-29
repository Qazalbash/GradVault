#include "Location.hpp"

#include <iostream>

#include "Map.hpp"
#include "Player.hpp"
#include "Road.hpp"

Location::Location(std::string name,
                   std::string special)  // initializes attributes
{
    m_name = name;
    if (special != "") {
        m_special = special;
    } else {
        m_special = "";
    }
    m_north = nullptr;
    m_south = nullptr;
    m_east  = nullptr;
    m_west  = nullptr;
}

Location::Location() {
    m_north = nullptr;
    m_south = nullptr;
    m_east  = nullptr;
    m_west  = nullptr;
}

std::string Location::get_name() { return m_name; }

std::string Location::get_special() { return m_special; }

void Location::set_name(std::string name) { m_name = name; }

void Location::set_special(std::string special_) { m_special = special_; }

void Location::remove_special(std::string characteristic) {
    if (characteristic == m_special) {
        m_special = "";
    }
}

void Location::set_north(Location *place, int days_required) {
    m_north = new Road(this, place, days_required);
}

void Location::set_south(Location *place, int days_required) {
    m_south = new Road(this, place, days_required);
}

void Location::set_east(Location *place, int days_required) {
    m_east = new Road(this, place, days_required);
}

void Location::set_west(Location *place, int days_required) {
    m_west = new Road(this, place, days_required);
}

Road *Location::get_road_north() { return m_north; }

Road *Location::get_road_south() { return m_south; }

Road *Location::get_road_east() { return m_east; }

Road *Location::get_road_west() { return m_west; }

Location *Location::get_north() {
    if (m_north != nullptr)  // checking if a location exists in the north
    {
        return m_north->get_end(this);
    } else {
        return nullptr;
    }
}

Location *Location::get_south() {
    if (m_south != nullptr)  // checking if a location exists in the south
    {
        return m_south->get_end(this);
    } else {
        return nullptr;
    }
}

Location *Location::get_east() {
    if (m_east != nullptr)  // checking if a location exists in the east
    {
        return m_east->get_end(this);
    } else {
        return nullptr;
    }
}

Location *Location::get_west() {
    if (m_west != nullptr)  // checking if a location exists in the west
    {
        return m_west->get_end(this);
    } else {
        return nullptr;
    }
}

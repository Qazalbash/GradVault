#include "Road.hpp"

#include <iostream>

#include "Location.hpp"
#include "Map.hpp"
#include "Player.hpp"

Road::Road(Location *a, Location *b, int days_required) {
    m_a    = a;
    m_b    = b;
    m_days = days_required;
}

int Road::get_days() { return m_days; }

Location *Road::get_end(Location *initial) {
    if (initial == m_a) {
        return m_b;
    } else {
        return m_a;
    }
}
#include "Player.hpp"

#include <iostream>

#include "Location.hpp"
#include "Map.hpp"
#include "Road.hpp"

Player::Player() { m_crystals = 0; }

void Player::add_crystal() { m_crystals++; }

int Player::get_crystals() { return m_crystals; }

Location *Player::get_location() { return m_location; }

void Player::set_location(Location *x) { m_location = x; }
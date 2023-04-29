#pragma once
#include <cstdlib>
#include <iostream>

#include "truck.hpp"
using namespace std;

class Box {
    // initializing the attributes
    int length, width, height;

public:

    // declearing the methods to be used in box class
    Box();
    int volume();
    // making box class fimilar with Truck class
    friend class Truck;
};
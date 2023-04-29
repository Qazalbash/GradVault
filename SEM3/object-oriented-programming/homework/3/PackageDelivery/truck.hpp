#pragma once
#include "box.hpp"

class Truck {
    // initializing the attributes
    string driver;
    int    petrol;
    int    money;
    int    fullMileage;
    int    emptyMileage;
    Box    box[10];

public:

    // declearing the methods to be used in truck class
    void  load(int numBox);
    void  unload();
    float cost();
    void  update();
    // constructor
    Truck(string driver, int petrol, int money, int fullMileage,
          int emptyMileage);
};